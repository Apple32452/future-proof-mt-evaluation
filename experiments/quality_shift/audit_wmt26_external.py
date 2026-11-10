#!/usr/bin/env python3
"""Replay and independently audit the recorded WMT26 external experiment.

Checks *every* chosen item against the frozen selector, every saved trial-0
metric against a scalar SciPy oracle, all split/group aggregations, calibrated
system groups, gap audits, and upstream input/completion hashes.  The selector
replay is not an independent implementation of the six selection heuristics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit_quality_shift_results import _independent_metrics, _independent_group_means
from build_wmt26_transfer_matrices import sha256_file
from run_quality_shift_pilot import METHODS, STOCHASTIC, select_indices
from run_wmt26_external import (
    METRICS, REGIMES, PRIMARY, checked_matrix,
    partition_documents, save_json,
)


class ExternalAuditError(ValueError):
    pass


def equal_float(observed, expected, context: str):
    a, b = float(observed), float(expected)
    if np.isnan(a) and np.isnan(b):
        return
    if not np.isfinite(a) or not np.isfinite(b) or not np.isclose(a, b, atol=1e-9, rtol=1e-9):
        raise ExternalAuditError(f"Incorrect {context}: saved={a}, recomputed={b}")


def check_saved_means(raw: pd.DataFrame, keys: list[str], actual_path: Path):
    expected = _independent_group_means(raw, keys)
    observed = pd.read_csv(actual_path)
    actual_columns = keys + list(METRICS)
    if list(observed) != actual_columns or len(observed) != len(expected):
        raise ExternalAuditError(f"Unexpected aggregate shape: {actual_path}")
    expected = expected.sort_values(keys).reset_index(drop=True)
    observed = observed.sort_values(keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(expected, observed, check_exact=False,
                                      rtol=1e-9, atol=1e-9,
                                      check_dtype=False)
    except AssertionError as exc:
        raise ExternalAuditError(f"Incorrect aggregate values in {actual_path}: {exc}") from exc
    return expected


def audit_language(frame: pd.DataFrame, panel: dict, folder: Path, *,
                   seed: int, splits: int, trials: int, budgets: list[int]) -> dict:
    saved = pd.read_csv(folder / "trials.csv")
    manifests = json.loads((folder / "split_manifest.json").read_text())
    gaps = pd.read_csv(folder / "quality_gap_audit.csv")
    if len(manifests) != splits or len(gaps) != splits * len(REGIMES):
        raise ExternalAuditError(f"Unexpected split count in {folder}")
    columns = {"split", "regime", "method", "budget", "trial", "selected_indices", *METRICS}
    if not columns <= set(saved):
        raise ExternalAuditError(f"Incomplete trial CSV: {folder}")
    if saved.duplicated(["split", "regime", "method", "budget", "trial"]).any():
        raise ExternalAuditError("Duplicate selection trial")

    target = panel["target_models"]
    dev = panel["development_models_by_coverage"]
    targets = frame[target].to_numpy(dtype=float)
    devs = frame[dev].to_numpy(dtype=float)
    ids = frame.sentence_id.astype(str).to_numpy()
    docs = frame.document_id.to_numpy()
    lengths = frame.source_text.astype(str).str.len().to_numpy(dtype=float)
    replayed, metrics_checked, gap_i = 0, 0, 0

    for split in range(splits):
        cal_mask = partition_documents(docs, seed=seed, split=split)
        cal, ev = np.flatnonzero(cal_mask), np.flatnonzero(~cal_mask)
        manifest = manifests[split]
        if manifest["split"] != split or manifest["calibration_ids"] != ids[cal].tolist() \
                or manifest["evaluation_ids"] != ids[ev].tolist() or manifest["fixed_target_systems"] != target:
            raise ExternalAuditError(f"Incorrect WMT26 document/target manifest at split {split}")
        if set(docs[cal]) & set(docs[ev]):
            raise ExternalAuditError(f"Document leakage at split {split}")
        rng = np.random.default_rng(np.random.SeedSequence([seed, split, 10]))
        rng.permutation(np.unique(docs))
        dcal = devs[cal].mean(axis=0)
        ordered = np.argsort(dcal, kind="stable")
        groups = {
            "Lower-six": np.sort(ordered[:6]),
            "Upper-six": np.sort(ordered[-6:]),
            "Random-six": np.sort(rng.choice(len(dev), size=6, replace=False)),
        }
        true_target = targets[ev].mean(axis=0)
        pair_gaps = abs(true_target[:, None] - true_target[None, :])[np.triu_indices(6, 1)]
        for ri, regime in enumerate(REGIMES):
            group = groups[regime]
            observed_names = manifest["development_groups"].get(regime)
            if observed_names != [dev[i] for i in group]:
                raise ExternalAuditError(f"Wrong calibration-only group at split {split}/{regime}")
            dev_ev = devs[ev][:, group]
            dev_eval_means = dev_ev.mean(axis=0)
            gap_row = gaps.iloc[gap_i]
            gap_i += 1
            expected_gap = {
                "calibration_mean_quality_gap": targets[cal].mean() - devs[cal][:, group].mean(),
                "evaluation_mean_quality_gap": true_target.mean() - dev_eval_means.mean(),
                "evaluation_min_target_minus_max_dev": true_target.min() - dev_eval_means.max(),
                "target_pair_median_gap": np.median(pair_gaps),
                "eligible_pairs_gap1": (pair_gaps > 1).sum(),
                "n_calibration_items": len(cal), "n_evaluation_items": len(ev),
            }
            if int(gap_row["split"]) != split or gap_row["regime"] != regime:
                raise ExternalAuditError("Unexpected quality-gap row ordering")
            for name, value in expected_gap.items():
                equal_float(gap_row[name], value, f"quality-gap {name}")

            for mi, method in enumerate(METHODS):
                for budget in budgets:
                    subset = saved[(saved["split"] == split) & (saved.regime == regime) &
                                   (saved.method == method) & (saved.budget == budget)]
                    repetitions = trials if method in STOCHASTIC else 1
                    if len(subset) != repetitions or sorted(subset.trial.astype(int)) != list(range(repetitions)):
                        raise ExternalAuditError(f"Missing/extra trials at {split}/{regime}/{method}/{budget}")
                    for row in subset.itertuples(index=False):
                        selected = json.loads(row.selected_indices)
                        if len(selected) != budget or len(set(selected)) != budget or not set(selected) <= set(ev):
                            raise ExternalAuditError(f"Invalid/duplicate/leaked selection at {split}/{regime}/{method}/{budget}")
                        regime_seed = 0 if method in {"Random", "Source-length-stratified"} else ri
                        ss = np.random.SeedSequence([seed, split, regime_seed, mi, budget, row.trial, 70])
                        expected_local = select_indices(method, dev_ev, lengths[ev], dcal[group],
                                                       budget, np.random.default_rng(ss))
                        if selected != ev[expected_local].tolist():
                            raise ExternalAuditError(f"Unexpected selected indices at {split}/{regime}/{method}/{budget}/{row.trial}")
                        replayed += 1
                        if row.trial == 0:
                            estimate = targets[selected].mean(axis=0)
                            reference = _independent_metrics(estimate, true_target)
                            for metric in METRICS:
                                equal_float(getattr(row, metric), reference[metric],
                                            f"trial-zero {metric}")
                            metrics_checked += 1

    split_rows = check_saved_means(saved, ["split", "regime", "method", "budget"],
                                  folder / "split_summary.csv")
    check_saved_means(split_rows, ["regime", "method", "budget"], folder / "summary.csv")
    if len(saved) != replayed:
        raise ExternalAuditError("Unexpected leftover trial rows")
    return {"status": "passed", "checked_trials": replayed,
            "independent_trial_zero_metric_vectors": metrics_checked,
            "checked_split_summary_rows": len(split_rows),
            "checked_gap_rows": gap_i,
            "methodology": "All selectors replayed; every trial-zero vector recomputed with independent scalar SciPy metric oracle"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.results / "run_manifest.json").read_text())
    if manifest["status"] != "completed":
        parser.error("Cannot audit incomplete external experiment")
    config = manifest["protocol"]
    source = json.loads((args.matrix_dir / "matrix_manifest.json").read_text())
    if source.get("final_annotation_sha256") != "6023795cc592864a9d67ed6daaaada283041d0a8c87c4bfb4c3e5ae276c73b28":
        parser.error("WMT26 final-annotation provenance is missing or changed")
    if sha256_file(args.matrix_dir / "matrix_manifest.json") != config["matrix_manifest_sha256"]:
        parser.error("Changed source matrix manifest")
    panels = {p["language_pair"]: p for p in source["language_panels"]}
    reports = {}
    for language in config["languages"]:
        panel = panels[language]
        frame = checked_matrix(args.matrix_dir / panel["matrix_file"], panel)
        folder = args.results / "languages" / language
        marker = json.loads((folder / "completion.json").read_text())
        if marker["signature"] != manifest["signature"] or any(
            sha256_file(folder / name) != digest for name, digest in marker["files"].items()
        ):
            raise ExternalAuditError(f"Changed/missing completion hashes: {language}")
        budgets = config["input_summary"][language]["supported_budgets"]
        report = audit_language(frame, panel, folder, seed=config["seed"],
                                splits=config["splits"], trials=config["trials"], budgets=budgets)
        save_json(folder / "audit_report.json", report)
        reports[language] = {"sha256": sha256_file(folder / "audit_report.json"), **report}
        print(f"AUDIT PASSED {language}: {report['checked_trials']} selections", flush=True)
    total = sum(row["checked_trials"] for row in reports.values())
    report = {"status": "passed", "languages": len(reports),
              "total_checked_trials": total,
              "total_independent_metric_vectors": sum(r["independent_trial_zero_metric_vectors"] for r in reports.values()),
              "language_reports": reports}
    save_json(args.results / "audit_manifest.json", report)
    print(f"WMT26_AUDIT_PASSED: {len(reports)} languages; {total} saved selections", flush=True)


if __name__ == "__main__":
    main()
