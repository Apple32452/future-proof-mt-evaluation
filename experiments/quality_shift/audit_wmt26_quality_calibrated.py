#!/usr/bin/env python3
"""Independent metric / aggregate audit and complete selector replay.

This audits WMT26 quality-calibrated runs but shares selector code with the
runner; the scalar SciPy metric oracle and aggregation are independent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_quality_shift_results import _independent_metrics
from audit_wmt26_external import check_saved_means, equal_float
from build_wmt26_transfer_matrices import sha256_file
from run_wmt26_external import METRICS, checked_matrix, partition_documents, save_json
from run_wmt26_quality_calibrated import (
    METHODS, N_CLUSTERS, STOCHASTIC_METHODS, calibrated_systems, choose,
    standardize_development, vector_kmeans,
)


class QualityCalibratedAuditError(ValueError):
    pass


def audit_language(frame: pd.DataFrame, panel: dict, folder: Path, *,
                   seed: int, splits: int, trials: int, budgets: list[int]):
    saved = pd.read_csv(folder / "trials.csv")
    saved_gaps = pd.read_csv(folder / "quality_gap_audit.csv")
    manifests = json.loads((folder / "split_manifest.json").read_text())
    if len(manifests) != splits or len(saved_gaps) != splits:
        raise QualityCalibratedAuditError("Missing document split manifests or gap audits")
    required = {"split", "method", "budget", "trial", "selected_indices", "weights", *METRICS}
    if not required <= set(saved):
        raise QualityCalibratedAuditError("Missing trial metrics or selected item trace")
    if saved.duplicated(["split", "method", "budget", "trial"]).any():
        raise QualityCalibratedAuditError("Duplicate selection trial")
    names = panel["target_models"] + panel["development_models_by_coverage"]
    scores = frame[names].to_numpy(float)
    docs = frame.document_id.to_numpy()
    ids = frame.sentence_id.astype(str).to_numpy()
    lengths = frame.source_text.astype(str).str.len().to_numpy(dtype=float)
    total_replays = total_independent = 0
    for split in range(splits):
        cal_mask = partition_documents(docs, seed=seed, split=split)
        cal, ev = np.flatnonzero(cal_mask), np.flatnonzero(~cal_mask)
        ranks = np.argsort(-scores[cal].mean(axis=0), kind="stable")
        expected_hi, expected_lo = np.sort(ranks[:6]), np.sort(ranks[-6:])
        hi, lo, calibration_means = calibrated_systems(scores, cal_mask)
        if not np.array_equal(expected_hi, hi) or not np.array_equal(expected_lo, lo):
            raise QualityCalibratedAuditError("Non-calibration-only system assignment")
        if set(docs[cal]) & set(docs[ev]):
            raise QualityCalibratedAuditError("Document leakage")
        manifest = manifests[split]
        if manifest["split"] != split or manifest["calibration_ids"] != ids[cal].tolist() \
                or manifest["evaluation_ids"] != ids[ev].tolist() \
                or manifest["targets_selected_from_calibration"] != [names[i] for i in hi] \
                or manifest["development_selected_from_calibration"] != [names[i] for i in lo]:
            raise QualityCalibratedAuditError("Incorrect calibration-only identity manifest")
        target_eval = scores[ev][:, hi]
        dev_eval = scores[ev][:, lo]
        dev_cal = scores[cal][:, lo]
        reference = target_eval.mean(axis=0)
        standardized = standardize_development(dev_eval, dev_cal)
        labels = vector_kmeans(standardized,
                               seed=int(np.random.SeedSequence([seed, split, 91]).generate_state(1)[0]))
        if manifest["cluster_sizes"] != np.bincount(labels, minlength=N_CLUSTERS).tolist():
            raise QualityCalibratedAuditError("Clustering changed")
        gap = saved_gaps.iloc[split]
        differences = abs(reference[:, None] - reference[None, :])[np.triu_indices(6, 1)]
        expected = {
            "n_calibration_items": len(cal), "n_evaluation_items": len(ev),
            "calibration_mean_target_minus_dev": calibration_means[hi].mean() - calibration_means[lo].mean(),
            "evaluation_mean_target_minus_dev": reference.mean() - dev_eval.mean(),
            "evaluation_min_target_minus_max_dev": reference.min() - dev_eval.mean(axis=0).max(),
            "evaluation_target_pair_median_gap": np.median(differences),
            "evaluation_target_pairs_gap1": (differences > 1).sum(),
        }
        if int(gap["split"]) != split:
            raise QualityCalibratedAuditError("Quality gap split ordering changed")
        for key, value in expected.items():
            equal_float(gap[key], value, f"calibration-only quality gap {key}")
        for mi, method in enumerate(METHODS):
            for budget in budgets:
                rows = saved[(saved.split == split) & (saved.method == method) & (saved.budget == budget)]
                repetitions = trials if method in STOCHASTIC_METHODS else 1
                if len(rows) != repetitions or sorted(rows.trial.astype(int)) != list(range(repetitions)):
                    raise QualityCalibratedAuditError("Missing/extra selection trials")
                for row in rows.itertuples(index=False):
                    selected = json.loads(row.selected_indices)
                    weights = np.asarray(json.loads(row.weights), dtype=float)
                    if len(selected) != budget or len(set(selected)) != budget or not set(selected) <= set(ev):
                        raise QualityCalibratedAuditError("Duplicated, ineligible, or leaked selection")
                    if weights.shape != (budget,) or not np.isfinite(weights).all() \
                            or np.min(weights) <= 0 or not np.isclose(weights.sum(), 1, atol=1e-12):
                        raise QualityCalibratedAuditError("Invalid evaluation-sampling weights")
                    ss = np.random.SeedSequence([seed, split, mi, budget, row.trial, 127])
                    local, replayed_weights = choose(method, dev_eval, lengths[ev],
                                                      dev_cal.mean(axis=0), standardized, labels,
                                                      budget, np.random.default_rng(ss))
                    if selected != ev[local].tolist() or not np.allclose(weights, replayed_weights,
                                                                          rtol=0, atol=1e-12):
                        raise QualityCalibratedAuditError("Target-free selector replay failed")
                    total_replays += 1
                    if row.trial == 0:
                        weighted_estimate = (target_eval[local] * weights[:, None]).sum(axis=0)
                        oracle = _independent_metrics(weighted_estimate, reference)
                        for name in METRICS:
                            equal_float(getattr(row, name), oracle[name], f"independent {name}")
                        total_independent += 1
    split_means = check_saved_means(saved, ["split", "method", "budget"], folder / "split_summary.csv")
    check_saved_means(split_means, ["method", "budget"], folder / "summary.csv")
    if total_replays != len(saved):
        raise QualityCalibratedAuditError("Unchecked extra trial rows")
    return {"status": "passed", "checked_selections": int(total_replays),
            "independent_metric_vectors": int(total_independent),
            "checked_split_aggregates": int(len(split_means)),
            "calibration_document_splits": splits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.results / "run_manifest.json").read_text())
    if manifest["status"] != "completed":
        parser.error("Cannot audit an unfinished WMT26 experiment")
    config = manifest["protocol"]
    if sha256_file(args.matrix_dir / "matrix_manifest.json") != config["source_matrix_manifest_sha256"]:
        parser.error("Source human-score matrix manifest changed")
    source = json.loads((args.matrix_dir / "matrix_manifest.json").read_text())
    if source.get("final_annotation_sha256") != "6023795cc592864a9d67ed6daaaada283041d0a8c87c4bfb4c3e5ae276c73b28":
        parser.error("Unexpected human-annotation source")
    panels = {row["language_pair"]: row for row in source["language_panels"]}
    reports = {}
    for language in config["languages"]:
        panel = panels[language]
        frame = checked_matrix(args.matrix_dir / panel["matrix_file"], panel)
        folder = args.results / "languages" / language
        completion = json.loads((folder / "completion.json").read_text())
        if completion["signature"] != manifest["signature"] or any(
                sha256_file(folder / filename) != digest for filename, digest in completion["files"].items()):
            raise QualityCalibratedAuditError(f"Incomplete/changed language output: {language}")
        result = audit_language(frame, panel, folder, seed=config["seed"], splits=config["splits"],
                                trials=config["trials"], budgets=config["input_summary"][language]["supported_budgets"])
        save_json(folder / "audit_report.json", result)
        reports[language] = {**result, "audit_report_sha256": sha256_file(folder / "audit_report.json")}
        print(f"AUDIT PASSED {language}: {result['checked_selections']} selections", flush=True)
    summary = {"status": "passed", "languages": len(reports),
               "total_checked_selections": sum(r["checked_selections"] for r in reports.values()),
               "independent_metric_vectors": sum(r["independent_metric_vectors"] for r in reports.values()),
               "language_reports": reports}
    save_json(args.results / "audit_manifest.json", summary)
    print(f"WMT26_QUALITY_CALIBRATED_AUDIT_PASSED: {summary['total_checked_selections']} selections", flush=True)


if __name__ == "__main__":
    main()
