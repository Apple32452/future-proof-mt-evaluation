#!/usr/bin/env python3
"""Independent document/selection/metric/aggregation replay of WMT24 pairvar."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_quality_shift_results import _independent_metrics
from audit_wmt26_external import check_saved_means, equal_float
from build_wmt26_transfer_matrices import sha256_file
from pairvar_sampling import allocated_counts
from run_quality_shift_multilingual import bootstrap_interval
from run_wmt26_external import METRICS, save_json
from run_wmt26_quality_calibrated import vector_kmeans
from run_wmt24_pairvar import (
    METHODS, STOCHASTIC, aggregate, choose_roles, load_panels,
    make_select_context, select, select_seed, split_three_document_sets,
)


class PairvarAuditError(ValueError):
    pass


def same_vector(observed, expected, name: str):
    if not np.allclose(np.asarray(observed, dtype=float), np.asarray(expected, dtype=float),
                       atol=1e-12, rtol=0, equal_nan=True):
        raise PairvarAuditError(f"Incorrect {name}")


def audit_language(frame: pd.DataFrame, names: list[str], folder: Path, *,
                   seed: int, splits: int, trials: int, budgets: list[int]):
    saved = pd.read_csv(folder / "trials.csv")
    gaps = pd.read_csv(folder / "quality_gap_audit.csv")
    manifests = json.loads((folder / "split_manifest.json").read_text())
    if len(gaps) != splits or len(manifests) != splits:
        raise PairvarAuditError("Incomplete document-split manifests or calibration gaps")
    if not {"split", "method", "budget", "trial", "selected_indices", "weights", *METRICS} <= set(saved):
        raise PairvarAuditError("Missing required trial columns")
    if saved.duplicated(["split", "method", "budget", "trial"]).any():
        raise PairvarAuditError("Duplicate trial record")
    scores = frame[names].to_numpy(float)
    ids = frame.sentence_id.astype(str).to_numpy()
    docs = frame.document_id.astype(str).to_numpy()
    seen, independent = 0, 0
    for split in range(splits):
        role, variance, evaluation = split_three_document_sets(docs, seed, split)
        trace = manifests[split]
        if trace["split"] != split or trace["role_calibration_ids"] != ids[role].tolist() \
                or trace["variance_calibration_ids"] != ids[variance].tolist() \
                or trace["evaluation_ids"] != ids[evaluation].tolist():
            raise PairvarAuditError("Document-ID split mismatch or cross-document leakage")
        rank = np.argsort(-scores[role].mean(axis=0), kind="stable")
        high, low, _ = choose_roles(scores[role])
        if not np.array_equal(high, np.sort(rank[:4])) or not np.array_equal(low, np.sort(rank[-4:])):
            raise PairvarAuditError("Roles depend on non-role-calibration scores")
        target, dev, means, dev_eval, dev_var, design, standardized = make_select_context(
            scores, role, variance, evaluation)
        if not np.array_equal(high, target) or not np.array_equal(low, dev) \
                or trace["targets_selected_from_role_calibration"] != [names[i] for i in target] \
                or trace["development_selected_from_role_calibration"] != [names[i] for i in dev]:
            raise PairvarAuditError("Calibration-selected target/development model mismatch")
        clusters = vector_kmeans(standardized, seed=int(np.random.SeedSequence(
            [seed, split, 91]).generate_state(1)[0]))
        if trace["stratum_sizes"] != design["sizes"].tolist() or \
                trace["variance_calibration_stratum_sizes"] != np.bincount(
                    design["calibration_labels"], minlength=3).tolist() or \
                trace["cluster_sizes"] != np.bincount(clusters, minlength=5).tolist():
            raise PairvarAuditError("Development-score stratum/cluster composition changed")
        for field, expected in (("difficulty_tertile_boundaries", design["boundaries"]),
                                ("pair_importance", design["pair_weights"]),
                                ("pairwise_dispersion", design["dispersion"])):
            same_vector(trace[field], expected, field)
        for budget in budgets:
            for method in ("Cal-pairvar-proportional", "Cal-pairvar-Neyman"):
                spread = design["dispersion"] if method == "Cal-pairvar-Neyman" else None
                expected = allocated_counts(design["sizes"], budget, spread)
                if trace["allocations"][str(budget)][method] != expected.tolist():
                    raise PairvarAuditError("Pairwise Neyman integer allocation has changed")
        target_eval = scores[evaluation][:, target]
        ref = target_eval.mean(axis=0)
        aa, bb = np.triu_indices(4, 1)
        delta = abs(ref[aa] - ref[bb])
        expected_gap = {
            "n_role_cal_items": len(role), "n_variance_cal_items": len(variance),
            "n_evaluation_items": len(evaluation),
            "calibration_mean_target_minus_dev": means[target].mean() - means[dev].mean(),
            "evaluation_mean_target_minus_dev": ref.mean() - dev_eval.mean(),
            "evaluation_min_target_minus_max_dev": ref.min() - dev_eval.mean(axis=0).max(),
            "evaluation_target_pair_median_gap": np.median(delta),
            "evaluation_target_pairs_gap1": (delta > 1).sum(),
        }
        observed_gap = gaps.iloc[split]
        if int(observed_gap["split"]) != split:
            raise PairvarAuditError("Out-of-order quality-gap audit")
        for key, value in expected_gap.items():
            equal_float(observed_gap[key], value, f"quality gap {key}")
        for mi, method in enumerate(METHODS):
            for budget in budgets:
                subset = saved[(saved.split == split) & (saved.method == method) &
                               (saved.budget == budget)]
                required = trials if method in STOCHASTIC else 1
                if len(subset) != required or sorted(subset.trial.astype(int)) != list(range(required)):
                    raise PairvarAuditError("Missing method/budget/trial records")
                for row in subset.itertuples(index=False):
                    selected = json.loads(row.selected_indices)
                    weights = np.asarray(json.loads(row.weights), dtype=float)
                    if len(selected) != budget or len(set(selected)) != budget \
                            or not set(selected) <= set(evaluation) or weights.shape != (budget,) \
                            or not np.isfinite(weights).all() or weights.min() <= 0 \
                            or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12):
                        raise PairvarAuditError("A selected target item is missing or belongs to calibration")
                    # Reproduce the target-free selection using only frozen calibration
                    # and development information, including independent RNG streams.
                    local, expected_weights = select(
                        method, dev_eval, dev_var, standardized, clusters, design,
                        budget, select_seed(seed, split, mi, budget, int(row.trial)))
                    if selected != evaluation[local].tolist():
                        raise PairvarAuditError("Selected item IDs do not replay")
                    same_vector(weights, expected_weights, "Horvitz--Thompson weights")
                    seen += 1
                    if int(row.trial) == 0:
                        estimated = (target_eval[local] * weights[:, None]).sum(axis=0)
                        measured = _independent_metrics(estimated, ref)
                        for metric in METRICS:
                            equal_float(getattr(row, metric), measured[metric],
                                        f"independent trial0 metric {metric}")
                        independent += 1
    if seen != len(saved):
        raise PairvarAuditError("Extra unverified trial rows")
    split_means = check_saved_means(saved, ["split", "method", "budget"],
                                    folder / "split_summary.csv")
    check_saved_means(split_means, ["method", "budget"], folder / "summary.csv")
    return {"status": "passed", "checked_selections": seen,
            "independent_metric_vectors": independent,
            "verified_document_splits": splits,
            "checked_split_summaries": len(split_means)}


def audit_global(out: Path, config: dict):
    joined = pd.read_csv(out / "language_summary.csv")
    expected = []
    for cohort in config["selected_cohorts"]:
        for language in config["cohorts"][cohort]:
            row = pd.read_csv(out / "languages" / cohort / language / "summary.csv")
            row.insert(0, "language_pair", language)
            row.insert(0, "cohort", cohort)
            expected.append(row)
    union = pd.concat(expected, ignore_index=True)
    try:
        pd.testing.assert_frame_equal(joined, union, check_exact=False,
                                      rtol=1e-9, atol=1e-9, check_dtype=False)
    except AssertionError as exc:
        raise PairvarAuditError("Incorrect global language means") from exc
    macro = pd.read_csv(out / "macro_summary.csv")
    paired = pd.read_csv(out / "paired_language_comparisons.csv")
    macros_checked = paired_checked = 0
    for cohort, sub in union.groupby("cohort", sort=False):
        baseline = sub[sub.method == "Random"].set_index(["language_pair", "budget"])
        for (method, budget), group in sub.groupby(["method", "budget"], sort=True):
            found = macro[(macro.cohort == cohort) & (macro.method == method) &
                          (macro.budget == budget)]
            if len(found) != 1 or int(found.iloc[0].n_languages) != len(group):
                raise PairvarAuditError("Missing global macro cell")
            for metric in METRICS:
                equal_float(found.iloc[0][metric], group[metric].mean(), f"macro {metric}")
                differences, langs = [], []
                for langrow in group.itertuples(index=False):
                    base = float(baseline.loc[(langrow.language_pair, budget), metric])
                    val = float(getattr(langrow, metric))
                    if np.isfinite(base) and np.isfinite(val):
                        diff = val - base
                        if metric == "mae":
                            diff = -diff
                        if metric in {"pairwise_accuracy", "pairwise_accuracy_gap1", "top1_accuracy"}:
                            diff *= 100
                        differences.append(diff)
                        langs.append(langrow.language_pair)
                found_pair = paired[(paired.cohort == cohort) &
                                    (paired.method == method) & (paired.budget == budget) &
                                    (paired.metric == metric)]
                if len(found_pair) != 1 or int(found_pair.iloc[0].n_languages) != len(differences):
                    raise PairvarAuditError("Missing paired-language comparison")
                row = found_pair.iloc[0]
                equal_float(row.difference_vs_random,
                            np.mean(differences) if differences else np.nan,
                            f"paired {metric}")
                low, high = bootstrap_interval(np.asarray(differences), langs,
                    config["bootstrap_replicates"], config["seed"])
                equal_float(row.ci95_lower, low, f"bootstrap lower {metric}")
                equal_float(row.ci95_upper, high, f"bootstrap upper {metric}")
                paired_checked += 1
            macros_checked += 1
    if macros_checked != len(macro) or paired_checked != len(paired):
        raise PairvarAuditError("Unverified extra global result rows")
    return {"verified_macro_cells": macros_checked,
            "verified_paired_comparisons": paired_checked}


def audit_results(wmt24: Path, wmt26: Path, results: Path):
    manifest = json.loads((results / "run_manifest.json").read_text())
    config = manifest["config"]
    from run_wmt26_quality_calibrated import fingerprint
    if fingerprint(config) != manifest["signature"]:
        raise PairvarAuditError("Run-manifest signature changed")
    here = Path(__file__).resolve().parent
    for filename, digest in config["code_sha256"].items():
        if sha256_file(here / filename) != digest:
            raise PairvarAuditError(f"Protocol/experiment code digest changed: {filename}")
    panels, hashes, _, inputs = load_panels(
        wmt24, wmt26, seed=config["seed"], splits=config["splits"],
        budgets=config["budgets"], selected_cohorts=config["selected_cohorts"])
    if hashes != config["source_matrix_manifest_sha256"]:
        raise PairvarAuditError("Source matrix manifests differ from run")
    for name, meta in inputs.items():
        if meta != config["input_summary"][name]:
            raise PairvarAuditError("Source matrix inventory changed")
    reports = {}
    for cohort in config["selected_cohorts"]:
        for language in config["cohorts"][cohort]:
            frame, model_names = panels[(cohort, language)]
            folder = results / "languages" / cohort / language
            completion = json.loads((folder / "completion.json").read_text())
            if completion.get("status") != "complete" or any(
                    sha256_file(folder / name) != digest for name, digest in
                    completion["artifact_sha256"].items()):
                raise PairvarAuditError("Incomplete or modified experiment artifact")
            report = audit_language(
                frame, model_names, folder, seed=config["seed"],
                splits=config["splits"], trials=config["trials"],
                budgets=inputs[f"{cohort}/{language}"]["supported_budgets"])
            save_json(folder / "audit_report.json", report)
            reports[f"{cohort}/{language}"] = report
            print(f"AUDIT PASSED {cohort}/{language}: "
                  f"{report['checked_selections']} selections", flush=True)
    global_report = audit_global(results, config)
    output = {"status": "passed", "n_languages": len(reports),
              "replayed_selections": sum(x["checked_selections"] for x in reports.values()),
              "independent_trial0_metric_vectors": sum(
                  x["independent_metric_vectors"] for x in reports.values()),
              **global_report, "reports_by_language": reports}
    save_json(results / "audit_manifest.json", output)
    print("WMT24_PAIRVAR_AUDIT_PASSED", output["replayed_selections"], flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wmt24-matrix-dir", type=Path, required=True)
    parser.add_argument("--wmt26-matrix-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    audit_results(args.wmt24_matrix_dir, args.wmt26_matrix_dir, args.results)


if __name__ == "__main__":
    main()
