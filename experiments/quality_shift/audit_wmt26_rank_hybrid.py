#!/usr/bin/env python3
"""Independent WMT26 rank-hybrid audit: checksum, replay, metrics, aggregates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_quality_shift_results import _independent_metrics
from audit_wmt26_external import check_saved_means, equal_float
from build_wmt26_transfer_matrices import sha256_file
from rank_hybrid_methods import rank_information_bins
from run_wmt26_external import METRICS, checked_matrix, partition_documents, save_json
from run_wmt26_quality_calibrated import (
    calibrated_systems, standardize_development, vector_kmeans, N_CLUSTERS,
)
from run_wmt26_rank_hybrid import METHODS, STOCHASTIC, select, SOURCE_SHA, HOLDOUT, EXPLORATORY


class RankHybridAuditError(ValueError):
    pass


def audit_language(frame: pd.DataFrame, panel: dict, folder: Path, *,
                   seed: int, splits: int, trials: int, budgets: list[int]):
    saved = pd.read_csv(folder / "trials.csv")
    gap_rows = pd.read_csv(folder / "quality_gap_audit.csv")
    manifests = json.loads((folder / "split_manifest.json").read_text())
    if len(manifests) != splits or len(gap_rows) != splits:
        raise RankHybridAuditError("Missing split manifests or quality audits")
    required = {"split", "method", "budget", "trial", "selected_indices", "weights", *METRICS}
    if not required <= set(saved) or saved.duplicated(["split", "method", "budget", "trial"]).any():
        raise RankHybridAuditError("Incomplete or duplicate trial records")
    names = panel["target_models"] + panel["development_models_by_coverage"]
    scores = frame[names].to_numpy(float)
    ids = frame.sentence_id.astype(str).to_numpy()
    docs = frame.document_id.to_numpy()
    lengths = frame.source_text.astype(str).str.len().to_numpy(dtype=float)
    replayed = independent = 0
    for split in range(splits):
        cal_mask = partition_documents(docs, seed=seed, split=split)
        cal, ev = np.flatnonzero(cal_mask), np.flatnonzero(~cal_mask)
        rank = np.argsort(-scores[cal].mean(axis=0), kind="stable")
        hi, lo, cal_means = calibrated_systems(scores, cal_mask)
        if not np.array_equal(hi, np.sort(rank[:6])) or not np.array_equal(lo, np.sort(rank[-6:])):
            raise RankHybridAuditError("Calibration-only roles incorrect")
        if set(docs[cal]) & set(docs[ev]):
            raise RankHybridAuditError("Cross-split document leakage")
        trace = manifests[split]
        if trace["split"] != split or trace["calibration_ids"] != ids[cal].tolist() \
                or trace["evaluation_ids"] != ids[ev].tolist() \
                or trace["targets_selected_from_calibration"] != [names[i] for i in hi] \
                or trace["development_selected_from_calibration"] != [names[i] for i in lo]:
            raise RankHybridAuditError("Saved document or model identities differ")
        target_eval = scores[ev][:, hi]
        dev_eval = scores[ev][:, lo]
        dev_cal = scores[cal][:, lo]
        ref = target_eval.mean(axis=0)
        normalized = standardize_development(dev_eval, dev_cal)
        cluster_labels = vector_kmeans(normalized, seed=int(
            np.random.SeedSequence([seed, split, 91]).generate_state(1)[0]))
        info_bins, _ = rank_information_bins(dev_eval)
        if trace["cluster_sizes"] != np.bincount(cluster_labels, minlength=N_CLUSTERS).tolist() \
                or trace["rank_utility_bin_sizes"] != np.bincount(info_bins, minlength=5).tolist():
            raise RankHybridAuditError("Development-only clustering/strata mismatch")
        differences = abs(ref[:, None] - ref[None, :])[np.triu_indices(6, 1)]
        expected = {
            "n_calibration_items": len(cal), "n_evaluation_items": len(ev),
            "calibration_mean_target_minus_dev": cal_means[hi].mean() - cal_means[lo].mean(),
            "evaluation_mean_target_minus_dev": ref.mean() - dev_eval.mean(),
            "evaluation_min_target_minus_max_dev": ref.min() - dev_eval.mean(axis=0).max(),
            "evaluation_target_pair_median_gap": np.median(differences),
            "evaluation_target_pairs_gap1": (differences > 1).sum(),
        }
        gap = gap_rows.iloc[split]
        if int(gap["split"]) != split:
            raise RankHybridAuditError("Quality gap split order changed")
        for name, value in expected.items():
            equal_float(gap[name], value, f"calibration gap {name}")
        for mi, method in enumerate(METHODS):
            for budget in budgets:
                rows = saved[(saved.split == split) & (saved.method == method) & (saved.budget == budget)]
                reps = trials if method in STOCHASTIC else 1
                if len(rows) != reps or sorted(rows.trial.astype(int)) != list(range(reps)):
                    raise RankHybridAuditError(f"Missing trials: split {split} {method} {budget}")
                for row in rows.itertuples(index=False):
                    selected = json.loads(row.selected_indices)
                    weights = np.asarray(json.loads(row.weights), dtype=float)
                    if len(selected) != budget or len(set(selected)) != budget \
                            or not set(selected) <= set(ev) or weights.shape != (budget,) \
                            or not np.isfinite(weights).all() or weights.min() <= 0 \
                            or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12):
                        raise RankHybridAuditError("Invalid evaluation-only sampling/weights")
                    ss = np.random.SeedSequence([seed, split, mi, budget, row.trial, 127])
                    local, replay_weights = select(
                        method, dev_eval, lengths[ev], dev_cal.mean(axis=0),
                        normalized, cluster_labels, budget, np.random.default_rng(ss))
                    if selected != ev[local].tolist() \
                            or not np.allclose(weights, replay_weights, rtol=0, atol=1e-12):
                        raise RankHybridAuditError("Target-free selector replay failed")
                    replayed += 1
                    if row.trial == 0:
                        expected_metrics = _independent_metrics(
                            (target_eval[local] * weights[:, None]).sum(axis=0), ref)
                        for name in METRICS:
                            equal_float(getattr(row, name), expected_metrics[name],
                                        f"independent {name}")
                        independent += 1
    split_summary = check_saved_means(saved, ["split", "method", "budget"],
                                      folder / "split_summary.csv")
    check_saved_means(split_summary, ["method", "budget"], folder / "summary.csv")
    if replayed != len(saved):
        raise RankHybridAuditError("Unchecked extra trial rows")
    return {"status": "passed", "checked_selections": int(replayed),
            "independent_metric_vectors": int(independent),
            "checked_split_summaries": int(len(split_summary)),
            "verified_document_splits": int(splits)}


def audit_global(results: Path, config: dict):
    """Check summary identities and independent macro and paired point estimates."""
    joined = pd.read_csv(results / "language_summary.csv")
    reconstructed = []
    for language in config["languages"]:
        language_frame = pd.read_csv(results / "languages" / language / "summary.csv")
        language_frame.insert(0, "language_pair", language)
        reconstructed.append(language_frame)
    combined = pd.concat(reconstructed, ignore_index=True)
    if not combined.equals(joined):
        raise RankHybridAuditError("Global language aggregation inconsistent")
    macro = pd.read_csv(results / "macro_summary.csv")
    paired = pd.read_csv(results / "paired_language_comparisons.csv")
    cohorts = {"holdout12": HOLDOUT, "previous7_exploratory": EXPLORATORY,
               "all19_descriptive": tuple(config["languages"])}
    expected_macro = expected_paired = 0
    for cohort, members in cohorts.items():
        subset = combined[combined.language_pair.isin(members)]
        if subset.empty:
            continue
        baselines = subset[subset.method == "Random"].set_index(["language_pair", "budget"])
        for (method, budget), group in subset.groupby(["method", "budget"], sort=True):
            row = macro[(macro.cohort == cohort) & (macro.method == method) & (macro.budget == budget)]
            if len(row) != 1 or int(row.iloc[0].n_languages) != len(group):
                raise RankHybridAuditError("Macro cohort/method completeness failed")
            expected_macro += 1
            for metric in METRICS:
                equal_float(row.iloc[0][metric], group[metric].mean(), f"macro {metric}")
                delta = []
                for observation in group.itertuples(index=False):
                    base = float(baselines.loc[(observation.language_pair, budget), metric])
                    actual = float(getattr(observation, metric))
                    if np.isfinite(base) and np.isfinite(actual):
                        delta.append(base - actual if metric == "mae" else actual - base)
                row_delta = paired[(paired.cohort == cohort) & (paired.method == method) &
                                   (paired.budget == budget) & (paired.metric == metric)]
                if len(row_delta) != 1 or int(row_delta.iloc[0].n_languages) != len(delta):
                    raise RankHybridAuditError("Paired comparison completeness failed")
                equal_float(row_delta.iloc[0].gain_vs_random,
                            np.mean(delta) if delta else np.nan, "paired delta")
                expected_paired += 1
    if len(macro) != expected_macro or len(paired) != expected_paired:
        raise RankHybridAuditError("Unexpected global summary rows")
    return {"macro_rows": expected_macro, "paired_rows": expected_paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.results / "run_manifest.json").read_text())
    if manifest["status"] != "completed":
        parser.error("Cannot audit incomplete experiment")
    config = manifest["protocol"]
    if sha256_file(args.matrix_dir / "matrix_manifest.json") != config["source_matrix_manifest_sha256"]:
        parser.error("Input matrix manifest changed")
    source = json.loads((args.matrix_dir / "matrix_manifest.json").read_text())
    if source.get("final_annotation_sha256") != SOURCE_SHA:
        parser.error("Unexpected WMT26 source digest")
    panels = {p["language_pair"]: p for p in source["language_panels"]}
    reports = {}
    for language in config["languages"]:
        panel = panels[language]
        frame = checked_matrix(args.matrix_dir / panel["matrix_file"], panel)
        folder = args.results / "languages" / language
        complete = json.loads((folder / "completion.json").read_text())
        if complete["signature"] != manifest["signature"] or any(
            sha256_file(folder / name) != digest for name, digest in complete["files"].items()
        ):
            raise RankHybridAuditError(f"Corrupt or incomplete direction {language}")
        report = audit_language(frame, panel, folder, seed=config["seed"],
                                splits=config["splits"], trials=config["trials"],
                                budgets=config["input_summary"][language]["supported_budgets"])
        save_json(folder / "audit_report.json", report)
        reports[language] = report
        print(f"AUDIT PASSED {language}: {report['checked_selections']} trials", flush=True)
    aggregates = audit_global(args.results, config)
    summary = {"status": "passed", "languages": len(reports),
               "checked_selections": sum(r["checked_selections"] for r in reports.values()),
               "independent_metric_vectors": sum(r["independent_metric_vectors"] for r in reports.values()),
               **aggregates, "language_reports": reports}
    save_json(args.results / "audit_manifest.json", summary)
    print(f"WMT26_RANK_HYBRID_AUDIT_PASSED {summary['checked_selections']}", flush=True)


if __name__ == "__main__":
    main()
