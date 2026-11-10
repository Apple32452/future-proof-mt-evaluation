#!/usr/bin/env python3
"""Frozen rank-aware hybrid WMT26 experiment with unseen-direction holdout.

All sampling receives six lower-calibration development systems, never six
stronger target systems' held-out scores. This is *incremental* evaluation
label budgeting conditional on previously available development annotations.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from build_wmt26_transfer_matrices import sha256_file
from rank_hybrid_methods import rank_aware_hybrid_sample, rank_information_bins
from run_quality_shift_multilingual import bootstrap_interval
from run_quality_shift_pilot import score_batch
from run_wmt26_external import METRICS, checked_matrix, partition_documents, save_json
from run_wmt26_quality_calibrated import (
    calibrated_systems, choose as baseline_choose, fingerprint,
    standardize_development, supported_budgets, vector_kmeans, N_CLUSTERS,
)

HERE = Path(__file__).resolve().parent
HOLDOUT = (
    "ces_Latn---deu_Latn", "eng_Latn---bel_Cyrl", "eng_Latn---ces_Latn",
    "eng_Latn---deu_Latn", "eng_Latn---ekk_Latn", "eng_Latn---hye_Armn",
    "eng_Latn---isl_Latn", "eng_Latn---kaz_Cyrl", "eng_Latn---lij_Latn",
    "eng_Latn---sme_Latn", "eng_Latn---zho_Hans", "eng_Latn---zho_Hant_TW",
)
EXPLORATORY = (
    "eng_Latn---arz_Arab", "eng_Latn---rus_Cyrl", "eng_Latn---jpn_Jpan",
    "eng_Latn---ukr_Cyrl", "eng_Latn---ind_Latn", "eng_Latn---tha_Thai",
    "zho_Hans---jpn_Jpan",
)
METHODS = ("Random", "Hard-only", "All-dev variance", "Difficulty-stratified",
           "Dev-cluster-proportional", "Rank-aware hybrid")
STOCHASTIC = {"Random", "Difficulty-stratified", "Dev-cluster-proportional", "Rank-aware hybrid"}
SOURCE_SHA = "6023795cc592864a9d67ed6daaaada283041d0a8c87c4bfb4c3e5ae276c73b28"


def select(method: str, dev_eval: np.ndarray, source_lengths: np.ndarray,
           dev_calibration_means: np.ndarray, standardized: np.ndarray,
           cluster_labels: np.ndarray, budget: int, rng: np.random.Generator):
    """Target-free method selection, yielding local indices and mean weights."""
    if method == "Rank-aware hybrid":
        chosen, weights, _, _ = rank_aware_hybrid_sample(dev_eval, budget, rng)
    elif method in METHODS:
        chosen, weights = baseline_choose(method, dev_eval, source_lengths,
                                          dev_calibration_means, standardized,
                                          cluster_labels, budget, rng)
    else:
        raise ValueError(f"Unknown selector: {method}")
    if len(chosen) != budget or len(set(chosen)) != budget \
            or np.any(chosen < 0) or np.any(chosen >= len(dev_eval)) \
            or not np.isclose(weights.sum(), 1, atol=1e-12) \
            or not np.isfinite(weights).all() or np.min(weights) <= 0:
        raise AssertionError("Invalid target-blind sampling indices or weights")
    return chosen, weights


def run_language(frame: pd.DataFrame, panel: dict, *, seed: int, splits: int,
                 trials: int, budgets: list[int], output: Path):
    names = panel["target_models"] + panel["development_models_by_coverage"]
    scores = frame[names].to_numpy(float)
    ids = frame.sentence_id.astype(str).to_numpy()
    docs = frame.document_id.to_numpy()
    lengths = frame.source_text.astype(str).str.len().to_numpy(dtype=float)
    records, gaps, manifests = [], [], []
    for split in range(splits):
        cal_mask = partition_documents(docs, seed=seed, split=split)
        cal, ev = np.flatnonzero(cal_mask), np.flatnonzero(~cal_mask)
        if not len(cal) or len(ev) < max(budgets) or set(docs[cal]) & set(docs[ev]):
            raise ValueError("Invalid calibration/evaluation document split")
        hi, lo, means = calibrated_systems(scores, cal_mask)
        target_eval = scores[ev][:, hi]
        dev_eval = scores[ev][:, lo]
        dev_cal = scores[cal][:, lo]
        ref = target_eval.mean(axis=0)
        standardized = standardize_development(dev_eval, dev_cal)
        clusters = vector_kmeans(standardized, seed=int(
            np.random.SeedSequence([seed, split, 91]).generate_state(1)[0]))
        utility_bins, _ = rank_information_bins(dev_eval)
        gaps_between = abs(ref[:, None] - ref[None, :])[np.triu_indices(6, 1)]
        manifests.append({
            "split": split, "calibration_ids": ids[cal].tolist(),
            "evaluation_ids": ids[ev].tolist(),
            "targets_selected_from_calibration": [names[i] for i in hi],
            "development_selected_from_calibration": [names[i] for i in lo],
            "cluster_sizes": np.bincount(clusters, minlength=N_CLUSTERS).tolist(),
            "rank_utility_bin_sizes": np.bincount(utility_bins, minlength=5).tolist(),
        })
        gaps.append({
            "split": split, "n_calibration_items": len(cal),
            "n_evaluation_items": len(ev),
            "calibration_mean_target_minus_dev": float(means[hi].mean() - means[lo].mean()),
            "evaluation_mean_target_minus_dev": float(ref.mean() - dev_eval.mean()),
            "evaluation_min_target_minus_max_dev": float(ref.min() - dev_eval.mean(axis=0).max()),
            "evaluation_target_pair_median_gap": float(np.median(gaps_between)),
            "evaluation_target_pairs_gap1": int((gaps_between > 1).sum()),
        })
        for method_i, method in enumerate(METHODS):
            for budget in budgets:
                repetitions = trials if method in STOCHASTIC else 1
                for trial in range(repetitions):
                    ss = np.random.SeedSequence([seed, split, method_i, budget, trial, 127])
                    picked, weights = select(
                        method, dev_eval, lengths[ev], dev_cal.mean(axis=0),
                        standardized, clusters, budget, np.random.default_rng(ss))
                    estimated = (target_eval[picked] * weights[:, None]).sum(axis=0)
                    metric_values = score_batch(estimated[None, :], ref)
                    records.append({
                        "split": split, "method": method, "budget": budget, "trial": trial,
                        "selected_indices": json.dumps(ev[picked].tolist()),
                        "weights": json.dumps(weights.tolist()),
                        **{m: float(metric_values[m][0]) for m in METRICS},
                    })
        print(f"{panel['language_pair']} split {split+1}/{splits}, eval={len(ev)}", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    frame_trials = pd.DataFrame.from_records(records)
    frame_trials.to_csv(output / "trials.csv", index=False)
    frame_split = frame_trials.groupby(["split", "method", "budget"], as_index=False)[list(METRICS)].mean()
    frame_split.to_csv(output / "split_summary.csv", index=False)
    frame_split.groupby(["method", "budget"], as_index=False)[list(METRICS)].mean().to_csv(
        output / "summary.csv", index=False)
    pd.DataFrame.from_records(gaps).to_csv(output / "quality_gap_audit.csv", index=False)
    save_json(output / "split_manifest.json", manifests)


def aggregate(output: Path, languages: list[str], bootstraps: int, seed: int):
    rows = []
    for language in languages:
        row = pd.read_csv(output / "languages" / language / "summary.csv")
        row.insert(0, "language_pair", language)
        rows.append(row)
    long = pd.concat(rows, ignore_index=True)
    long.to_csv(output / "language_summary.csv", index=False)
    cohorts = {"holdout12": [d for d in HOLDOUT if d in languages],
               "previous7_exploratory": [d for d in EXPLORATORY if d in languages],
               "all19_descriptive": languages}
    macro, paired = [], []
    for cohort, members in cohorts.items():
        if not members:
            continue
        subset = long[long.language_pair.isin(members)]
        random = subset[subset.method == "Random"].set_index(["language_pair", "budget"])
        for (method, budget), group in subset.groupby(["method", "budget"], sort=True):
            summary = {"cohort": cohort, "method": method, "budget": int(budget),
                       "n_languages": len(group)}
            summary.update({m: float(group[m].mean()) for m in METRICS})
            macro.append(summary)
            for metric in METRICS:
                changes, observed = [], []
                for language_row in group.itertuples(index=False):
                    base = float(random.loc[(language_row.language_pair, budget), metric])
                    value = float(getattr(language_row, metric))
                    if np.isfinite(base) and np.isfinite(value):
                        changes.append(base - value if metric == "mae" else value - base)
                        observed.append(language_row.language_pair)
                low, high = bootstrap_interval(np.asarray(changes), observed, bootstraps, seed)
                paired.append({
                    "cohort": cohort, "method": method, "budget": int(budget),
                    "metric": metric, "n_languages": len(changes),
                    "gain_vs_random": float(np.mean(changes)) if changes else np.nan,
                    "ci95_lower": low, "ci95_upper": high,
                })
    pd.DataFrame.from_records(macro).to_csv(output / "macro_summary.csv", index=False)
    pd.DataFrame.from_records(paired).to_csv(output / "paired_language_comparisons.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=list(HOLDOUT + EXPLORATORY))
    parser.add_argument("--budgets", nargs="+", type=int, default=[10, 25, 50, 100])
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.languages or len(set(args.languages)) != len(args.languages) \
            or not set(args.languages) <= set(HOLDOUT + EXPLORATORY):
        parser.error("Languages must be distinct members of the 19 frozen directions")
    if not args.budgets or len(set(args.budgets)) != len(args.budgets) \
            or any(b < 5 or b % 5 for b in args.budgets):
        parser.error("Budgets must be unique positive multiples of five")
    if min(args.splits, args.trials, args.bootstrap_replicates) <= 0:
        parser.error("Counts must be positive")
    manifest_path = args.matrix_dir / "matrix_manifest.json"
    source = json.loads(manifest_path.read_text())
    if source.get("final_annotation_sha256") != SOURCE_SHA:
        parser.error("Source WMT26 human annotations do not match pinned digest")
    panels = {p["language_pair"]: p for p in source["language_panels"]}
    frames, inputs = {}, {}
    for language in args.languages:
        panel = panels[language]
        frame = checked_matrix(args.matrix_dir / panel["matrix_file"], panel)
        budgets, smallest = supported_budgets(frame, args.seed, args.splits, args.budgets)
        if (language in HOLDOUT and 10 not in budgets) or \
                (language in EXPLORATORY and 25 not in budgets):
            parser.error(f"Missing frozen primary budget for {language}")
        frames[language] = frame
        inputs[language] = {
            "n_items": len(frame), "n_documents": int(frame.document_id.nunique()),
            "min_evaluation_items": smallest, "supported_budgets": budgets,
            "matrix_sha256": panel["matrix_sha256"],
        }
    code_names = ("rank_hybrid_methods.py", "run_wmt26_rank_hybrid.py",
                  "run_wmt26_quality_calibrated.py", "run_wmt26_external.py",
                  "run_quality_shift_pilot.py", "run_quality_shift_multilingual.py")
    protocol = {
        "languages": args.languages, "holdout12": list(HOLDOUT),
        "previous7_exploratory": list(EXPLORATORY),
        "budgets": args.budgets, "splits": args.splits, "trials": args.trials,
        "seed": args.seed, "bootstrap_replicates": args.bootstrap_replicates,
        "methods": list(METHODS), "n_target_systems": 6,
        "n_development_systems": 6, "n_clusters": N_CLUSTERS,
        "source_matrix_manifest_sha256": sha256_file(manifest_path),
        "code_sha256": {file: sha256_file(HERE / file) for file in code_names},
        "input_summary": inputs,
    }
    signature = fingerprint(protocol)
    if args.resume:
        old = json.loads((args.out / "run_manifest.json").read_text())
        if old["signature"] != signature:
            parser.error("Protocol/code/data changed; refuse unsafe resume")
    else:
        if args.out.exists():
            parser.error("Output directory already exists; use a new location")
        (args.out / "languages").mkdir(parents=True)
        save_json(args.out / "run_manifest.json", {
            "status": "running", "signature": signature, "protocol": protocol})
        save_json(args.out / "analysis_plan.json", {
            "protocol_reference": "experiments/quality_shift/WMT26_RANK_HYBRID_PROTOCOL.md",
            "status": "frozen before evaluating 12 new direction outcomes",
            "primary": "holdout12, budget10 macro pairwise_accuracy, 20 document-disjoint splits",
            "secondary": "previous7, budget25; and all other feasible budgets descriptive",
            "practical_gain_threshold": 0.02,
            "comparison_methods": list(METHODS),
            "calibration": "Top six targets and bottom six development models from calibration documents only",
            "selection": "Six development systems' scores on evaluation pool only; full pool observed",
            "label_cost": "Incremental strong-target judgments; excludes setup/development annotations",
            "limitations": "Nonrandom complete-case cESA observation and shared WMT26 data",
        })
    for language in args.languages:
        folder = args.out / "languages" / language
        if folder.exists():
            if not args.resume:
                raise ValueError("Language folder already exists without --resume")
            marker = json.loads((folder / "completion.json").read_text())
            if marker["signature"] != signature or any(
                sha256_file(folder / f) != digest for f, digest in marker["files"].items()
            ):
                raise ValueError(f"Corrupted checkpoint {language}")
            print(f"RESUMED {language}: hashes verified", flush=True)
            continue
        run_language(frames[language], panels[language], seed=args.seed,
                     splits=args.splits, trials=args.trials,
                     budgets=inputs[language]["supported_budgets"], output=folder)
        tracked = ("trials.csv", "split_summary.csv", "summary.csv",
                   "quality_gap_audit.csv", "split_manifest.json")
        save_json(folder / "completion.json", {"signature": signature,
                  "files": {name: sha256_file(folder / name) for name in tracked}})
        print(f"COMPLETE {language}", flush=True)
    aggregate(args.out, args.languages, args.bootstrap_replicates, args.seed)
    save_json(args.out / "run_manifest.json", {
        "status": "completed", "signature": signature, "protocol": protocol,
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__,
    })
    print("WMT26_RANK_HYBRID_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
