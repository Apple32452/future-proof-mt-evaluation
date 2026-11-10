#!/usr/bin/env python3
"""Exploratory WMT26 calibration-defined stronger-target experiment.

At each disjoint-document split, human judgments on calibration documents
choose six high-score *target* systems and six low-score *development* systems
from the existing 18-system complete-case panel. Held-out target judgments
cannot influence selection. Calibration labels are an additional setup cost.

Alongside frozen WMT25 selectors, test two NEW development-feature samplers:
multi-dimensional mean-balanced coreset and cluster-proportional sampling.
The latter estimates a stratified mean using population-fraction weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from build_wmt26_transfer_matrices import sha256_file
from run_quality_shift_multilingual import bootstrap_interval
from run_quality_shift_pilot import METHODS as LEGACY_METHODS, STOCHASTIC, score_batch, select_indices
from run_wmt26_external import DEFAULT_LANGUAGES, METRICS, PRIMARY, checked_matrix, partition_documents, save_json


HERE = Path(__file__).resolve().parent
METHODS = (*LEGACY_METHODS, "Dev-feature-mean-balance", "Dev-cluster-proportional")
STOCHASTIC_METHODS = STOCHASTIC | {"Dev-cluster-proportional"}
N_TARGET = N_DEV = 6
N_CLUSTERS = 5


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def calibrated_systems(all_scores: np.ndarray, calibration_mask: np.ndarray):
    """No held-out score value is accessed when assigning the two system roles."""
    calibration_means = all_scores[calibration_mask].mean(axis=0)
    if not np.isfinite(calibration_means).all() or len(calibration_means) < 18:
        raise ValueError("Need 18 finite calibration-scored models")
    rank = np.argsort(-calibration_means, kind="stable")
    high = np.sort(rank[:N_TARGET])
    low = np.sort(rank[-N_DEV:])
    if set(high) & set(low):
        raise AssertionError("The target/development panels overlap")
    return high, low, calibration_means


def standardize_development(dev_eval: np.ndarray, dev_cal: np.ndarray):
    center = dev_cal.mean(axis=0)
    spread = np.maximum(dev_cal.std(axis=0), 1.0)
    return (dev_eval - center) / spread


def vector_kmeans(features: np.ndarray, seed: int, k: int = N_CLUSTERS):
    """Deterministic seeded k-means++ with explicit empty-cluster handling."""
    if len(features) < k or features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Invalid multivariate development-score features")
    rng = np.random.default_rng(seed)
    centers = [features[int(rng.integers(len(features)))].copy()]
    min_dist = ((features - centers[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        if min_dist.sum() <= 1e-15:
            index = int(rng.integers(len(features)))
        else:
            index = int(rng.choice(len(features), p=min_dist / min_dist.sum()))
        centers.append(features[index].copy())
        min_dist = np.minimum(min_dist, ((features - centers[-1]) ** 2).sum(axis=1))
    centroids = np.stack(centers)
    for _ in range(15):
        distance = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distance, axis=1)
        updated = np.stack([features[labels == index].mean(axis=0) if (labels == index).any()
                            else centroids[index] for index in range(k)])
        if np.allclose(centroids, updated, atol=1e-10, rtol=0):
            centroids = updated
            break
        centroids = updated
    distance = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distance, axis=1)


def mean_balance_indices(features: np.ndarray, budget: int) -> np.ndarray:
    """Greedy representative subset matching the full vector of dev means."""
    desired = features.mean(axis=0)
    chosen = np.zeros(len(features), dtype=bool)
    picked: list[int] = []
    running = np.zeros(features.shape[1], dtype=float)
    for count in range(budget):
        errors = (((features + running) / (count + 1) - desired) ** 2).sum(axis=1)
        errors[chosen] = np.inf
        index = int(np.argmin(errors))
        picked.append(index)
        chosen[index] = True
        running += features[index]
    return np.asarray(picked, dtype=int)


def cluster_proportional_indices(labels: np.ndarray, budget: int, rng: np.random.Generator):
    """At least one draw per cluster; Horvitz--Thompson cluster weights."""
    strata = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    counts = np.asarray([len(s) for s in strata], dtype=int)
    if budget < len(strata) or budget > counts.sum():
        raise ValueError("Budget must cover every active development cluster")
    allocated = np.ones(len(strata), dtype=int)
    target_counts = budget * counts / counts.sum()
    for _ in range(budget - len(strata)):
        deficit = np.where(allocated < counts, target_counts - allocated, -np.inf)
        allocated[int(np.argmax(deficit))] += 1
    picked = np.concatenate([rng.choice(stratum, size=int(take), replace=False)
                             for stratum, take in zip(strata, allocated)])
    weights = np.concatenate([np.full(int(take), len(stratum) / (len(labels) * int(take)))
                              for stratum, take in zip(strata, allocated)])
    if not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12):
        raise AssertionError("Cluster estimator weights do not sum to one")
    return picked.astype(int), weights


def choose(method: str, dev_eval: np.ndarray, source_lengths: np.ndarray,
           dev_calibration_means: np.ndarray, standardized: np.ndarray,
           labels: np.ndarray, budget: int, rng: np.random.Generator):
    if method == "Dev-feature-mean-balance":
        chosen = mean_balance_indices(standardized, budget)
        weights = np.full(budget, 1 / budget)
    elif method == "Dev-cluster-proportional":
        chosen, weights = cluster_proportional_indices(labels, budget, rng)
    else:
        chosen = select_indices(method, dev_eval, source_lengths, dev_calibration_means,
                                budget, rng)
        weights = np.full(budget, 1 / budget)
    if len(chosen) != budget or len(set(chosen)) != budget:
        raise AssertionError("Selection must have unique indices of the requested size")
    return chosen, weights


def supported_budgets(frame: pd.DataFrame, seed: int, splits: int, requested: list[int]):
    docs = frame.document_id.to_numpy()
    smallest = min((~partition_documents(docs, seed=seed, split=split)).sum()
                   for split in range(splits))
    return [b for b in requested if b <= smallest], int(smallest)


def run_language(frame: pd.DataFrame, panel: dict, *, seed: int, splits: int,
                 trials: int, budgets: list[int], output: Path):
    model_names = panel["target_models"] + panel["development_models_by_coverage"]
    scores = frame[model_names].to_numpy(dtype=float)
    ids = frame.sentence_id.astype(str).to_numpy()
    docs = frame.document_id.to_numpy()
    lengths = frame.source_text.astype(str).str.len().to_numpy(dtype=float)
    records, gap_rows, manifests = [], [], []
    for split in range(splits):
        cal_mask = partition_documents(docs, seed=seed, split=split)
        cal, ev = np.flatnonzero(cal_mask), np.flatnonzero(~cal_mask)
        if not len(cal) or len(ev) < max(budgets) or set(docs[cal]) & set(docs[ev]):
            raise ValueError("Empty/leaking calibration or evaluation split")
        high, low, calibrations = calibrated_systems(scores, cal_mask)
        high_names = [model_names[index] for index in high]
        low_names = [model_names[index] for index in low]
        target_eval = scores[ev][:, high]
        dev_eval = scores[ev][:, low]
        dev_cal = scores[cal][:, low]
        reference = target_eval.mean(axis=0)
        standardized = standardize_development(dev_eval, dev_cal)
        labels = vector_kmeans(standardized, seed=int(np.random.SeedSequence([seed, split, 91]).generate_state(1)[0]))
        target_pair_differences = abs(reference[:, None] - reference[None, :])[np.triu_indices(N_TARGET, 1)]
        manifests.append({"split": split, "calibration_ids": ids[cal].tolist(),
                          "evaluation_ids": ids[ev].tolist(),
                          "targets_selected_from_calibration": high_names,
                          "development_selected_from_calibration": low_names,
                          "cluster_sizes": np.bincount(labels, minlength=N_CLUSTERS).tolist()})
        gap_rows.append({"split": split, "n_calibration_items": len(cal),
                         "n_evaluation_items": len(ev),
                         "calibration_mean_target_minus_dev": float(calibrations[high].mean() - calibrations[low].mean()),
                         "evaluation_mean_target_minus_dev": float(reference.mean() - dev_eval.mean()),
                         "evaluation_min_target_minus_max_dev": float(reference.min() - dev_eval.mean(axis=0).max()),
                         "evaluation_target_pair_median_gap": float(np.median(target_pair_differences)),
                         "evaluation_target_pairs_gap1": int((target_pair_differences > 1).sum())})
        for method_i, method in enumerate(METHODS):
            repeats = trials if method in STOCHASTIC_METHODS else 1
            for budget in budgets:
                for trial in range(repeats):
                    ss = np.random.SeedSequence([seed, split, method_i, budget, trial, 127])
                    chosen, weights = choose(method, dev_eval, lengths[ev], dev_cal.mean(axis=0),
                                              standardized, labels, budget, np.random.default_rng(ss))
                    estimated = (target_eval[chosen] * weights[:, None]).sum(axis=0)
                    measured = score_batch(estimated[None, :], reference)
                    record = {"split": split, "method": method, "budget": budget,
                              "trial": trial, "selected_indices": json.dumps(ev[chosen].tolist()),
                              "weights": json.dumps(weights.tolist())}
                    record.update({name: float(measured[name][0]) for name in METRICS})
                    records.append(record)
        print(f"{panel['language_pair']} split {split + 1}/{splits}: targets={len(high)} "+
              f"dev={len(low)} eval={len(ev)}", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    trials_frame = pd.DataFrame.from_records(records)
    trials_frame.to_csv(output / "trials.csv", index=False)
    split_summary = trials_frame.groupby(["split", "method", "budget"], as_index=False)[list(METRICS)].mean()
    split_summary.to_csv(output / "split_summary.csv", index=False)
    split_summary.groupby(["method", "budget"], as_index=False)[list(METRICS)].mean().to_csv(output / "summary.csv", index=False)
    pd.DataFrame.from_records(gap_rows).to_csv(output / "quality_gap_audit.csv", index=False)
    save_json(output / "split_manifest.json", manifests)


def aggregate(output: Path, languages: list[str], bootstraps: int, seed: int):
    joined = []
    for language in languages:
        frame = pd.read_csv(output / "languages" / language / "summary.csv")
        frame.insert(0, "language_pair", language)
        joined.append(frame)
    joined = pd.concat(joined, ignore_index=True)
    joined.to_csv(output / "language_summary.csv", index=False)
    cohorts = {"wmt25_overlap_primary": [name for name in PRIMARY if name in languages],
               "all_wmt26": languages,
               "additional_directions": [name for name in languages if name not in PRIMARY]}
    macro, paired = [], []
    for cohort, group_names in cohorts.items():
        if not group_names:
            continue
        subset = joined[joined.language_pair.isin(group_names)]
        baseline = subset[subset.method == "Random"].set_index(["language_pair", "budget"])
        for (method, budget), group in subset.groupby(["method", "budget"], sort=True):
            summary = {"cohort": cohort, "method": method, "budget": int(budget),
                       "n_languages": len(group)}
            summary.update({m: float(group[m].mean()) for m in METRICS})
            macro.append(summary)
            for metric in METRICS:
                changes, observed_languages = [], []
                for line in group.itertuples(index=False):
                    b = float(baseline.loc[(line.language_pair, budget), metric])
                    observed = float(getattr(line, metric))
                    if np.isfinite(b) and np.isfinite(observed):
                        changes.append(observed - b)
                        observed_languages.append(line.language_pair)
                lower, upper = bootstrap_interval(np.array(changes), observed_languages, bootstraps, seed)
                paired.append({"cohort": cohort, "method": method, "budget": int(budget),
                               "metric": metric, "n_languages": len(changes),
                               "signed_difference_method_minus_random": float(np.mean(changes)) if changes else np.nan,
                               "ci95_lower": lower, "ci95_upper": upper})
    pd.DataFrame.from_records(macro).to_csv(output / "macro_summary.csv", index=False)
    pd.DataFrame.from_records(paired).to_csv(output / "paired_language_comparisons.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES))
    parser.add_argument("--budgets", nargs="+", type=int, default=[10, 25, 50, 100])
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.languages)) != len(args.languages) or not set(args.languages) <= set(DEFAULT_LANGUAGES):
        parser.error("Choose unique language directions from the seven documented WMT26 panels")
    if len(set(args.budgets)) != len(args.budgets) or any(b <= 0 or b % 5 for b in args.budgets):
        parser.error("Budgets must be unique positive multiples of five")
    if min(args.splits, args.trials, args.bootstrap_replicates) <= 0:
        parser.error("Counts must be strictly positive")
    manifest_path = args.matrix_dir / "matrix_manifest.json"
    source = json.loads(manifest_path.read_text())
    if source.get("final_annotation_sha256") != "6023795cc592864a9d67ed6daaaada283041d0a8c87c4bfb4c3e5ae276c73b28":
        parser.error("Unverified WMT26 annotations")
    panels = {row["language_pair"]: row for row in source["language_panels"]}
    frames, input_summary = {}, {}
    for language in args.languages:
        panel = panels[language]
        frame = checked_matrix(args.matrix_dir / panel["matrix_file"], panel)
        budgets, min_eval = supported_budgets(frame, args.seed, args.splits, args.budgets)
        if 25 not in budgets:
            parser.error(f"Missing primary 25-item budget for {language}")
        input_summary[language] = {"n_items": len(frame), "n_documents": frame.document_id.nunique(),
                                   "min_evaluation_items": min_eval, "supported_budgets": budgets,
                                   "matrix_sha256": panel["matrix_sha256"]}
        frames[language] = frame
    config = {"languages": args.languages, "budgets": args.budgets, "splits": args.splits,
              "trials": args.trials, "seed": args.seed, "bootstrap_replicates": args.bootstrap_replicates,
              "targets_selected_using": "calibration documents only", "methods": list(METHODS),
              "n_target_systems": N_TARGET, "n_development_systems": N_DEV, "n_clusters": N_CLUSTERS,
              "source_matrix_manifest_sha256": sha256_file(manifest_path),
              "code_sha256": {name: sha256_file(HERE / name) for name in
                              ("run_wmt26_quality_calibrated.py", "run_wmt26_external.py",
                               "run_quality_shift_pilot.py", "run_quality_shift_multilingual.py")},
              "input_summary": input_summary}
    signature = fingerprint(config)
    if args.resume:
        old = json.loads((args.out / "run_manifest.json").read_text())
        if old["signature"] != signature:
            parser.error("Different code/data/protocol; cannot resume this run")
    else:
        if args.out.exists():
            parser.error("Use an unused output directory; never overwrite previous experiments")
        (args.out / "languages").mkdir(parents=True)
        save_json(args.out / "run_manifest.json", {"status": "running", "signature": signature, "protocol": config})
        save_json(args.out / "analysis_plan.json", {
            "status": "exploratory follow-up designed after inspecting WMT26 coverage and pilot quality-gap checks",
            "primary": "Macro pairwise ranking accuracy at 25 additional evaluation labels, four WMT25-overlapping directions",
            "new_methods": ["Dev-feature-mean-balance", "Dev-cluster-proportional"],
            "roles": "Six target and six development models assigned using calibration document mean cESA; exclude the middle six",
            "nonleakage": "Evaluation target judgments absent from role and item selection functions",
            "calibration_cost": "Full 18-system calibration judgments assumed available and not included in incremental evaluation-label budgets",
            "limitations": "Complete-case selection, dynamic cESA missingness, retrospective model identity adaptation; no cross-year numeric comparisons",
        })
    for language in args.languages:
        folder = args.out / "languages" / language
        if folder.exists():
            if not args.resume:
                raise ValueError(f"Results already exist: {folder}")
            marker = json.loads((folder / "completion.json").read_text())
            if marker["signature"] != signature or any(sha256_file(folder / filename) != digest
                                                      for filename, digest in marker["files"].items()):
                raise ValueError(f"Corrupted saved language outputs: {folder}")
            print(f"RESUMED {language}: hashes verified", flush=True)
            continue
        run_language(frames[language], panels[language], seed=args.seed, splits=args.splits,
                     trials=args.trials, budgets=input_summary[language]["supported_budgets"], output=folder)
        files = ("trials.csv", "split_summary.csv", "summary.csv", "quality_gap_audit.csv", "split_manifest.json")
        save_json(folder / "completion.json", {"signature": signature,
                  "files": {name: sha256_file(folder / name) for name in files}})
        print(f"COMPLETE {language}", flush=True)
    aggregate(args.out, args.languages, args.bootstrap_replicates, args.seed)
    save_json(args.out / "run_manifest.json", {"status": "completed", "signature": signature,
              "protocol": config, "python": platform.python_version(), "numpy": np.__version__,
              "pandas": pd.__version__})
    print("WMT26_QUALITY_CALIBRATED_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
