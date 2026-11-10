#!/usr/bin/env python3
"""Frozen WMT24 fresh-data and WMT26 exploratory pairwise-variance experiment."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from build_wmt24_esa_matrices import LANGUAGES as WMT24, SOURCE_COMMIT
from build_wmt26_transfer_matrices import sha256_file
from pairvar_sampling import allocated_counts, calibration_design, draw_strata
from run_quality_shift_multilingual import bootstrap_interval
from run_quality_shift_pilot import score_batch
from run_wmt26_external import METRICS, checked_matrix, partition_documents, save_json
from run_wmt26_quality_calibrated import (
    choose as baseline_choose, fingerprint, standardize_development,
    supported_budgets, vector_kmeans,
)
from run_wmt26_rank_hybrid import EXPLORATORY as WMT26, SOURCE_SHA


HERE = Path(__file__).resolve().parent
COHORTS = {"wmt24_fresh9": WMT24, "wmt26_exploratory7": WMT26}
METHODS = ("Random", "Difficulty-stratified", "Dev-cluster-proportional",
           "Hard-only", "Cal-pairvar-proportional", "Cal-pairvar-Neyman")
STOCHASTIC = set(METHODS) - {"Hard-only"}
DEFAULT_BUDGETS = [10, 25, 50, 100]


def checked_wmt24(path: Path, panel: dict) -> pd.DataFrame:
    if sha256_file(path) != panel["matrix_sha256"]:
        raise ValueError(f"Modified WMT24 matrix: {path}")
    frame = pd.read_csv(path)
    columns = ["sentence_id", "document_id", "domain", *panel["model_names"]]
    if list(frame) != columns or len(frame) < 250:
        raise ValueError(f"Wrong WMT24 complete-panel matrix schema: {path}")
    if frame[columns[:3]].isna().any().any() or frame.sentence_id.duplicated().any():
        raise ValueError(f"Missing or duplicated matrix identifiers: {path}")
    if any(not str(item).startswith(str(doc) + "_#_") for item, doc in
           zip(frame.sentence_id, frame.document_id)):
        raise ValueError("WMT24 document-to-segment mismatch")
    scores = frame[panel["model_names"]].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 100:
        raise ValueError("WMT24 complete-case ratings must be finite in [0,100]")
    return frame


def split_three_document_sets(docs: np.ndarray, seed: int, split: int):
    calibration = partition_documents(docs, seed=seed, split=split)
    distinct = np.unique(docs[calibration])
    if len(distinct) < 4:
        raise ValueError("Not enough documents for separate role and variance calibration")
    rng = np.random.default_rng(np.random.SeedSequence([seed, split, 33]))
    role_docs = rng.permutation(distinct)[:len(distinct) // 2]
    role = np.flatnonzero(calibration & np.isin(docs, role_docs))
    variance = np.flatnonzero(calibration & ~np.isin(docs, role_docs))
    eval_items = np.flatnonzero(~calibration)
    if not len(role) or len(variance) < 2 or not len(eval_items):
        raise ValueError("Empty calibration or evaluation partition")
    for a, b in ((role, variance), (role, eval_items), (variance, eval_items)):
        if set(docs[a]) & set(docs[b]):
            raise AssertionError("Cross-partition document leakage")
    return role, variance, eval_items


def choose_roles(role_scores: np.ndarray):
    if role_scores.ndim != 2 or role_scores.shape[1] < 10:
        raise ValueError("Roles need at least ten scored systems")
    means = role_scores.mean(axis=0)
    if not np.isfinite(means).all():
        raise ValueError("Invalid calibration role quality")
    ranked = np.argsort(-means, kind="stable")
    target = np.sort(ranked[:4])
    development = np.sort(ranked[-4:])
    if len(set(target) & set(development)):
        raise AssertionError("Target and development roles overlap")
    return target, development, means


def make_select_context(scores: np.ndarray, role: np.ndarray, variance: np.ndarray,
                        evaluation: np.ndarray):
    target, dev, role_means = choose_roles(scores[role])
    dev_eval = scores[evaluation][:, dev]
    dev_var = scores[variance][:, dev]
    design = calibration_design(scores[role][:, target], scores[variance][:, target],
                                dev_eval, dev_var)
    standardized = standardize_development(dev_eval, dev_var)
    return target, dev, role_means, dev_eval, dev_var, design, standardized


def select(method: str, dev_eval: np.ndarray, dev_var: np.ndarray,
           standardized: np.ndarray, clusters: np.ndarray, design: dict,
           budget: int, rng: np.random.Generator):
    """No evaluation target scores accepted; weights sum to one."""
    if method in {"Cal-pairvar-proportional", "Cal-pairvar-Neyman"}:
        dispersion = design["dispersion"] if method == "Cal-pairvar-Neyman" else None
        allocation = allocated_counts(design["sizes"], budget, dispersion)
        indices, weights = draw_strata(design["labels"], allocation, rng)
    elif method in METHODS:
        # Source-length baseline is intentionally excluded; this experiment
        # uses only the common development-label information available in ESA.
        indices, weights = baseline_choose(
            method, dev_eval, np.zeros(len(dev_eval)), dev_var.mean(axis=0),
            standardized, clusters, budget, rng)
    else:
        raise ValueError(f"Unrecognized selection method: {method}")
    if len(indices) != budget or len(set(indices)) != budget \
            or np.any(indices < 0) or np.any(indices >= len(dev_eval)) \
            or not np.isfinite(weights).all() or np.min(weights) <= 0 \
            or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12):
        raise AssertionError("Invalid evaluation-only selection or unbiased estimator weights")
    return indices, weights


def select_seed(seed: int, split: int, method_index: int, budget: int, trial: int):
    return np.random.default_rng(np.random.SeedSequence(
        [seed, split, method_index, budget, trial, 2024]))


def run_language(frame: pd.DataFrame, names: list[str], *, language: str,
                 seed: int, splits: int, trials: int, budgets: list[int], output: Path):
    scores = frame[names].to_numpy(float)
    ids = frame.sentence_id.astype(str).to_numpy()
    docs = frame.document_id.astype(str).to_numpy()
    rows, quality_gaps, manifests = [], [], []
    for split in range(splits):
        role, variance, ev = split_three_document_sets(docs, seed, split)
        if len(ev) < max(budgets):
            raise ValueError(f"Insufficient evaluation items for {language} split {split}")
        target, dev, means, dev_eval, dev_var, design, standardized = make_select_context(
            scores, role, variance, ev)
        clusters = vector_kmeans(standardized, seed=int(np.random.SeedSequence(
            [seed, split, 91]).generate_state(1)[0]))
        target_eval = scores[ev][:, target]
        reference = target_eval.mean(axis=0)
        aa, bb = np.triu_indices(4, 1)
        gaps = abs(reference[aa] - reference[bb])
        quality_gaps.append({
            "split": split, "n_role_cal_items": len(role),
            "n_variance_cal_items": len(variance), "n_evaluation_items": len(ev),
            "calibration_mean_target_minus_dev": float(means[target].mean() - means[dev].mean()),
            "evaluation_mean_target_minus_dev": float(reference.mean() - dev_eval.mean()),
            "evaluation_min_target_minus_max_dev": float(reference.min() - dev_eval.mean(axis=0).max()),
            "evaluation_target_pair_median_gap": float(np.median(gaps)),
            "evaluation_target_pairs_gap1": int((gaps > 1).sum()),
        })
        manifest = {
            "split": split, "role_calibration_ids": ids[role].tolist(),
            "variance_calibration_ids": ids[variance].tolist(),
            "evaluation_ids": ids[ev].tolist(),
            "targets_selected_from_role_calibration": [names[i] for i in target],
            "development_selected_from_role_calibration": [names[i] for i in dev],
            "cluster_sizes": np.bincount(clusters, minlength=5).tolist(),
            "stratum_sizes": design["sizes"].tolist(),
            "variance_calibration_stratum_sizes": np.bincount(
                design["calibration_labels"], minlength=3).tolist(),
            "difficulty_tertile_boundaries": design["boundaries"].tolist(),
            "pair_importance": design["pair_weights"].tolist(),
            "pairwise_dispersion": design["dispersion"].tolist(),
            "allocations": {
                str(b): {method: allocated_counts(design["sizes"], b,
                    design["dispersion"] if method == "Cal-pairvar-Neyman" else None).tolist()
                    for method in ("Cal-pairvar-proportional", "Cal-pairvar-Neyman")}
                for b in budgets},
        }
        manifests.append(manifest)
        for mi, method in enumerate(METHODS):
            repeats = trials if method in STOCHASTIC else 1
            for budget in budgets:
                selections, item_weights = [], []
                for trial in range(repeats):
                    selected, weights = select(method, dev_eval, dev_var,
                        standardized, clusters, design, budget,
                        select_seed(seed, split, mi, budget, trial))
                    selections.append(selected)
                    item_weights.append(weights)
                picks = np.stack(selections)
                weights = np.stack(item_weights)
                estimated = (target_eval[picks] * weights[:, :, None]).sum(axis=1)
                measurements = score_batch(estimated, reference)
                for trial in range(repeats):
                    rows.append({
                        "split": split, "method": method, "budget": budget, "trial": trial,
                        "selected_indices": json.dumps(ev[picks[trial]].tolist()),
                        "weights": json.dumps(weights[trial].tolist()),
                        **{metric: float(measurements[metric][trial]) for metric in METRICS},
                    })
        print(f"{language} split {split + 1}/{splits}, "
              f"role={len(role)}, variance={len(variance)}, eval={len(ev)}", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    trials_df = pd.DataFrame.from_records(rows)
    trials_df.to_csv(output / "trials.csv", index=False)
    split_summary = trials_df.groupby(["split", "method", "budget"], as_index=False)[list(METRICS)].mean()
    split_summary.to_csv(output / "split_summary.csv", index=False)
    split_summary.groupby(["method", "budget"], as_index=False)[list(METRICS)].mean().to_csv(
        output / "summary.csv", index=False)
    pd.DataFrame.from_records(quality_gaps).to_csv(output / "quality_gap_audit.csv", index=False)
    save_json(output / "split_manifest.json", manifests)
    save_json(output / "completion.json", {
        "status": "complete", "language_pair": language,
        "splits": splits, "trial_rows": len(rows), "budgets": budgets,
        "artifact_sha256": {f: sha256_file(output / f) for f in
                            ("trials.csv", "summary.csv", "split_summary.csv",
                             "quality_gap_audit.csv", "split_manifest.json")},
    })


def aggregate(out: Path, available: dict[str, list[str]], replicates: int, seed: int):
    joined = []
    for cohort, names in available.items():
        for language in names:
            f = pd.read_csv(out / "languages" / cohort / language / "summary.csv")
            f.insert(0, "language_pair", language)
            f.insert(0, "cohort", cohort)
            joined.append(f)
    all_languages = pd.concat(joined, ignore_index=True)
    all_languages.to_csv(out / "language_summary.csv", index=False)
    macro, paired = [], []
    for cohort, subset in all_languages.groupby("cohort", sort=False):
        random = subset[subset.method == "Random"].set_index(["language_pair", "budget"])
        for (method, budget), group in subset.groupby(["method", "budget"], sort=True):
            outrow = {"cohort": cohort, "method": method, "budget": int(budget),
                      "n_languages": len(group)}
            outrow.update({m: float(group[m].mean()) for m in METRICS})
            macro.append(outrow)
            for metric in METRICS:
                changes, langnames = [], []
                for row in group.itertuples(index=False):
                    base = float(random.loc[(row.language_pair, budget), metric])
                    observed = float(getattr(row, metric))
                    if np.isfinite(base) and np.isfinite(observed):
                        difference = observed - base
                        if metric == "mae":
                            difference = -difference
                        if metric in {"pairwise_accuracy", "pairwise_accuracy_gap1", "top1_accuracy"}:
                            difference *= 100  # percentage points
                        changes.append(difference)
                        langnames.append(row.language_pair)
                lower, upper = bootstrap_interval(
                    np.asarray(changes), langnames, replicates, seed)
                paired.append({"cohort": cohort, "method": method,
                               "budget": int(budget), "metric": metric,
                               "n_languages": len(changes),
                               "difference_vs_random": float(np.mean(changes)) if changes else np.nan,
                               "ci95_lower": lower, "ci95_upper": upper,
                               "wins": int(np.sum(np.asarray(changes) > 1e-10)),
                               "losses": int(np.sum(np.asarray(changes) < -1e-10)),
                               "unit": "percentage_points" if metric in
                               {"pairwise_accuracy", "pairwise_accuracy_gap1", "top1_accuracy"}
                               else "score_points" if metric in {"mae", "bias"} else "correlation"})
    pd.DataFrame.from_records(macro).to_csv(out / "macro_summary.csv", index=False)
    pd.DataFrame.from_records(paired).to_csv(out / "paired_language_comparisons.csv", index=False)


def load_panels(wmt24: Path, wmt26: Path, *, seed: int, splits: int,
                budgets: list[int], selected_cohorts: list[str]):
    manifests = {}
    panels = {}
    available = {}
    inputs = {}
    for cohort, root in (("wmt24_fresh9", wmt24), ("wmt26_exploratory7", wmt26)):
        if cohort not in selected_cohorts:
            continue
        source_path = root / "matrix_manifest.json"
        source = json.loads(source_path.read_text())
        if cohort == "wmt24_fresh9":
            if source.get("official_source_commit") != SOURCE_COMMIT:
                raise ValueError("Unverified WMT24 ESA source commit")
        elif source.get("final_annotation_sha256") != SOURCE_SHA:
            raise ValueError("Unverified WMT26 cESA source digest")
        files = {p["language_pair"]: p for p in source["language_panels"]}
        names = []
        for language in COHORTS[cohort]:
            panel = files[language]
            frame = (checked_wmt24(root / panel["matrix_file"], panel)
                     if cohort == "wmt24_fresh9" else
                     checked_matrix(root / panel["matrix_file"], panel))
            candidates = (panel["model_names"] if cohort == "wmt24_fresh9" else
                          panel["target_models"] + panel["development_models_by_coverage"])
            if len(candidates) not in (10, 18) or len(set(candidates)) != len(candidates):
                raise ValueError("Model identity mismatch")
            valid, least = supported_budgets(frame, seed, splits, budgets)
            if 25 not in valid:
                raise ValueError(f"Missing predeclared budget25 on {cohort} {language}")
            panels[(cohort, language)] = (frame, list(candidates))
            names.append(language)
            inputs[f"{cohort}/{language}"] = {
                "matrix_sha256": panel["matrix_sha256"],
                "n_items": len(frame), "n_documents": int(frame.document_id.nunique()),
                "min_evaluation_items": least, "supported_budgets": valid,
                "model_names": candidates,
            }
        manifests[cohort] = sha256_file(source_path)
        available[cohort] = names
    return panels, manifests, available, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wmt24-matrix-dir", type=Path, required=True)
    parser.add_argument("--wmt26-matrix-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", default=list(COHORTS))
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(args.cohorts) != len(set(args.cohorts)) or not set(args.cohorts) <= set(COHORTS):
        parser.error("Invalid or duplicate frozen cohort names")
    if len(args.budgets) != len(set(args.budgets)) or not args.budgets or \
            25 not in args.budgets or any(b < 5 or b % 5 for b in args.budgets):
        parser.error("Unique budgets must include 25 and be positive multiples of five")
    if min(args.splits, args.trials, args.bootstrap_replicates) < 1:
        parser.error("All replication counts must be positive")
    panels, source_manifest_hashes, groups, inputs = load_panels(
        args.wmt24_matrix_dir, args.wmt26_matrix_dir, seed=args.seed,
        splits=args.splits, budgets=args.budgets, selected_cohorts=args.cohorts)
    code_files = ("WMT24_PAIRVAR_PROTOCOL.md", "build_wmt24_esa_matrices.py",
                  "pairvar_sampling.py", "run_wmt24_pairvar.py",
                  "run_wmt26_quality_calibrated.py", "run_wmt26_external.py",
                  "run_quality_shift_pilot.py", "run_quality_shift_multilingual.py")
    config = {
        "protocol_status": "frozen_before_WMT24_outcomes", "seed": args.seed,
        "splits": args.splits, "trials": args.trials,
        "budgets": args.budgets, "bootstrap_replicates": args.bootstrap_replicates,
        "methods": list(METHODS), "selected_cohorts": args.cohorts,
        "cohorts": groups, "source_matrix_manifest_sha256": source_manifest_hashes,
        "code_sha256": {file: sha256_file(HERE / file) for file in code_files},
        "input_summary": inputs,
        "incremental_target_cost": "4 times budget; excludes role/variance calibration and full development scores",
    }
    signature = fingerprint(config)
    if args.resume:
        old = json.loads((args.out / "run_manifest.json").read_text())
        if old["signature"] != signature:
            parser.error("Code, data or protocol changed; refusing incompatible resume")
    else:
        if args.out.exists():
            parser.error("Output exists. Never silently overwrite experiment data")
        (args.out / "languages").mkdir(parents=True)
        save_json(args.out / "run_manifest.json", {
            "signature": signature, "config": config,
            "environment": {"python": platform.python_version(),
                            "numpy": np.__version__, "pandas": pd.__version__},
        })
    for cohort in args.cohorts:
        for language in groups[cohort]:
            folder = args.out / "languages" / cohort / language
            if args.resume and (folder / "completion.json").exists():
                saved = json.loads((folder / "completion.json").read_text())
                if saved.get("status") != "complete" or any(
                    sha256_file(folder / f) != digest for f, digest in
                    saved["artifact_sha256"].items()):
                    parser.error(f"Invalid completion/checksum for {folder}")
                print(f"SKIPPING VERIFIED COMPLETED {cohort}/{language}", flush=True)
                continue
            if folder.exists():
                parser.error(f"Incomplete output exists: {folder}; inspect before resuming")
            frame, names = panels[(cohort, language)]
            run_language(frame, names, language=language, seed=args.seed,
                         splits=args.splits, trials=args.trials,
                         budgets=inputs[f"{cohort}/{language}"]["supported_budgets"],
                         output=folder)
    aggregate(args.out, groups, args.bootstrap_replicates, args.seed)
    print("WMT24_PAIRVAR_EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
