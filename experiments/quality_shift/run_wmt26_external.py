#!/usr/bin/env python3
"""WMT26 external test of frozen WMT25 sentence-selection rules.

Six named 2026 systems are the targets.  Development systems are selected by
judgment availability (NOT score value), then split into lower/upper/random
groups using only a separate calibration-document split.  All selection uses
development scores; target evaluation judgments are hidden until scoring.

The estimand is restricted to the WMT26 *complete-case annotated panel*, not
the full WMT26 test set.  cESA is different from the WMT25 human score scale.
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

from build_wmt26_transfer_matrices import TARGETS, sha256_file
from run_quality_shift_multilingual import bootstrap_interval
from run_quality_shift_pilot import METHODS, STOCHASTIC, score_batch, select_indices


HERE = Path(__file__).resolve().parent
PRIMARY = (
    "eng_Latn---arz_Arab", "eng_Latn---rus_Cyrl",
    "eng_Latn---jpn_Jpan", "eng_Latn---ukr_Cyrl",
)
SUPPLEMENTAL = (
    "eng_Latn---ind_Latn", "eng_Latn---tha_Thai", "zho_Hans---jpn_Jpan",
)
DEFAULT_LANGUAGES = PRIMARY + SUPPLEMENTAL
REGIMES = ("Lower-six", "Upper-six", "Random-six")
METRICS = ("pairwise_accuracy", "pairwise_accuracy_gap1", "spearman",
           "kendall_tau_b", "top1_accuracy", "mae", "bias")


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def partition_documents(documents: np.ndarray, *, seed: int, split: int) -> np.ndarray:
    unique = np.unique(documents)
    n_cal = math.ceil(len(unique) * .25)
    rng = np.random.default_rng(np.random.SeedSequence([seed, split, 10]))
    return np.isin(documents, rng.permutation(unique)[:n_cal])


def available_budgets(df: pd.DataFrame, budgets: tuple[int, ...], splits: int, seed: int):
    counts = []
    documents = df.document_id.to_numpy()
    for split in range(splits):
        counts.append(int((~partition_documents(documents, seed=seed, split=split)).sum()))
    valid = [b for b in budgets if min(counts) >= b and min(counts) // 5 >= b // 5]
    return valid, min(counts), max(counts)


def checked_matrix(path: Path, panel: dict) -> pd.DataFrame:
    if sha256_file(path) != panel["matrix_sha256"]:
        raise ValueError(f"WMT26 matrix checksum changed: {path}")
    df = pd.read_csv(path)
    required = ["sentence_id", "source_text", "document_id", "domain",
                *panel["target_models"], *panel["development_models_by_coverage"]]
    if list(df) != required:
        raise ValueError(f"Unexpected score columns/order: {path}")
    if len(df) < 2 or df.sentence_id.isna().any() or df.sentence_id.duplicated().any():
        raise ValueError(f"Missing/duplicate segment identifiers: {path}")
    if df.document_id.isna().any() or df.source_text.isna().any():
        raise ValueError(f"Missing source/document metadata: {path}")
    if any(not str(s).startswith(f"{doc}_#_") for s, doc in
           zip(df.sentence_id, df.document_id)):
        raise ValueError(f"Segment-to-document grouping invalid: {path}")
    y = df[required[4:]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all() or np.min(y) < 0 or np.max(y) > 100:
        raise ValueError(f"Incomplete/out-of-range WMT26 cESA score panel: {path}")
    if set(panel["target_models"]) & set(panel["development_models_by_coverage"]):
        raise ValueError("Target and development systems overlap")
    return df


def run_language(df: pd.DataFrame, panel: dict, *, splits: int, trials: int,
                 budgets: list[int], seed: int, output: Path):
    """Keep target evaluation scores outside the selection API."""
    target = panel["target_models"]
    dev = panel["development_models_by_coverage"]
    ty = df[target].to_numpy(dtype=float)
    dy = df[dev].to_numpy(dtype=float)
    ids = df.sentence_id.astype(str).to_numpy()
    docs = df.document_id.to_numpy()
    lengths = df.source_text.astype(str).str.len().to_numpy(dtype=float)
    records, split_audit, gaps = [], [], []
    for split in range(splits):
        cal_mask = partition_documents(docs, seed=seed, split=split)
        cal = np.flatnonzero(cal_mask)
        ev = np.flatnonzero(~cal_mask)
        if not len(cal) or not len(ev) or len(ev) < max(budgets):
            raise ValueError(f"Unusable held-out document split {split}")
        if set(docs[cal]) & set(docs[ev]):
            raise AssertionError("Document leakage")
        rng = np.random.default_rng(np.random.SeedSequence([seed, split, 10]))
        # Reproduce the frozen pilot calibration permutation, including the
        # random state consumed by sampling calibration documents.
        rng.permutation(np.unique(docs))
        dcal = dy[cal].mean(axis=0)
        ordered = np.argsort(dcal, kind="stable")
        groups = {
            "Lower-six": np.sort(ordered[:6]),
            "Upper-six": np.sort(ordered[-6:]),
            "Random-six": np.sort(rng.choice(len(dev), size=6, replace=False)),
        }
        reference = ty[ev].mean(axis=0)
        diffs = abs(reference[:, None] - reference[None, :])[np.triu_indices(len(target), 1)]
        split_audit.append({
            "split": split, "calibration_ids": ids[cal].tolist(),
            "evaluation_ids": ids[ev].tolist(),
            "fixed_target_systems": target,
            "development_groups": {r: [dev[i] for i in m] for r, m in groups.items()},
        })
        for ri, regime in enumerate(REGIMES):
            group = groups[regime]
            dev_ev = dy[ev][:, group]
            dev_eval_mean = dev_ev.mean(axis=0)
            gaps.append({
                "split": split, "regime": regime,
                "n_calibration_items": len(cal), "n_evaluation_items": len(ev),
                "calibration_mean_quality_gap": float(ty[cal].mean() - dy[cal][:, group].mean()),
                "evaluation_mean_quality_gap": float(reference.mean() - dev_eval_mean.mean()),
                "evaluation_min_target_minus_max_dev": float(reference.min() - dev_eval_mean.max()),
                "target_pair_median_gap": float(np.median(diffs)),
                "eligible_pairs_gap1": int((diffs > 1).sum()),
            })
            for mi, method in enumerate(METHODS):
                repeats = trials if method in STOCHASTIC else 1
                for budget in budgets:
                    selected = []
                    for trial in range(repeats):
                        # Controls identical across regimes, matching WMT25.
                        regime_seed = 0 if method in {"Random", "Source-length-stratified"} else ri
                        ss = np.random.SeedSequence([seed, split, regime_seed,
                                                     mi, budget, trial, 70])
                        indices = select_indices(method, dev_ev, lengths[ev], dcal[group],
                                                 budget, np.random.default_rng(ss))
                        selected.append(indices)
                    selected = np.stack(selected)
                    estimates = ty[ev][selected].mean(axis=1)
                    values = score_batch(estimates, reference)
                    for trial in range(repeats):
                        record = {"split": split, "regime": regime, "method": method,
                                  "budget": budget, "trial": trial, "n_evaluation_items": len(ev),
                                  "selected_indices": json.dumps(ev[selected[trial]].tolist())}
                        record.update({m: float(values[m][trial]) for m in METRICS})
                        records.append(record)
        print(f"{panel['language_pair']} split {split+1}/{splits}: {len(cal)} cal / {len(ev)} eval", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    trials_df = pd.DataFrame.from_records(records)
    trials_df.to_csv(output / "trials.csv", index=False)
    split_means = trials_df.groupby(["split", "regime", "method", "budget"], as_index=False)[list(METRICS)].mean()
    split_means.to_csv(output / "split_summary.csv", index=False)
    summary = split_means.groupby(["regime", "method", "budget"], as_index=False)[list(METRICS)].mean()
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame.from_records(gaps).to_csv(output / "quality_gap_audit.csv", index=False)
    save_json(output / "split_manifest.json", split_audit)
    return summary


def aggregate(out: Path, languages: list[str], bootstraps: int, seed: int):
    frames = []
    for language in languages:
        frame = pd.read_csv(out / "languages" / language / "summary.csv")
        frame.insert(0, "language_pair", language)
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)
    long.to_csv(out / "language_summary.csv", index=False)
    primary = [p for p in PRIMARY if p in languages]
    cohorts = {"wmt25_overlap_primary": primary,
               "all_wmt26": languages,
               "nonoverlap_supplementary": [p for p in languages if p not in PRIMARY]}
    macro, paired = [], []
    for cohort, members in cohorts.items():
        if not members:
            continue
        subset = long[long.language_pair.isin(members)]
        baselines = subset[subset.method == "Random"].set_index(
            ["language_pair", "regime", "budget"])
        for (regime, method, budget), group in subset.groupby(
                ["regime", "method", "budget"], sort=True):
            item = {"cohort": cohort, "regime": regime, "method": method,
                    "budget": int(budget), "n_languages": len(group)}
            item.update({m: float(group[m].mean()) for m in METRICS})
            macro.append(item)
            for metric in METRICS:
                deltas, names = [], []
                for row in group.itertuples(index=False):
                    base = float(baselines.loc[(row.language_pair, regime, budget), metric])
                    value = float(getattr(row, metric))
                    if np.isfinite(base) and np.isfinite(value):
                        # MAE is a loss; positive gains mean lower error.
                        deltas.append(base - value if metric == "mae" else value - base)
                        names.append(row.language_pair)
                interval = bootstrap_interval(np.array(deltas), names, bootstraps, seed)
                paired.append({"cohort": cohort, "regime": regime, "method": method,
                               "budget": int(budget), "metric": metric,
                               "n_languages": len(names),
                               "gain_vs_random": float(np.mean(deltas)) if deltas else np.nan,
                               "ci95_lower": interval[0], "ci95_upper": interval[1]})
    pd.DataFrame(macro).to_csv(out / "macro_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired_language_comparisons.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES))
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--budgets", nargs="+", type=int, default=[10, 25, 50, 100])
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.languages)) != len(args.languages) or not set(args.languages) <= set(DEFAULT_LANGUAGES):
        parser.error("Supply a unique subset of the seven pre-registered directions")
    if len(set(args.budgets)) != len(args.budgets) or not args.budgets or any(b % 5 for b in args.budgets):
        parser.error("Supply unique positive budgets divisible by five")
    if min(args.splits, args.trials, args.bootstrap_replicates, *args.budgets) <= 0:
        parser.error("All counts must be positive")
    matrix_manifest_path = args.matrix_dir / "matrix_manifest.json"
    source = json.loads(matrix_manifest_path.read_text())
    if source.get("status") != "built" or source.get("target_models_are_fixed_before_outcomes") is not True:
        parser.error("Unverified WMT26 complete-case panel manifest")
    panels = {p["language_pair"]: p for p in source["language_panels"]}
    if tuple(source["language_panels"][0]["target_models"]) != TARGETS:
        parser.error("Unexpected 2026 target system identities")
    inputs, frames = {}, {}
    for language in args.languages:
        panel = panels[language]
        if not panel.get("matrix_file"):
            parser.error(f"No complete-case matrix for {language}")
        frame = checked_matrix(args.matrix_dir / panel["matrix_file"], panel)
        supported, least, most = available_budgets(frame, tuple(args.budgets),
                                                    args.splits, args.seed)
        if 25 in args.budgets and 25 not in supported:
            parser.error(f"{language} cannot support primary 25-item budget over all splits")
        inputs[language] = {"n_items": len(frame), "n_documents": frame.document_id.nunique(),
                            "min_evaluation_items": least, "max_evaluation_items": most,
                            "supported_budgets": supported,
                            "matrix_sha256": panel["matrix_sha256"],
                            "target_models": panel["target_models"],
                            "development_models": panel["development_models_by_coverage"]}
        frames[language] = frame
    config = {"languages": args.languages, "splits": args.splits,
              "trials": args.trials, "seed": args.seed, "budgets": args.budgets,
              "bootstrap_replicates": args.bootstrap_replicates,
              "methods": list(METHODS), "targets": list(TARGETS),
              "matrix_manifest_sha256": sha256_file(matrix_manifest_path),
              "code_sha256": {filename: sha256_file(HERE / filename) for filename in (
                  "run_wmt26_external.py", "build_wmt26_transfer_matrices.py",
                  "run_quality_shift_pilot.py", "run_quality_shift_multilingual.py")},
              "input_summary": inputs,
              "protocol_warning": "Only jointly human-scored WMT26 items; dynamic coverage is not missing at random; no numerical cross-year score comparison"}
    signature = fingerprint(config)
    if args.resume:
        previous = json.loads((args.out / "run_manifest.json").read_text())
        if previous["signature"] != signature:
            parser.error("Resume signature mismatch; different data/code/protocol")
    else:
        if args.out.exists():
            parser.error("Use a new output directory; prior results are never overwritten")
        (args.out / "languages").mkdir(parents=True)
        save_json(args.out / "run_manifest.json", {
            "status": "running", "signature": signature, "protocol": config})
        save_json(args.out / "analysis_plan.json", {
            "primary": "Four directions overlapping WMT25, 25-item pairwise accuracy",
            "supplementary": "Three additional WMT26 directions, other supported budgets, MAE and bias",
            "selection": "Targets fixed by identity; development models selected by availability masks only; calibration and evaluation split by document",
            "quality_gap": "Descriptive only; never filter by realized target advantage",
            "interpretation": "WMT26 newer-model external validation on a selected human-scored core, not chronological same-item score transfer or a causal quality-shift study",
        })
        pd.DataFrame.from_records([{"language_pair": lp, **{k: v for k, v in p.items()
            if k not in ("target_models", "development_models")}}
            for lp, p in inputs.items()]).to_csv(args.out / "input_audit.csv", index=False)
    for language in args.languages:
        final = args.out / "languages" / language
        if final.exists():
            marker_path = final / "completion.json"
            if not args.resume or not marker_path.is_file():
                raise ValueError(f"Existing incomplete language results: {final}; preserve and inspect before continuing")
            marker = json.loads(marker_path.read_text())
            if marker.get("signature") != signature or any(
                sha256_file(final / name) != checksum for name, checksum in marker["files"].items()
            ):
                raise ValueError(f"Changed or missing results in {final}")
            print(f"RESUMED {language}: complete and hash-verified", flush=True)
            continue
        panel = panels[language]
        budgets = inputs[language]["supported_budgets"]
        run_language(frames[language], panel, splits=args.splits, trials=args.trials,
                     budgets=budgets, seed=args.seed, output=final)
        files = ("trials.csv", "split_summary.csv", "summary.csv",
                 "quality_gap_audit.csv", "split_manifest.json")
        save_json(final / "completion.json", {"signature": signature,
                  "files": {name: sha256_file(final / name) for name in files}})
        print(f"COMPLETE {language}", flush=True)
    aggregate(args.out, args.languages, args.bootstrap_replicates, args.seed)
    save_json(args.out / "run_manifest.json", {"status": "completed",
              "signature": signature, "protocol": config,
              "python": platform.python_version(), "numpy": np.__version__,
              "pandas": pd.__version__})
    print("WMT26_EXTERNAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
