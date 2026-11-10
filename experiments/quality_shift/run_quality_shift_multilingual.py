#!/usr/bin/env python3
"""Run the frozen quality-shift pilot across the predeclared WMT25 languages.

The Czech-German development pilot is reproduced but excluded from the primary
new-language summary. This remains retrospective within WMT25, not a test of
chronologically new systems. Language-pair bootstrap intervals are exploratory:
directions share source documents and system families.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from audit_quality_shift_results import audit_results
from build_quality_shift_matrices import LANGUAGE_PAIRS, sha256_file
from run_quality_shift_pilot import METHODS, json_write, run_pilot

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PILOT_HASH = "8c3d925b2eb58f0ac638ac92d005252dc3cc4ba485c7bfed81c1ccea75183695"
PILOT_LANGUAGE = "cs-de_DE"
METRICS = ("pairwise_accuracy", "pairwise_accuracy_gap1", "spearman",
           "kendall_tau_b", "top1_accuracy", "mae", "bias")
GROUPS = ["regime", "method", "budget"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def preflight(path: Path, language: str, splits: int,
              budgets: list[int], seed: int) -> dict:
    """Validate all scheduled splits without reading any selection outcome."""
    if not path.is_file():
        raise ValueError(f"Missing input matrix: {path}")
    df = pd.read_csv(path)
    if not {"sentence_id", "source_text"} <= set(df):
        raise ValueError(f"Missing metadata columns in {language}")
    if df.sentence_id.isna().any() or df.sentence_id.duplicated().any():
        raise ValueError(f"Missing/duplicate item ID in {language}")
    if df.source_text.isna().any() or not df.source_text.map(lambda s: isinstance(s, str) and bool(s.strip())).all():
        raise ValueError(f"Invalid source text in {language}")
    ids = df.sentence_id.astype(str).to_numpy()
    parts = [item.split("_#_") for item in ids]
    if any(len(p) != 4 or not all(p) or p[0] != language for p in parts):
        raise ValueError(f"Unexpected document/segment ID schema in {language}")
    systems = sorted(set(df) - {"sentence_id", "source_text"})
    if "refA" in systems or len(systems) < 12:
        raise ValueError(f"Require >=12 MT systems and exclude refA: {language}")
    y = df[systems].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"Incomplete/nonfinite scores in {language}; no imputation allowed")
    groups = np.array([item.rsplit("_#_", 1)[0] for item in ids])
    documents = np.unique(groups)
    n_cal = math.ceil(.25 * len(documents))
    if not 0 < n_cal < len(documents):
        raise ValueError(f"Cannot separate calibration/evaluation documents: {language}")
    eval_counts = []
    for split in range(splits):
        rng = np.random.default_rng(np.random.SeedSequence([seed, split, 10]))
        cal = rng.permutation(documents)[:n_cal]
        n_eval = int((~np.isin(groups, cal)).sum())
        eval_counts.append(n_eval)
    available = [m for m in budgets if min(eval_counts) >= m and min(eval_counts) // 5 >= m // 5]
    excluded = [m for m in budgets if m not in available]
    if not available or (25 in budgets and 25 not in available):
        raise ValueError(f"{language} cannot support the primary requested budget on every seeded split")
    return {"language_pair": language, "filename": path.name,
            "input_sha256": sha256_file(path), "n_items": len(df),
            "n_systems": len(systems), "n_documents": len(documents),
            "calibration_documents": n_cal, "evaluation_items_min": min(eval_counts),
            "evaluation_items_max": max(eval_counts), "systems": systems,
            "available_budgets": available, "excluded_budgets": excluded,
            "lower_upper_development_overlap": max(0, 18 - len(systems))}


def compare_pilot(actual: Path, reference: Path) -> dict:
    comparisons = {}
    for filename in ("summary.csv", "split_summary.csv", "quality_gap_audit.csv"):
        expected, observed = pd.read_csv(reference / filename), pd.read_csv(actual / filename)
        pd.testing.assert_frame_equal(expected, observed, check_exact=False,
                                      rtol=1e-12, atol=1e-12)
        columns = expected.select_dtypes(include="number").columns
        difference = np.abs(expected[columns].to_numpy() - observed[columns].to_numpy())
        comparisons[filename] = {"rows": len(expected),
                                 "max_numeric_difference": float(np.nanmax(difference))}
    if read_json(reference / "split_manifest.json") != read_json(actual / "split_manifest.json"):
        raise ValueError("Czech-German partition/system manifest does not reproduce")
    result = {"status": "passed", "comparisons": comparisons,
              "split_manifest_exact_match": True,
              "reference_directory": str(reference.relative_to(ROOT))}
    json_write(actual / "pilot_reproduction.json", result)
    return result


def bootstrap_interval(values: np.ndarray, languages: list[str],
                       replicates: int, seed: int) -> tuple[float, float]:
    """Identical language lists share the same resampling indices across methods."""
    if len(values) < 2:
        return float("nan"), float("nan")
    key = int.from_bytes(hashlib.sha256("|".join(languages).encode()).digest()[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([seed, key]))
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    interval = np.quantile(values[indices].mean(axis=1), [.025, .975])
    return float(interval[0]), float(interval[1])


def aggregate_results(output: Path, languages: list[str],
                      bootstrap_replicates: int, bootstrap_seed: int) -> dict:
    collected = {}
    for filename in ("summary.csv", "split_summary.csv", "quality_gap_audit.csv", "metric_valid_counts.csv"):
        frames = []
        for language in languages:
            frame = pd.read_csv(output / "languages" / language / filename)
            frame.insert(0, "language_pair", language)
            frames.append(frame)
        collected[filename] = pd.concat(frames, ignore_index=True)
    language_summary = collected["summary.csv"]
    language_summary.to_csv(output / "language_summary.csv", index=False)
    for name in ("split_summary.csv", "quality_gap_audit.csv", "metric_valid_counts.csv"):
        collected[name].to_csv(output / name, index=False)
    split_valid = collected["split_summary.csv"].groupby(["language_pair"] + GROUPS)[list(METRICS)].count()
    split_valid.rename(columns={m: m + "_n_valid_splits" for m in METRICS}).reset_index().to_csv(
        output / "language_metric_valid_counts.csv", index=False)

    gaps = collected["quality_gap_audit.csv"]
    gap_rows = []
    for (language, regime), group in gaps.groupby(["language_pair", "regime"]):
        gap_rows.append({"language_pair": language, "regime": regime,
                         "n_splits": len(group),
                         "evaluation_mean_quality_gap": group.evaluation_mean_quality_gap.mean(),
                         "mean_gap_min": group.evaluation_mean_quality_gap.min(),
                         "mean_gap_max": group.evaluation_mean_quality_gap.max(),
                         "n_positive_mean_gap_splits": int((group.evaluation_mean_quality_gap > 0).sum()),
                         "n_strict_separation_splits": int((group.evaluation_min_target_minus_max_dev > 0).sum()),
                         "target_pair_median_gap": group.target_pair_median_gap.mean(),
                         "eligible_pairs_gap1_mean": group.eligible_pairs_gap1.mean()})
    pd.DataFrame(gap_rows).to_csv(output / "language_quality_gaps.csv", index=False)

    cohorts = {"new_languages": [lp for lp in languages if lp != PILOT_LANGUAGE],
               "all_languages": languages,
               "pilot_only": [lp for lp in languages if lp == PILOT_LANGUAGE]}
    macro_rows, comparison_rows = [], []
    for cohort, members in cohorts.items():
        if not members:
            continue
        data = language_summary[language_summary.language_pair.isin(members)]
        baseline = data[data.method == "Random"].set_index(["language_pair", "regime", "budget"])
        for (regime, method, budget), group in data.groupby(GROUPS):
            group = group.sort_values("language_pair")
            macro = {"cohort": cohort, "regime": regime, "method": method,
                     "budget": int(budget), "n_languages": len(group)}
            for metric in METRICS:
                macro[metric] = group[metric].mean()
                macro[metric + "_n_languages"] = int(group[metric].notna().sum())
                if method == "Random" or metric == "bias":
                    continue
                paired, names = [], []
                random_values, method_values = [], []
                for _, row in group.iterrows():
                    control = baseline.loc[(row.language_pair, regime, budget), metric]
                    value = row[metric]
                    if np.isfinite(control) and np.isfinite(value):
                        sign = -1 if metric == "mae" else 1
                        scale = 100 if metric in {"pairwise_accuracy", "pairwise_accuracy_gap1", "top1_accuracy"} else 1
                        paired.append(sign * scale * (value - control))
                        names.append(row.language_pair)
                        random_values.append(float(control))
                        method_values.append(float(value))
                values = np.asarray(paired, dtype=float)
                low, high = bootstrap_interval(values, names, bootstrap_replicates, bootstrap_seed)
                comparison_rows.append({"cohort": cohort, "regime": regime, "method": method,
                                        "budget": int(budget), "metric": metric,
                                        "n_languages": len(values),
                                        "random_mean": np.mean(random_values) if len(values) else np.nan,
                                        "method_mean": np.mean(method_values) if len(values) else np.nan,
                                        "improvement": values.mean() if len(values) else np.nan,
                                        "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                                        "wins": int((values > 1e-10).sum()),
                                        "ties": int((np.abs(values) <= 1e-10).sum()),
                                        "losses": int((values < -1e-10).sum()),
                                        "unit": "percentage_points" if metric in {"pairwise_accuracy", "pairwise_accuracy_gap1", "top1_accuracy"} else "score_units" if metric == "mae" else "correlation",
                                        "interpretation": "positive favors method; exploratory language-pair bootstrap, not independent-dataset inference; no interval with fewer than two languages"})
            macro_rows.append(macro)
    pd.DataFrame(macro_rows).to_csv(output / "macro_summary.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output / "paired_language_comparisons.csv", index=False)
    return {"language_summary_rows": len(language_summary),
            "split_summary_rows": len(collected["split_summary.csv"]),
            "quality_gap_rows": len(gaps), "cohorts": cohorts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGE_PAIRS))
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--budgets", nargs="+", type=int, default=[10, 25, 50, 100])
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261110)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.splits, args.trials, args.bootstrap_replicates, *args.budgets) < 1:
        parser.error("All counts and budgets must be positive")
    if len(set(args.budgets)) != len(args.budgets) or any(m % 5 for m in args.budgets):
        parser.error("Use unique budgets divisible by five to preserve the pilot allocation")
    if len(set(args.languages)) != len(args.languages) or not set(args.languages) <= set(LANGUAGE_PAIRS):
        parser.error("Languages must be a unique subset of the fixed WMT25 roster")
    if sha256_file(HERE / "run_quality_shift_pilot.py") != PILOT_HASH:
        parser.error("The frozen pilot implementation changed; review the protocol before continuing")
    scientific_scripts = ["run_quality_shift_pilot.py", "build_quality_shift_matrices.py",
                          "audit_quality_shift_results.py", Path(__file__).name]
    hashes = {name: sha256_file(HERE / name) for name in scientific_scripts}
    inputs = {lp: preflight(args.matrix_dir / f"{lp}_human_scores.csv", lp,
                            args.splits, args.budgets, args.seed) for lp in args.languages}
    protocol = {"languages": args.languages, "splits": args.splits, "trials": args.trials,
                "budgets": args.budgets, "seed": args.seed, "methods": list(METHODS),
                "script_sha256": hashes, "inputs": inputs,
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": args.bootstrap_seed}
    signature = fingerprint(protocol)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "run_manifest.json"
    if any(args.out.iterdir()):
        if not args.resume or not manifest_path.is_file():
            parser.error("Output directory is not empty; choose a new directory or a compatible --resume")
        previous = read_json(manifest_path)
        if previous["signature"] != signature:
            parser.error("Resume protocol/input/code mismatch; existing results will not be overwritten")
        manifest = previous
    else:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit = "unavailable"
        manifest = {"status": "running", "started_utc": utc_now(), "signature": signature,
                    "protocol": protocol, "repository_base_commit": commit,
                    "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                    "pandas": pd.__version__, "scipy": scipy.__version__},
                    "completed_languages": []}
        json_write(manifest_path, manifest)
        json_write(args.out / "analysis_plan.json", {
            "created_before_multilingual_execution_utc": utc_now(),
            "primary_metric": "pairwise_accuracy", "primary_budget": 25,
            "primary_cohort": [lp for lp in args.languages if lp != PILOT_LANGUAGE],
            "supplementary_cohorts": ["all_languages", "pilot_only"],
            "aggregation": "trials within splits, splits within languages, equal language weights",
            "do_not_filter": "retain every scheduled split regardless of realized quality gap or method performance",
            "budget_eligibility": "before outcomes, exclude an entire language-budget combination if any scheduled split has too few evaluation items; do not resample splits or lower budgets silently",
            "inference": "paired language-mean differences; bootstrap intervals are exploratory, without p-values or multiplicity-adjusted claims",
            "limitations": ["shared source documents and system families across language directions",
                            "not a genuinely newer-model or chronological evaluation",
                            "target calibration judgments define groups; target evaluation scores are hidden from selectors",
                            "changing development groups changes quality, identity and family composition",
                            "13 non-pilot directions were used in earlier research; this is not a new independent dataset",
                            "gap greater than one is a sensitivity threshold, not a significance test"]})
        pd.DataFrame(inputs.values()).drop(columns="systems").to_csv(args.out / "input_audit.csv", index=False)
        coverage = [{"language_pair": lp, "budget": m, "eligible": m in info["available_budgets"],
                     "min_evaluation_items": info["evaluation_items_min"],
                     "reason": "all seeded splits supported" if m in info["available_budgets"] else "at least one seeded split has insufficient evaluation items; whole language-budget excluded"}
                    for lp, info in inputs.items() for m in args.budgets]
        pd.DataFrame(coverage).to_csv(args.out / "budget_coverage.csv", index=False)
        input_manifest = args.matrix_dir / "matrix_manifest.json"
        if input_manifest.is_file():
            json_write(args.out / "source_data_manifest.json", read_json(input_manifest))
    language_root = args.out / "languages"
    language_root.mkdir(exist_ok=True)
    for position, language in enumerate(args.languages, 1):
        destination = language_root / language
        if destination.exists():
            marker = destination / "completion.json"
            if not args.resume or not marker.is_file() or read_json(marker)["signature"] != signature:
                raise ValueError(f"Cannot overwrite incompatible/incomplete language output: {destination}")
            completed = read_json(marker)
            for filename, digest in completed.get("output_sha256", {}).items():
                if not (destination / filename).is_file() or sha256_file(destination / filename) != digest:
                    raise ValueError(f"Completed output changed or is corrupt: {destination / filename}")
            if not completed.get("output_sha256"):
                raise ValueError(f"Completed output has no integrity hashes: {destination}")
            if language not in manifest["completed_languages"]:
                manifest["completed_languages"].append(language)
                json_write(manifest_path, manifest)
            print(f"RESUME {language}: validated completed run", flush=True)
            continue
        temporary = Path(tempfile.mkdtemp(prefix=f".{language}-", dir=language_root))
        print(f"LANGUAGE {position}/{len(args.languages)}: {language}", flush=True)
        df = pd.read_csv(args.matrix_dir / inputs[language]["filename"])
        eligible_budgets = inputs[language]["available_budgets"]
        config = run_pilot(df, temporary, args.splits, args.trials, args.seed, eligible_budgets)
        config.update({"language_pair": language, "input_sha256": inputs[language]["input_sha256"],
                       "scipy": scipy.__version__, "pilot_script_sha256": PILOT_HASH})
        json_write(temporary / "run_config.json", config)
        audit_results(df, temporary, args.splits, args.trials, eligible_budgets, args.seed)
        if language == PILOT_LANGUAGE and (args.splits, args.trials, args.seed, eligible_budgets) == (20, 100, 20261106, [10, 25, 50, 100]):
            compare_pilot(temporary, ROOT / "results" / "quality_shift_cs_de_pilot")
        output_hashes = {path.name: sha256_file(path) for path in temporary.iterdir() if path.is_file()}
        json_write(temporary / "completion.json", {"status": "completed_and_audited",
                   "signature": signature, "language_pair": language, "completed_utc": utc_now(),
                   "output_sha256": output_hashes})
        temporary.rename(destination)
        manifest["completed_languages"].append(language)
        json_write(manifest_path, manifest)
        print(f"AUDIT PASSED: {language}", flush=True)
    totals = aggregate_results(args.out, args.languages, args.bootstrap_replicates, args.bootstrap_seed)
    manifest.update({"status": "completed", "completed_utc": utc_now(), "aggregate": totals})
    json_write(manifest_path, manifest)
    print(json.dumps({"status": "completed", "output": str(args.out), **totals}, indent=2))


if __name__ == "__main__":
    main()
