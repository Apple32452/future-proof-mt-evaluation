#!/usr/bin/env python3
"""Audit saved quality-shift runs against their supplied item-level matrix.

All saved selections are checked. Metrics are independently recomputed for trial
zero of every split/regime/method/budget, while every trial enters an independent
reaggregation. This is an audit, not another selector implementation.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.stats import kendalltau, spearmanr

try:
    from .run_quality_shift_pilot import EPS, METHODS, STOCHASTIC
except ImportError:  # Direct script invocation and unittest discovery.
    from run_quality_shift_pilot import EPS, METHODS, STOCHASTIC


METRICS = (
    "pairwise_accuracy", "pairwise_accuracy_gap1", "spearman", "kendall_tau_b",
    "top1_accuracy", "mae", "bias",
)
REGIMES = ("Lower-six", "Upper-six", "Random-six")
GROUP_KEYS = ["split", "regime", "method", "budget"]
TRIAL_KEYS = GROUP_KEYS + ["trial"]
SUMMARY_KEYS = ["regime", "method", "budget"]
ATOL = 1e-10
RTOL = 1e-10


class QualityShiftAuditError(ValueError):
    """A saved run does not satisfy the frozen pilot's audit contract."""


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sign_with_tolerance(value: float) -> int:
    return 0 if abs(value) <= EPS else (1 if value > 0 else -1)


def _independent_metrics(estimate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Scalar oracle using SciPy rank statistics and explicit pairwise counts.

The frozen pilot treats pair differences at most EPS as ties for PA/Kendall,
but uses exact ranks for Spearman. SciPy Kendall uses exact ties; if there are
nonzero near ties, explicit concordance/tie counts preserve the pilot's declared
EPS convention without changing its behavior.
    """
    estimate = np.asarray(estimate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if estimate.ndim != 1 or estimate.shape != reference.shape:
        raise ValueError("The metric oracle expects equal-length score vectors.")
    if not np.isfinite(estimate).all() or not np.isfinite(reference).all():
        raise ValueError("The metric oracle requires finite input scores.")
    pair_credits: list[float] = []
    gap_credits: list[float] = []
    concordant = discordant = reference_untied = estimate_untied = 0
    has_numerical_near_tie = False
    for first in range(len(reference)):
        for second in range(first + 1, len(reference)):
            reference_gap = float(reference[first] - reference[second])
            estimate_gap = float(estimate[first] - estimate[second])
            rs = _sign_with_tolerance(reference_gap)
            es = _sign_with_tolerance(estimate_gap)
            reference_untied += int(rs != 0)
            estimate_untied += int(es != 0)
            if rs and es:
                concordant += int(rs == es)
                discordant += int(rs != es)
            has_numerical_near_tie |= (
                0 < abs(reference_gap) <= EPS or 0 < abs(estimate_gap) <= EPS
            )
            if rs:
                credit = 0.5 if es == 0 else float(rs == es)
                pair_credits.append(credit)
                if abs(reference_gap) > 1.0:
                    gap_credits.append(credit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = float(spearmanr(reference, estimate).statistic)
        tau = float(kendalltau(reference, estimate, variant="b").statistic)
    if has_numerical_near_tie:
        denominator = math.sqrt(reference_untied * estimate_untied)
        tau = (concordant - discordant) / denominator if denominator else float("nan")
    true_best = set(np.flatnonzero(np.abs(reference - reference.max()) <= EPS))
    estimated_best = set(np.flatnonzero(np.abs(estimate - estimate.max()) <= EPS))
    errors = [float(estimated - actual) for estimated, actual in zip(estimate, reference)]
    return {
        "pairwise_accuracy": float(np.mean(pair_credits)) if pair_credits else float("nan"),
        "pairwise_accuracy_gap1": float(np.mean(gap_credits)) if gap_credits else float("nan"),
        "spearman": rho,
        "kendall_tau_b": tau,
        "top1_accuracy": len(true_best & estimated_best) / len(estimated_best),
        "mae": float(np.mean([abs(value) for value in errors])),
        "bias": float(np.mean(errors)),
    }


def _has_near_ties(values: np.ndarray) -> bool:
    gaps = np.abs(values[:, None] - values[None, :])
    return bool(((gaps > 0) & (gaps <= EPS)).any())


def _independent_group_means(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key))
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            observed = values[~np.isnan(values)]
            row[metric] = float(np.mean(observed)) if len(observed) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=keys + list(METRICS))


def audit_results(df: pd.DataFrame, output: Path, splits: int, trials: int,
                  budgets: list[int], seed: int) -> dict[str, Any]:
    """Validate a frozen-pilot output directory and write an explicit audit report.

Returns the report on success. On failure, writes a failed report and raises
QualityShiftAuditError. Existing pilot result files are never modified. The
caller remains responsible for verifying the raw file's hash/provenance before
passing its dataframe to this function.
    """
    output = Path(output)
    if not output.is_dir():
        raise QualityShiftAuditError(f"Audit directory does not exist: {output}")
    report: dict[str, Any] = {
        "audit_version": 1,
        "status": "in_progress",
        "checks": [],
        "scope": {
            "matrix": "Caller-supplied item-level dataframe; raw-file provenance is the caller's responsibility.",
            "selections": "Every trial: metadata, unique evaluation-only indices, budgets, and shared controls.",
            "metrics": "Independent metric oracle for trial 0 of every split/regime/method/budget; all trials reaggregated.",
            "rank_oracles": "SciPy Spearman/Kendall; explicit EPS-aware Kendall counts only for numerical near ties.",
            "selector_reimplementation": False,
            "statistical_scope": "Within-language consistency audit, not a significance test or proof of generalization.",
        },
        "numeric_tolerance": {"atol": ATOL, "rtol": RTOL, "pair_tie_epsilon": EPS},
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
        },
        "selection_rows_checked": 0,
        "metric_rows_recomputed": 0,
        "metric_rows_with_numerical_near_ties": 0,
    }

    def require(condition: bool, name: str, detail: str = "") -> None:
        if not condition:
            report["checks"].append({"name": name, "passed": False, "detail": detail})
            raise QualityShiftAuditError(f"{name}: {detail}")

    def passed(name: str, detail: str = "") -> None:
        report["checks"].append({"name": name, "passed": True, "detail": detail})

    def compare_table(actual: pd.DataFrame, expected: pd.DataFrame,
                      keys: list[str], numeric: list[str], label: str) -> None:
        require(set(keys + numeric) <= set(actual.columns), label, "Missing required columns.")
        require(not actual[keys].isna().any().any(), label, "Missing key values.")
        require(not actual.duplicated(keys).any(), label, "Duplicate table keys.")
        actual_keys = set(actual[keys].itertuples(index=False, name=None))
        expected_keys = set(expected[keys].itertuples(index=False, name=None))
        require(actual_keys == expected_keys, label, "Stored and expected row keys differ.")
        merged = expected.merge(actual, on=keys, how="inner", suffixes=("_expected", "_saved"), validate="one_to_one")
        for column in numeric:
            left = pd.to_numeric(merged[column + "_expected"], errors="raise").to_numpy(dtype=float)
            right = pd.to_numeric(merged[column + "_saved"], errors="raise").to_numpy(dtype=float)
            close = np.isclose(left, right, atol=ATOL, rtol=RTOL, equal_nan=True)
            if not close.all():
                bad = int(np.flatnonzero(~close)[0])
                row_key = {key: str(merged.iloc[bad][key]) for key in keys}
                require(False, label, f"{column} mismatch at {row_key}: expected {left[bad]!r}, saved {right[bad]!r}.")
        passed(label, f"Verified {len(expected)} keyed rows.")

    try:
        require(splits > 0 and trials > 0 and bool(budgets), "arguments", "Positive counts and nonempty budgets required.")
        require(all(isinstance(value, (int, np.integer)) and value > 0 for value in budgets), "arguments", "Budgets must be positive integers.")
        require(len(set(budgets)) == len(budgets), "arguments", "Duplicate budgets are not allowed.")
        files = ["trials.csv", "split_manifest.json", "split_summary.csv", "summary.csv", "quality_gap_audit.csv", "run_config.json"]
        require(all((output / name).is_file() for name in files), "required_files", "One or more frozen-pilot output files are missing.")
        passed("required_files")
        require({"sentence_id", "source_text"} <= set(df.columns), "input_schema", "Required metadata columns are missing.")
        require(not df.sentence_id.isna().any() and not df.sentence_id.duplicated().any(), "input_ids", "Missing or duplicate item IDs.")
        require(not df.source_text.isna().any(), "input_schema", "Missing source text.")
        ids = df.sentence_id.astype(str).to_numpy()
        require(len(set(ids)) == len(ids), "input_ids", "Item IDs collide after conversion to strings.")
        # rpartition independently implements the pilot's removal of only the final component.
        parts = [value.rpartition("_#_") for value in ids]
        require(all(prefix and separator and segment for prefix, separator, segment in parts), "document_grouping", "Every item ID must include a nonempty document prefix and final segment component.")
        documents = np.array([prefix for prefix, _, _ in parts])
        unique_documents = np.unique(documents)
        require(len(unique_documents) >= 2, "document_grouping", "At least two source documents are required.")
        systems = sorted(column for column in df if column not in {"sentence_id", "source_text", "refA"})
        require(len(systems) >= 12, "input_schema", "At least 12 systems are required.")
        y = df[systems].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        require(np.isfinite(y).all(), "input_schema", "Nonfinite input scores.")
        passed("input_schema_and_document_grouping", f"{len(ids)} items, {len(unique_documents)} documents, {len(systems)} systems.")

        config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        expected_config = {
            "n_items": len(df), "n_systems": len(systems), "n_documents": len(unique_documents),
            "splits": splits, "trials_stochastic": trials, "trials_deterministic": 1,
            "seed": seed, "budgets": list(budgets), "methods": list(METHODS),
            "calibration_document_fraction": 0.25, "target_systems": 6, "dev_systems": 6,
            "upper_dev_count": 3,
        }
        for key, value in expected_config.items():
            require(config.get(key) == value, "run_configuration", f"{key}: expected {value!r}, saved {config.get(key)!r}.")
        passed("run_configuration")

        manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
        require(isinstance(manifest, list) and len(manifest) == splits, "manifest", "Unexpected manifest length/type.")
        manifest_ids = [entry.get("split") for entry in manifest]
        require(set(manifest_ids) == set(range(splits)) and len(set(manifest_ids)) == splits, "manifest", "Missing or duplicate split IDs.")
        by_split = {entry["split"]: entry for entry in manifest}
        contexts: dict[int, dict[str, Any]] = {}
        gap_rows: list[dict[str, Any]] = []
        group_overlap_counts: list[int] = []
        for split in range(splits):
            rng = np.random.default_rng(np.random.SeedSequence([seed, split, 10]))
            n_calibration_documents = math.ceil(0.25 * len(unique_documents))
            calibration_documents = set(rng.permutation(unique_documents)[:n_calibration_documents])
            calibration = np.array([index for index, doc in enumerate(documents) if doc in calibration_documents], dtype=int)
            evaluation = np.array([index for index, doc in enumerate(documents) if doc not in calibration_documents], dtype=int)
            require(len(calibration) > 0 and len(evaluation) >= max(budgets), "document_partitions", f"Insufficient items in split {split}.")
            entry = by_split[split]
            require(entry.get("calibration_ids") == ids[calibration].tolist(), "manifest_calibration_ids", f"Split {split} differs from the seeded document partition.")
            require(entry.get("evaluation_ids") == ids[evaluation].tolist(), "manifest_evaluation_ids", f"Split {split} differs from the seeded document partition.")
            require(not (set(documents[calibration]) & set(documents[evaluation])), "document_disjointness", f"Document leakage in split {split}.")
            require(set(calibration) | set(evaluation) == set(range(len(df))), "partition_coverage", f"Split {split} does not partition all input rows.")
            calibration_means = y[calibration].mean(axis=0)
            ordered = np.array(sorted(range(len(systems)), key=lambda index: calibration_means[index]), dtype=int)
            target = ordered[-6:]
            pool = ordered[:-6]
            development = {
                "Lower-six": pool[:6],
                "Upper-six": pool[-6:],
                "Random-six": np.sort(rng.choice(pool, size=6, replace=False)),
            }
            require(entry.get("target_systems") == [systems[index] for index in target], "target_assignment", f"Split {split} target systems/order do not match calibration-only assignment.")
            expected_development = {regime: [systems[index] for index in indices] for regime, indices in development.items()}
            require(entry.get("development_groups") == expected_development, "development_assignment", f"Split {split} development groups do not match calibration-only assignment.")
            for regime, indices in development.items():
                require(len(indices) == 6 and len(set(indices)) == 6 and not (set(indices) & set(target)), "system_disjointness", f"Invalid systems in split {split}, {regime}.")
            group_overlap_counts.append(len(set(development["Lower-six"]) & set(development["Upper-six"])))
            reference = y[evaluation][:, target].mean(axis=0)
            pair_gaps = np.array([abs(reference[a] - reference[b]) for a in range(6) for b in range(a + 1, 6)])
            mask = np.zeros(len(df), dtype=bool)
            mask[evaluation] = True
            contexts[split] = {
                "evaluation": evaluation, "evaluation_mask": mask, "target": target,
                "reference": reference, "eligible_primary": int((pair_gaps > EPS).sum()),
                "eligible_gap1": int((pair_gaps > 1.0).sum()),
            }
            for regime, indices in development.items():
                development_means = y[evaluation][:, indices].mean(axis=0)
                gap_rows.append({
                    "split": split, "regime": regime,
                    "n_calibration_items": len(calibration), "n_evaluation_items": len(evaluation),
                    "calibration_mean_quality_gap": float(calibration_means[target].mean() - calibration_means[indices].mean()),
                    "evaluation_mean_quality_gap": float(reference.mean() - development_means.mean()),
                    "evaluation_min_target_minus_max_dev": float(reference.min() - development_means.max()),
                    "target_score_range": float(reference.max() - reference.min()),
                    "target_pair_median_gap": float(np.median(pair_gaps)),
                    "eligible_pairs_gap1": int((pair_gaps > 1.0).sum()),
                })
        passed("seeded_manifest_ids_documents_and_systems", f"Independently reconstructed all {splits} partitions and quality groups.")
        report["lower_upper_development_overlap_by_split"] = group_overlap_counts
        expected_gaps = pd.DataFrame(gap_rows)
        gap_numeric = [column for column in expected_gaps if column not in ["split", "regime"]]
        compare_table(pd.read_csv(output / "quality_gap_audit.csv"), expected_gaps, ["split", "regime"], gap_numeric, "quality_gap_recomputation")

        saved_trials = pd.read_csv(output / "trials.csv")
        required_trial_columns = set(TRIAL_KEYS + list(METRICS) + ["selected_indices", "n_evaluation_items", "synthetic"])
        require(required_trial_columns <= set(saved_trials), "trial_schema", "Missing trial fields.")
        for key in ["split", "budget", "trial", "n_evaluation_items"]:
            values = pd.to_numeric(saved_trials[key], errors="raise").to_numpy(dtype=float)
            require(np.isfinite(values).all() and np.equal(values, np.floor(values)).all(), "trial_schema", f"Noninteger {key} values.")
            saved_trials[key] = values.astype(int)
        require(not saved_trials[TRIAL_KEYS].isna().any().any(), "trial_schema", "Missing trial keys.")
        require(not saved_trials.duplicated(TRIAL_KEYS).any(), "trial_uniqueness", "Duplicated trial keys.")
        expected_trial_count = splits * len(REGIMES) * len(budgets) * sum(trials if method in STOCHASTIC else 1 for method in METHODS)
        require(len(saved_trials) == expected_trial_count, "trial_row_count", f"Expected {expected_trial_count}, found {len(saved_trials)}.")
        expected_group_keys = {(split, regime, method, budget) for split in range(splits) for regime in REGIMES for method in METHODS for budget in budgets}
        observed_group_keys = set(saved_trials[GROUP_KEYS].itertuples(index=False, name=None))
        require(observed_group_keys == expected_group_keys, "trial_group_coverage", "Missing/unexpected split/regime/method/budget groups.")
        for key, group in saved_trials.groupby(GROUP_KEYS, sort=False):
            repetitions = trials if key[2] in STOCHASTIC else 1
            require(set(group.trial) == set(range(repetitions)) and len(group) == repetitions, "trial_repetitions", f"Unexpected repetitions for {key}.")
        for metric in METRICS:
            saved_trials[metric] = pd.to_numeric(saved_trials[metric], errors="raise")
            require(not np.isinf(saved_trials[metric].to_numpy(dtype=float)).any(), "metric_ranges", f"Infinite {metric} values.")
        for metric in ["pairwise_accuracy", "pairwise_accuracy_gap1", "top1_accuracy"]:
            values = saved_trials[metric].dropna().to_numpy(dtype=float)
            require(((values >= -ATOL) & (values <= 1.0 + ATOL)).all(), "metric_ranges", f"{metric} lies outside [0,1].")
        for metric in ["spearman", "kendall_tau_b"]:
            values = saved_trials[metric].dropna().to_numpy(dtype=float)
            require((np.abs(values) <= 1.0 + ATOL).all(), "metric_ranges", f"{metric} lies outside [-1,1].")
        require(saved_trials[["mae", "bias", "top1_accuracy"]].notna().all().all(), "metric_ranges", "MAE, bias, and top-1 must always be finite for finite input scores.")
        require((saved_trials.mae >= -ATOL).all(), "metric_ranges", "Negative MAE.")
        for split, group in saved_trials.groupby("split"):
            context = contexts[int(split)]
            for metric, eligible in [("pairwise_accuracy", context["eligible_primary"]), ("pairwise_accuracy_gap1", context["eligible_gap1"])]:
                require(bool(group[metric].notna().all()) if eligible else bool(group[metric].isna().all()), "undefined_pairwise_metrics", f"Incorrect missing-value pattern for {metric}, split {split}.")
        passed("trial_counts_keys_and_metric_domains", f"{expected_trial_count} trial rows.")

        count_rows = []
        for key, group in saved_trials.groupby(GROUP_KEYS, sort=True):
            count = dict(zip(GROUP_KEYS, key))
            count["n_trials"] = len(group)
            count.update({metric + "_n_finite": int(np.isfinite(group[metric].to_numpy(dtype=float)).sum()) for metric in METRICS})
            count_rows.append(count)
        pd.DataFrame(count_rows).to_csv(output / "metric_valid_counts.csv", index=False)
        report["metric_valid_counts_file"] = "metric_valid_counts.csv"

        controls: dict[tuple[Any, ...], tuple[int, ...]] = {}
        columns = list(saved_trials.columns)
        positions = {name: columns.index(name) for name in columns}
        for raw_row in saved_trials.itertuples(index=False, name=None):
            row = {name: raw_row[positions[name]] for name in TRIAL_KEYS}
            key = tuple(row[name] for name in TRIAL_KEYS)
            context = contexts[row["split"]]
            selected = json.loads(raw_row[positions["selected_indices"]])
            require(isinstance(selected, list) and all(isinstance(index, int) and not isinstance(index, bool) for index in selected), "selection_encoding", f"Noninteger index list at {key}.")
            require(len(selected) == row["budget"] and len(set(selected)) == row["budget"], "selection_budget_uniqueness", f"Incorrect size or duplicate item at {key}.")
            require(all(0 <= index < len(df) for index in selected), "selection_index_range", f"Out-of-range index at {key}.")
            require(bool(context["evaluation_mask"][selected].all()), "selection_evaluation_only", f"Calibration item selected at {key}.")
            require(raw_row[positions["n_evaluation_items"]] == len(context["evaluation"]), "trial_evaluation_size", f"Wrong evaluation size at {key}.")
            require(bool(raw_row[positions["synthetic"]]) == bool(config.get("synthetic_only")), "trial_data_kind", f"Synthetic/real flag mismatch at {key}.")
            if row["method"] in {"Random", "Source-length-stratified"}:
                control_key = (row["split"], row["method"], row["budget"], row["trial"])
                selection = tuple(selected)
                if control_key in controls:
                    require(selection == controls[control_key], "shared_control_selections", f"Control differs across regimes at {key}.")
                else:
                    controls[control_key] = selection
            report["selection_rows_checked"] += 1
            if row["trial"] == 0:
                # Contiguous m x K data matches the saved batch estimator's summation layout.
                target_values = np.ascontiguousarray(y[selected][:, context["target"]])
                estimate = target_values.mean(axis=0)
                reference = context["reference"]
                oracle = _independent_metrics(estimate, reference)
                for metric, expected in oracle.items():
                    actual = float(raw_row[positions[metric]])
                    require(bool(np.isclose(actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True)), "independent_trial0_metrics", f"{metric} mismatch at {key}: expected {expected!r}, saved {actual!r}.")
                report["metric_rows_recomputed"] += 1
                report["metric_rows_with_numerical_near_ties"] += int(_has_near_ties(estimate) or _has_near_ties(reference))
        passed("all_selection_ids_and_budgets", f"Checked {report['selection_rows_checked']} saved selections.")
        passed("shared_random_and_source_controls", f"Compared {len(controls)} controls across all three regimes.")
        require(report["metric_rows_recomputed"] == len(expected_group_keys), "metric_oracle_coverage", "Trial-zero metric oracle did not cover every group.")
        passed("independent_trial0_metrics", f"Recomputed {report['metric_rows_recomputed']} trial rows.")

        expected_split_summary = _independent_group_means(saved_trials, GROUP_KEYS)
        compare_table(pd.read_csv(output / "split_summary.csv"), expected_split_summary, GROUP_KEYS, list(METRICS), "split_summary_reaggregation")
        expected_summary = _independent_group_means(expected_split_summary, SUMMARY_KEYS)
        compare_table(pd.read_csv(output / "summary.csv"), expected_summary, SUMMARY_KEYS, list(METRICS), "equal_split_summary_reaggregation")
        report["observed_counts"] = {
            "trial_rows": len(saved_trials), "split_summary_rows": len(expected_split_summary),
            "summary_rows": len(expected_summary), "quality_gap_rows": len(expected_gaps),
            "manifest_splits": len(manifest),
        }
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        _json_write(output / "audit_report.json", report)
        if isinstance(exc, QualityShiftAuditError):
            raise
        raise QualityShiftAuditError(str(exc)) from exc
    _json_write(output / "audit_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Existing frozen-pilot output directory.")
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    report = audit_results(pd.read_csv(args.data), args.out, args.splits, args.trials, args.budgets, args.seed)
    print(json.dumps({"status": report["status"], "selection_rows_checked": report["selection_rows_checked"], "metric_rows_recomputed": report["metric_rows_recomputed"]}, indent=2))


if __name__ == "__main__":
    main()
