"""Preflight and language-level aggregation tests (no full benchmark execution)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from run_quality_shift_multilingual import aggregate_results, bootstrap_interval, preflight
from test_quality_shift_protocol import synthetic_matrix


def language_matrix(language: str = "en-cs_CZ") -> pd.DataFrame:
    frame = synthetic_matrix()
    frame["sentence_id"] = frame.sentence_id.str.replace("test_#_", language + "_#_", regex=False)
    return frame


class MultilingualPreflightTests(unittest.TestCase):
    def test_valid_matrix_preserves_complete_protocol(self) -> None:
        with tempfile.TemporaryDirectory(prefix="quality-shift-preflight-") as tmp:
            path = Path(tmp) / "en-cs_CZ_human_scores.csv"
            language_matrix().to_csv(path, index=False)
            result = preflight(path, "en-cs_CZ", 2, [5, 10, 25], 71)
            self.assertEqual(result["n_items"], 60)
            self.assertEqual(result["n_systems"], 18)
            self.assertEqual(result["n_documents"], 20)
            self.assertEqual(result["calibration_documents"], 5)
            self.assertEqual(result["evaluation_items_min"], 45)
            self.assertEqual(result["evaluation_items_max"], 45)
            self.assertEqual(result["lower_upper_development_overlap"], 0)
            self.assertEqual(result["available_budgets"], [5, 10, 25])
            self.assertEqual(result["excluded_budgets"], [])

    def test_budget_coverage_excludes_whole_language_budget(self) -> None:
        with tempfile.TemporaryDirectory(prefix="quality-shift-budget-coverage-") as tmp:
            path = Path(tmp) / "en-cs_CZ_human_scores.csv"
            language_matrix().to_csv(path, index=False)
            result = preflight(path, "en-cs_CZ", 20, [10, 25, 50], 71)
            self.assertEqual(result["available_budgets"], [10, 25])
            self.assertEqual(result["excluded_budgets"], [50])
            # No dropping or retrying the original twenty seeded document splits.
            self.assertEqual(result["evaluation_items_min"], 45)
            self.assertEqual(result["evaluation_items_max"], 45)

    def test_invalid_matrices_are_rejected(self) -> None:
        base = language_matrix()
        cases = {}
        duplicate = base.copy()
        duplicate.loc[1, "sentence_id"] = duplicate.loc[0, "sentence_id"]
        cases["duplicate"] = duplicate
        malformed = base.copy()
        malformed.loc[0, "sentence_id"] = "en-cs_CZ_#_no_segment"
        cases["malformed"] = malformed
        wrong_language = base.copy()
        wrong_language.loc[0, "sentence_id"] = "en-it_IT_#_news_#_doc_#_0"
        cases["wrong_language"] = wrong_language
        missing_score = base.copy()
        missing_score.loc[0, "system_00"] = np.nan
        cases["missing_score"] = missing_score
        invalid_source = base.copy()
        invalid_source.loc[0, "source_text"] = " "
        cases["empty_source"] = invalid_source
        reference = base.copy()
        reference["refA"] = 100.0
        cases["reference"] = reference
        for name, frame in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix="quality-shift-bad-input-") as tmp:
                path = Path(tmp) / "matrix.csv"
                frame.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    preflight(path, "en-cs_CZ", 2, [5, 10], 71)


def make_aggregate_fixture(output: Path) -> list[str]:
    languages = ["cs-de_DE", "en-cs_CZ", "en-it_IT"]
    # The pilot deliberately has a large positive effect: it must not enter the
    # primary new-language mean. The two new-language effects have opposite signs.
    values = {
        "cs-de_DE": {"Random": (.10, 10.), "Hard-only": (.90, 1.)},
        "en-cs_CZ": {"Random": (.60, 3.), "Hard-only": (.70, 2.)},
        "en-it_IT": {"Random": (.80, 1.), "Hard-only": (.60, 4.)},
    }
    for language in languages:
        destination = output / "languages" / language
        destination.mkdir(parents=True)
        summary_rows, split_rows, gap_rows, count_rows = [], [], [], []
        for regime in ["Lower-six", "Upper-six", "Random-six"]:
            for method, (pa, mae) in values[language].items():
                record = {
                    "regime": regime, "method": method, "budget": 25,
                    "pairwise_accuracy": pa, "pairwise_accuracy_gap1": pa,
                    "spearman": 2 * pa - 1, "kendall_tau_b": 2 * pa - 1,
                    "top1_accuracy": pa, "mae": mae,
                    "bias": .1 if method == "Random" else -2.,
                }
                if language == "en-it_IT" and method == "Hard-only":
                    record["spearman"] = np.nan
                summary_rows.append(record)
                for split in range(2):
                    split_rows.append({"split": split, **record})
                    n_trials = 100 if method == "Random" else 1
                    count = {"split": split, "regime": regime, "method": method,
                             "budget": 25, "n_trials": n_trials}
                    for metric in ["pairwise_accuracy", "pairwise_accuracy_gap1", "spearman",
                                   "kendall_tau_b", "top1_accuracy", "mae", "bias"]:
                        count[metric + "_n_finite"] = n_trials if np.isfinite(record[metric]) else 0
                    count_rows.append(count)
            for split in range(2):
                gap_rows.append({
                    "split": split, "regime": regime,
                    "evaluation_mean_quality_gap": 4.0 + split,
                    "evaluation_min_target_minus_max_dev": 1.0 if split == 0 else -1.0,
                    "target_pair_median_gap": 1.5, "eligible_pairs_gap1": 8 + split,
                })
        pd.DataFrame(summary_rows).to_csv(destination / "summary.csv", index=False)
        pd.DataFrame(split_rows).to_csv(destination / "split_summary.csv", index=False)
        pd.DataFrame(gap_rows).to_csv(destination / "quality_gap_audit.csv", index=False)
        pd.DataFrame(count_rows).to_csv(destination / "metric_valid_counts.csv", index=False)
    return languages


class MultilingualAggregationTests(unittest.TestCase):
    def test_primary_excludes_pilot_and_uses_paired_language_means(self) -> None:
        with tempfile.TemporaryDirectory(prefix="quality-shift-aggregation-") as tmp:
            output = Path(tmp)
            languages = make_aggregate_fixture(output)
            totals = aggregate_results(output, languages, 1000, 72)
            self.assertEqual(totals["cohorts"]["new_languages"], ["en-cs_CZ", "en-it_IT"])
            comparisons = pd.read_csv(output / "paired_language_comparisons.csv")
            primary = comparisons[(comparisons.cohort == "new_languages") &
                                  (comparisons.regime == "Lower-six") &
                                  (comparisons.method == "Hard-only")].set_index("metric")
            pa = primary.loc["pairwise_accuracy"]
            self.assertEqual(pa.n_languages, 2)
            self.assertAlmostEqual(pa.random_mean, .70)
            self.assertAlmostEqual(pa.method_mean, .65)
            self.assertAlmostEqual(pa.improvement, -5.0)
            self.assertEqual((pa.wins, pa.ties, pa.losses), (1, 0, 1))
            self.assertEqual(pa.unit, "percentage_points")
            mae = primary.loc["mae"]
            self.assertAlmostEqual(mae.random_mean, 2.0)
            self.assertAlmostEqual(mae.method_mean, 3.0)
            self.assertAlmostEqual(mae.improvement, -1.0)
            self.assertEqual(mae.unit, "score_units")
            self.assertAlmostEqual(primary.loc["top1_accuracy", "improvement"], -5.0)
            self.assertNotIn("bias", primary.index)
            all_languages = comparisons[(comparisons.cohort == "all_languages") &
                                        (comparisons.regime == "Lower-six") &
                                        (comparisons.method == "Hard-only") &
                                        (comparisons.metric == "pairwise_accuracy")].iloc[0]
            self.assertEqual(all_languages.n_languages, 3)
            self.assertAlmostEqual(all_languages.improvement, 70 / 3)

    def test_missing_secondary_metric_uses_matched_language_mask(self) -> None:
        with tempfile.TemporaryDirectory(prefix="quality-shift-metric-denominators-") as tmp:
            output = Path(tmp)
            languages = make_aggregate_fixture(output)
            aggregate_results(output, languages, 1000, 72)
            comparisons = pd.read_csv(output / "paired_language_comparisons.csv")
            row = comparisons[(comparisons.cohort == "new_languages") &
                              (comparisons.regime == "Lower-six") &
                              (comparisons.method == "Hard-only") &
                              (comparisons.metric == "spearman")].iloc[0]
            self.assertEqual(row.n_languages, 1)
            self.assertAlmostEqual(row.random_mean, .2)
            self.assertAlmostEqual(row.method_mean, .4)
            self.assertAlmostEqual(row.improvement, .2)
            self.assertTrue(np.isnan(row.bootstrap_ci95_low))
            self.assertTrue(np.isnan(row.bootstrap_ci95_high))
            valid = pd.read_csv(output / "language_metric_valid_counts.csv")
            missing = valid[(valid.language_pair == "en-it_IT") & (valid.method == "Hard-only")]
            self.assertTrue((missing.spearman_n_valid_splits == 0).all())

    def test_language_bootstrap_is_reproducible_and_paired(self) -> None:
        values = np.array([-12., -4., 1., 7., 20.])
        languages = ["a", "b", "c", "d", "e"]
        first = bootstrap_interval(values, languages, 2000, 81)
        second = bootstrap_interval(values, languages, 2000, 81)
        translated = bootstrap_interval(values + 3.25, languages, 2000, 81)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(np.array(first) + 3.25, translated, atol=1e-12)

    def test_bootstrap_does_not_imply_replication_for_one_language(self) -> None:
        self.assertTrue(all(np.isnan(x) for x in bootstrap_interval(np.array([]), [], 100, 1)))
        self.assertTrue(all(np.isnan(x) for x in bootstrap_interval(np.array([8.]), ["cs-de_DE"], 100, 1)))


if __name__ == "__main__":
    unittest.main()
