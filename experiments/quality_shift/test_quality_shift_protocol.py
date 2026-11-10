"""Regression tests for the frozen selector protocol and independent audit."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .audit_quality_shift_results import METRICS, QualityShiftAuditError, _independent_metrics, audit_results
    from .run_quality_shift_pilot import METHODS, run_pilot, score_batch
except ImportError:
    from audit_quality_shift_results import METRICS, QualityShiftAuditError, _independent_metrics, audit_results
    from run_quality_shift_pilot import METHODS, run_pilot, score_batch


def synthetic_matrix() -> pd.DataFrame:
    rng = np.random.default_rng(184)
    n_items, n_systems = 60, 18
    ability = np.linspace(0, 20, n_systems)
    difficulty = rng.normal(0, 7, n_items)
    scores = np.clip(65 + ability[None, :] - difficulty[:, None] + rng.normal(0, 5, (n_items, n_systems)), 0, 100)
    scores = np.round(scores * 2) / 2  # Include exact binary-representable score ties.
    frame = pd.DataFrame({
        "sentence_id": [f"test_#_news_#_doc{index // 3:03d}_#_{index % 3}" for index in range(n_items)],
        "source_text": ["test " + "x" * (index % 29) for index in range(n_items)],
    })
    for index in range(n_systems):
        frame[f"system_{index:02d}"] = scores[:, index]
    return frame


def run_quietly(frame: pd.DataFrame, output: Path, splits: int = 2, trials: int = 3,
                budgets: list[int] | None = None, seed: int = 71) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        run_pilot(frame, output, splits, trials, seed, budgets or [5, 10], synthetic=True)


class MetricOracleTests(unittest.TestCase):
    def test_ordinary_tied_and_constant_metric_agreement(self) -> None:
        cases = [
            ([1., 3., 6., 8.], [2., 4., 5., 9.]),
            ([1., 3., 6., 8.], [9., 5., 4., 2.]),
            ([1., 1., 3., 4.], [3., 3., 2., 4.]),
            ([1., 3., 6., 8.], [2., 2., 2., 2.]),
            ([3., 3., 3., 3.], [1., 2., 3., 4.]),
            ([3., 3., 3., 3.], [2., 2., 2., 2.]),
            ([0., 0.5e-12, 2.], [3., 2., 1.]),
        ]
        for reference, estimate in cases:
            with self.subTest(reference=reference, estimate=estimate):
                reference = np.array(reference)
                estimate = np.array(estimate)
                oracle = _independent_metrics(estimate, reference)
                actual = score_batch(estimate, reference)
                for metric in METRICS:
                    np.testing.assert_allclose(actual[metric][0], oracle[metric], atol=1e-12, rtol=1e-12, equal_nan=True, err_msg=metric)

    def test_tie_and_gap_boundary_conventions(self) -> None:
        result = _independent_metrics(np.array([2., 2.]), np.array([0., 1.]))
        self.assertEqual(result["pairwise_accuracy"], 0.5)
        self.assertTrue(np.isnan(result["pairwise_accuracy_gap1"]))
        result = _independent_metrics(np.array([0., 2.]), np.array([0., 1.5]))
        self.assertEqual(result["pairwise_accuracy_gap1"], 1.0)


class ProtocolAuditTests(unittest.TestCase):
    def test_full_small_run_passes_and_reaggregates(self) -> None:
        frame = synthetic_matrix()
        with tempfile.TemporaryDirectory(prefix="quality-shift-audit-") as tmp:
            output = Path(tmp) / "run"
            run_quietly(frame, output)
            report = audit_results(frame, output, 2, 3, [5, 10], 71)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["selection_rows_checked"], 144)
            self.assertEqual(report["metric_rows_recomputed"], 72)
            counts = pd.read_csv(output / "metric_valid_counts.csv")
            self.assertEqual(len(counts), 72)
            self.assertTrue((counts.mae_n_finite == counts.n_trials).all())
            saved_report = json.loads((output / "audit_report.json").read_text())
            self.assertEqual(saved_report["status"], "passed")

    def test_target_evaluation_perturbation_preserves_all_selections(self) -> None:
        frame = synthetic_matrix()
        with tempfile.TemporaryDirectory(prefix="quality-shift-leakage-") as tmp:
            first, second = Path(tmp) / "original", Path(tmp) / "perturbed"
            run_quietly(frame, first, splits=1, trials=2)
            manifest = json.loads((first / "split_manifest.json").read_text())[0]
            changed = frame.copy()
            evaluation = changed.sentence_id.isin(manifest["evaluation_ids"])
            target_names = manifest["target_systems"]
            rng = np.random.default_rng(907)
            changed.loc[evaluation, target_names] = rng.normal(30, 20, (int(evaluation.sum()), len(target_names)))
            # Calibration scores and all development evaluation scores remain fixed.
            run_quietly(changed, second, splits=1, trials=2)
            original_trials = pd.read_csv(first / "trials.csv")
            perturbed_trials = pd.read_csv(second / "trials.csv")
            columns = ["split", "regime", "method", "budget", "trial", "selected_indices"]
            pd.testing.assert_frame_equal(original_trials[columns], perturbed_trials[columns])
            self.assertEqual(set(original_trials.method), set(METHODS))
            self.assertFalse(np.allclose(original_trials.mae, perturbed_trials.mae))
            audit_results(changed, second, 1, 2, [5, 10], 71)

    def test_undefined_metrics_are_preserved_and_counted(self) -> None:
        frame = synthetic_matrix()
        system_columns = [column for column in frame if column.startswith("system_")]
        frame[system_columns] = 80.0
        with tempfile.TemporaryDirectory(prefix="quality-shift-constant-") as tmp:
            output = Path(tmp) / "run"
            run_quietly(frame, output, splits=1, trials=2)
            report = audit_results(frame, output, 1, 2, [5, 10], 71)
            self.assertEqual(report["status"], "passed")
            counts = pd.read_csv(output / "metric_valid_counts.csv")
            self.assertTrue((counts.pairwise_accuracy_n_finite == 0).all())
            self.assertTrue((counts.spearman_n_finite == 0).all())
            self.assertTrue((counts.kendall_tau_b_n_finite == 0).all())
            self.assertTrue((counts.top1_accuracy_n_finite == counts.n_trials).all())
            summary = pd.read_csv(output / "summary.csv")
            self.assertTrue(summary.spearman.isna().all())

    def test_tampered_selection_is_rejected(self) -> None:
        frame = synthetic_matrix()
        with tempfile.TemporaryDirectory(prefix="quality-shift-bad-selection-") as tmp:
            output = Path(tmp) / "run"
            run_quietly(frame, output)
            saved = pd.read_csv(output / "trials.csv")
            selected = json.loads(saved.loc[0, "selected_indices"])
            selected[0] = selected[1]
            saved.loc[0, "selected_indices"] = json.dumps(selected)
            saved.to_csv(output / "trials.csv", index=False)
            with self.assertRaisesRegex(QualityShiftAuditError, "selection_budget_uniqueness"):
                audit_results(frame, output, 2, 3, [5, 10], 71)
            self.assertEqual(json.loads((output / "audit_report.json").read_text())["status"], "failed")

    def test_tampered_trial0_metric_is_rejected(self) -> None:
        frame = synthetic_matrix()
        with tempfile.TemporaryDirectory(prefix="quality-shift-bad-metric-") as tmp:
            output = Path(tmp) / "run"
            run_quietly(frame, output)
            saved = pd.read_csv(output / "trials.csv")
            saved.loc[0, "pairwise_accuracy"] = 0.123456
            saved.to_csv(output / "trials.csv", index=False)
            with self.assertRaisesRegex(QualityShiftAuditError, "independent_trial0_metrics"):
                audit_results(frame, output, 2, 3, [5, 10], 71)

    def test_tampered_summary_is_rejected(self) -> None:
        frame = synthetic_matrix()
        with tempfile.TemporaryDirectory(prefix="quality-shift-bad-summary-") as tmp:
            output = Path(tmp) / "run"
            run_quietly(frame, output)
            summary = pd.read_csv(output / "summary.csv")
            summary.loc[0, "mae"] += 0.25
            summary.to_csv(output / "summary.csv", index=False)
            with self.assertRaisesRegex(QualityShiftAuditError, "equal_split_summary_reaggregation"):
                audit_results(frame, output, 2, 3, [5, 10], 71)

    def test_tampered_document_manifest_is_rejected(self) -> None:
        frame = synthetic_matrix()
        with tempfile.TemporaryDirectory(prefix="quality-shift-bad-manifest-") as tmp:
            output = Path(tmp) / "run"
            run_quietly(frame, output)
            path = output / "split_manifest.json"
            manifest = json.loads(path.read_text())
            manifest[0]["calibration_ids"][0] = manifest[0]["evaluation_ids"][0]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(QualityShiftAuditError, "manifest_calibration_ids"):
                audit_results(frame, output, 2, 3, [5, 10], 71)


if __name__ == "__main__":
    unittest.main()
