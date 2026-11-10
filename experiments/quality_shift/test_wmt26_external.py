"""Tiny synthetic tests for external WMT26 protocol; never research results."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from build_wmt26_transfer_matrices import TARGETS, select_panel, sha256_file
from run_wmt26_external import (
    METRICS, available_budgets, checked_matrix, partition_documents,
    run_language, aggregate, PRIMARY,
)
from audit_wmt26_external import audit_language, ExternalAuditError


def synthetic(n_docs=40):
    rng = np.random.default_rng(45)
    docs = [f"test_#_doc{i//3}" for i in range(3 * n_docs)]
    n = len(docs)
    matrix = {"sentence_id": [f"{doc}_#_{i%3}" for i, doc in enumerate(docs)],
              "source_text": ["x" * (10 + i % 61) for i in range(n)],
              "document_id": docs,
              "domain": ["test"] * n}
    for i, model in enumerate(TARGETS):
        matrix[model] = 80. + i + rng.normal(0, 3, n)
    dev = [f"dev{i:02d}" for i in range(12)]
    for i, model in enumerate(dev):
        matrix[model] = 50. + i + rng.normal(0, 4, n)
    return pd.DataFrame(matrix), {"language_pair": PRIMARY[0],
                                  "target_models": list(TARGETS),
                                  "development_models_by_coverage": dev}


class FrozenSystemPanelTests(unittest.TestCase):
    def test_panel_depends_only_on_observation_masks(self):
        universe = set(range(110))
        coverage = {m: set(universe) for m in TARGETS}
        for i in range(13):
            coverage[f"dev{i:02d}"] = set(range(110 - i))
        coverage["Human (from scratch)"] = universe
        chosen, target, core, missing = select_panel(coverage)
        self.assertEqual(chosen, [f"dev{i:02d}" for i in range(12)])
        self.assertEqual(len(target), 110)
        self.assertEqual(len(core), 99)
        self.assertFalse(missing)

    def test_target_is_never_replaced_with_contemporaneous_best(self):
        coverage = {m: set(range(50)) for m in TARGETS[:-1]}
        coverage.update({f"dev{i}": set(range(50)) for i in range(12)})
        chosen, _, core, missing = select_panel(coverage)
        self.assertEqual(chosen, [])
        self.assertFalse(core)
        self.assertEqual(missing, [TARGETS[-1]])

    def test_partitions_respect_whole_documents(self):
        df, _ = synthetic()
        cal = partition_documents(df.document_id.to_numpy(), seed=89, split=2)
        self.assertFalse(set(df.document_id[cal]) & set(df.document_id[~cal]))
        budgets, low, high = available_budgets(df, (10, 25, 50, 100), 8, 89)
        self.assertGreaterEqual(low, 25)
        self.assertGreaterEqual(high, low)
        self.assertIn(25, budgets)


class WMT26RunnerTests(unittest.TestCase):
    def test_column_validation_and_checksums(self):
        df, panel = synthetic()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "panel.csv"
            df.to_csv(path, index=False)
            panel["matrix_sha256"] = sha256_file(path)
            got = checked_matrix(path, panel)
            self.assertEqual(len(got), len(df))
            df["sentence_id"] = "duplicate"
            df.to_csv(path, index=False)
            panel["matrix_sha256"] = sha256_file(path)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                checked_matrix(path, panel)

    def test_target_evaluation_perturbation_does_not_change_any_selection(self):
        frame, panel = synthetic()
        with tempfile.TemporaryDirectory() as d:
            out1, out2 = Path(d) / "a", Path(d) / "b"
            run_language(frame, panel, splits=2, trials=3,
                         budgets=[10, 25], seed=89, output=out1)
            changed = frame.copy()
            for index, name in enumerate(TARGETS):
                changed[name] = np.clip(90 - 2 * index + np.sin(np.arange(len(changed))), 0, 100)
            run_language(changed, panel, splits=2, trials=3,
                         budgets=[10, 25], seed=89, output=out2)
            one, two = pd.read_csv(out1 / "trials.csv"), pd.read_csv(out2 / "trials.csv")
            self.assertEqual(len(one), 144)
            self.assertEqual(len(one), len(two))
            self.assertEqual(one.selected_indices.tolist(), two.selected_indices.tolist())
            self.assertEqual(json.loads((out1 / "split_manifest.json").read_text()),
                             json.loads((out2 / "split_manifest.json").read_text()))
            self.assertNotEqual(one.mae.mean(), two.mae.mean())
            self.assertEqual(len(pd.read_csv(out1 / "quality_gap_audit.csv")), 6)
            self.assertEqual(len(pd.read_csv(out1 / "summary.csv")), 36)

    def test_equal_language_aggregation_and_random_zero_delta(self):
        df, panel = synthetic()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            lang = PRIMARY[0]
            run_language(df, panel, splits=2, trials=2, budgets=[10, 25],
                         seed=91, output=root / "languages" / lang)
            aggregate(root, [lang], bootstraps=100, seed=4)
            macro = pd.read_csv(root / "macro_summary.csv")
            paired = pd.read_csv(root / "paired_language_comparisons.csv")
            self.assertIn("wmt25_overlap_primary", macro.cohort.unique())
            zero = paired.loc[paired.method == "Random", "gain_vs_random"].dropna()
            self.assertTrue((abs(zero) < 1e-12).all())
            self.assertTrue(set(METRICS) <= set(paired.metric))


class WMT26IndependentAuditTests(unittest.TestCase):
    def test_audits_synthetic_run_and_rejects_changed_metrics(self):
        frame, panel = synthetic()
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "language"
            run_language(frame, panel, splits=2, trials=3, budgets=[10, 25],
                         seed=89, output=folder)
            report = audit_language(frame, panel, folder, seed=89, splits=2,
                                    trials=3, budgets=[10, 25])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["checked_trials"], 144)
            tampered = pd.read_csv(folder / "trials.csv")
            tampered.loc[0, "mae"] += 1
            tampered.to_csv(folder / "trials.csv", index=False)
            with self.assertRaisesRegex(ExternalAuditError, "trial-zero mae"):
                audit_language(frame, panel, folder, seed=89, splits=2,
                               trials=3, budgets=[10, 25])

    def test_rejects_changed_quality_gap(self):
        frame, panel = synthetic()
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "language"
            run_language(frame, panel, splits=1, trials=2, budgets=[10],
                         seed=9, output=folder)
            tampered = pd.read_csv(folder / "quality_gap_audit.csv")
            tampered.loc[0, "evaluation_mean_quality_gap"] -= 2
            tampered.to_csv(folder / "quality_gap_audit.csv", index=False)
            with self.assertRaisesRegex(ExternalAuditError, "quality-gap evaluation_mean_quality_gap"):
                audit_language(frame, panel, folder, seed=9, splits=1,
                               trials=2, budgets=[10])


if __name__ == "__main__":
    unittest.main()
