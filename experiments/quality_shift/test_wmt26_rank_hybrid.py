"""Synthetic regression tests of the frozen ranking-aware WMT26 protocol."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from audit_wmt26_rank_hybrid import RankHybridAuditError, audit_language
from rank_hybrid_methods import hybrid_allocation, rank_aware_hybrid_sample, rank_information_bins
from run_wmt26_external import partition_documents
from run_wmt26_quality_calibrated import calibrated_systems
from run_wmt26_rank_hybrid import METHODS, run_language, select
from test_wmt26_quality_calibrated import synthetic


class HybridAllocationTests(unittest.TestCase):
    def test_allocation_and_weights(self):
        counts = np.full(5, 20)
        allocated = hybrid_allocation(counts, 25)
        np.testing.assert_array_equal(allocated[:3], [2, 2, 2])
        self.assertEqual(int(allocated.sum()), 25)
        self.assertEqual(int(allocated[3] + allocated[4]), 19)
        rng = np.random.default_rng(49)
        dev = rng.normal(60, 8, (100, 6))
        chosen, weights, bins, utility = rank_aware_hybrid_sample(dev, 25, rng)
        self.assertEqual(len(set(chosen)), 25)
        self.assertEqual(int(np.bincount(bins, minlength=5).sum()), 100)
        self.assertAlmostEqual(float(weights.sum()), 1, places=12)
        self.assertTrue((weights > 0).all())
        self.assertTrue((utility >= 0).all())

    def test_nearly_full_budget_has_valid_fallback(self):
        dev = np.arange(150, dtype=float).reshape(25, 6)
        for budget in [5, 10, 25]:
            picked, w, _, _ = rank_aware_hybrid_sample(dev, budget, np.random.default_rng(3))
            self.assertEqual(len(set(picked)), budget)
            self.assertAlmostEqual(float(w.sum()), 1, places=12)
        allocation = hybrid_allocation(np.array([5] * 5), 25)
        np.testing.assert_array_equal(allocation, [5, 5, 5, 5, 5])

    def test_reproducible_and_target_free(self):
        rng = np.random.default_rng(23)
        dev = rng.normal(50, 10, (75, 6))
        labels1, utils1 = rank_information_bins(dev)
        labels2, utils2 = rank_information_bins(dev.copy())
        np.testing.assert_array_equal(labels1, labels2)
        np.testing.assert_array_equal(utils1, utils2)
        for method in METHODS:
            from run_wmt26_quality_calibrated import standardize_development, vector_kmeans
            standardized = standardize_development(dev, dev)
            clusters = vector_kmeans(standardized, 17)
            selections = [select(method, dev, np.arange(75, dtype=float),
                                 dev.mean(axis=0), standardized, clusters, 10,
                                 np.random.default_rng(66)) for _ in range(2)]
            np.testing.assert_array_equal(selections[0][0], selections[1][0])
            np.testing.assert_array_equal(selections[0][1], selections[1][1])


class SyntheticExperimentTests(unittest.TestCase):
    def test_perturb_target_evaluation_preserves_all_selections(self):
        frame, panel = synthetic()
        seed = 39
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "first", Path(temp) / "second"
            run_language(frame, panel, seed=seed, splits=1, trials=2,
                         budgets=[10, 25], output=first)
            changed = frame.copy()
            cal = partition_documents(frame.document_id.to_numpy(), seed=seed, split=0)
            names = panel["target_models"] + panel["development_models_by_coverage"]
            high, _, _ = calibrated_systems(frame[names].to_numpy(float), cal)
            for i in high:
                changed.loc[~cal, names[i]] = float(i + 70)
            run_language(changed, panel, seed=seed, splits=1, trials=2,
                         budgets=[10, 25], output=second)
            a = pd.read_csv(first / "trials.csv")
            b = pd.read_csv(second / "trials.csv")
            self.assertEqual(len(a), 20)
            self.assertEqual(a.selected_indices.tolist(), b.selected_indices.tolist())
            self.assertEqual(a.weights.tolist(), b.weights.tolist())
            self.assertNotAlmostEqual(a.mae.mean(), b.mae.mean())
            self.assertEqual(len(json.loads((first / "split_manifest.json").read_text())), 1)

    def test_independent_replay_detects_rank_metric_tampering(self):
        frame, panel = synthetic()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "run"
            run_language(frame, panel, seed=42, splits=2, trials=2,
                         budgets=[10, 25], output=folder)
            report = audit_language(frame, panel, folder, seed=42, splits=2,
                                    trials=2, budgets=[10, 25])
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["checked_selections"], 40)
            tampered = pd.read_csv(folder / "trials.csv")
            tampered.loc[0, "pairwise_accuracy"] = 0.123
            tampered.to_csv(folder / "trials.csv", index=False)
            with self.assertRaisesRegex((ValueError, AssertionError), "independent pairwise_accuracy"):
                audit_language(frame, panel, folder, seed=42, splits=2,
                               trials=2, budgets=[10, 25])


if __name__ == "__main__":
    unittest.main()
