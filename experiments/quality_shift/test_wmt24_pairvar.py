"""Synthetic, target-blind safety and independent auditing tests for pairvar."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from audit_wmt24_pairvar import PairvarAuditError, audit_language
from pairvar_sampling import (
    allocated_counts, calibration_design, draw_strata,
    make_strata, pairwise_dispersion, role_weights,
)
from run_wmt24_pairvar import (
    METHODS, make_select_context, run_language, select, select_seed,
    split_three_document_sets,
)
from run_wmt26_quality_calibrated import vector_kmeans


def synthetic_frame(docs: int = 36, per_doc: int = 5):
    rng = np.random.default_rng(801)
    data = []
    names = [f"system_{i}" for i in range(10)]
    qualities = np.array([89, 86, 83, 80, 72, 68, 54, 51, 47, 43], dtype=float)
    for doc in range(docs):
        for item in range(per_doc):
            d = f"synthetic_#_doc{doc:03d}"
            difficulty = rng.normal(0, 8)
            scores = np.clip(qualities - difficulty + rng.normal(0, 6, size=10), 0, 100)
            data.append([f"{d}_#_{item}", d, "synthetic", *scores])
    return pd.DataFrame(data, columns=["sentence_id", "document_id", "domain", *names]), names


class PairvarMethodTests(unittest.TestCase):
    def test_three_tertiles_equal_population_and_calibration_mapping(self):
        dev = np.arange(0, 80).reshape(20, 4).astype(float)
        labels, cal, cut = make_strata(dev, dev[:8])
        self.assertEqual(np.bincount(labels).tolist(), [7, 7, 6])
        self.assertEqual(len(cal), 8)
        self.assertEqual(len(cut), 2)

    def test_pair_weights_and_shrinkage_finite(self):
        role = np.array([[85, 84, 65, 35], [86, 83, 66, 35]], dtype=float)
        weights = role_weights(role)
        self.assertEqual(len(weights), 6)
        self.assertAlmostEqual(weights.sum(), 1)
        self.assertGreater(weights[0], weights[-1])
        variances = pairwise_dispersion(np.ones((5, 4)), np.zeros(5, dtype=int), weights)
        np.testing.assert_allclose(variances, np.zeros(3))

    def test_neyman_allocation_respects_bounds(self):
        sizes = np.array([3, 45, 2])
        for spread in (None, np.zeros(3), np.array([100., 0.1, 20.])):
            for budget in (3, 10, 25, 50):
                counts = allocated_counts(sizes, budget, spread)
                self.assertEqual(int(counts.sum()), budget)
                self.assertTrue(np.all(counts >= 1))
                self.assertTrue(np.all(counts <= sizes))

    def test_weights_recover_population_mean_on_constant_strata(self):
        labels = np.repeat(np.arange(3), [10, 20, 30])
        values = np.repeat([12., 33., 72.], [10, 20, 30])
        for spread in (None, np.array([2., 30., 5.])):
            counts = allocated_counts(np.array([10, 20, 30]), 20, spread)
            ids, weights = draw_strata(labels, counts, np.random.default_rng(91))
            self.assertAlmostEqual(float(weights @ values[ids]), float(values.mean()))
            self.assertAlmostEqual(float(weights.sum()), 1.)

    def test_calibration_target_perturbation_preserves_selections(self):
        frame, names = synthetic_frame()
        scores = frame[names].to_numpy(float)
        docs = frame.document_id.to_numpy()
        role, var, ev = split_three_document_sets(docs, 20261112, 0)
        tgt, dev, _, dy, cal, design, normalized = make_select_context(scores, role, var, ev)
        clusters = vector_kmeans(normalized, seed=15)
        before = []
        for mi, method in enumerate(METHODS):
            before.append(select(method, dy, cal, normalized, clusters, design, 10,
                                 select_seed(20261112, 0, mi, 10, 0)))
        different = scores.copy()
        different[np.ix_(ev, tgt)] = 100 - different[np.ix_(ev, tgt)]
        t2, d2, _, dy2, cal2, design2, std2 = make_select_context(different, role, var, ev)
        np.testing.assert_array_equal(tgt, t2)
        np.testing.assert_array_equal(dev, d2)
        clusters2 = vector_kmeans(std2, seed=15)
        for mi, method in enumerate(METHODS):
            after = select(method, dy2, cal2, std2, clusters2, design2, 10,
                           select_seed(20261112, 0, mi, 10, 0))
            np.testing.assert_array_equal(before[mi][0], after[0])
            np.testing.assert_allclose(before[mi][1], after[1])

    def test_documents_partition_into_three_disjoint_roles(self):
        frame, _ = synthetic_frame()
        docs = frame.document_id.to_numpy()
        a, b, c = split_three_document_sets(docs, 20261112, 2)
        self.assertEqual(set(a) | set(b) | set(c), set(range(len(frame))))
        self.assertFalse(set(docs[a]) & set(docs[b]))
        self.assertFalse(set(docs[a]) & set(docs[c]))
        self.assertFalse(set(docs[b]) & set(docs[c]))

    def test_full_synthetic_run_independent_audit_and_tampering(self):
        frame, names = synthetic_frame()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "test_run"
            run_language(frame, names, language="synthetic", seed=20261112,
                         splits=2, trials=3, budgets=[10, 25], output=out)
            check = lambda: audit_language(frame, names, out, seed=20261112,
                                           splits=2, trials=3, budgets=[10, 25])
            report = check()
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["checked_selections"], 2 * 2 * (5 * 3 + 1))

            trial_path = out / "trials.csv"
            original = pd.read_csv(trial_path)
            changed = original.copy()
            changed.loc[0, "pairwise_accuracy"] = 17.0
            changed.to_csv(trial_path, index=False)
            with self.assertRaises(ValueError):
                check()
            original.to_csv(trial_path, index=False)

            modified = original.copy()
            modified.loc[0, "selected_indices"] = json.dumps([0] * 10)
            modified.to_csv(trial_path, index=False)
            with self.assertRaises(ValueError):
                check()
            original.to_csv(trial_path, index=False)

            manifest_path = out / "split_manifest.json"
            manifests = json.loads(manifest_path.read_text())
            manifests[0]["role_calibration_ids"][0] = "tampered"
            manifest_path.write_text(json.dumps(manifests))
            with self.assertRaises(PairvarAuditError):
                check()


if __name__ == "__main__":
    unittest.main()
