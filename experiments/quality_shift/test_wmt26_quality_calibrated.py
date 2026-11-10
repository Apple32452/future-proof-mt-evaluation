"""Synthetic-only protocol checks for the exploratory stronger-system arm."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from build_wmt26_transfer_matrices import TARGETS
from run_wmt26_external import PRIMARY, partition_documents
from run_wmt26_quality_calibrated import (
    METHODS, calibrated_systems, choose, cluster_proportional_indices,
    mean_balance_indices, run_language, standardize_development, vector_kmeans,
)
from audit_wmt26_quality_calibrated import audit_language, QualityCalibratedAuditError


def synthetic(n_docs=24):
    rng = np.random.default_rng(144)
    documents = [f"test_#_doc{i//3}" for i in range(3 * n_docs)]
    data = {"sentence_id": [f"{doc}_#_{i%3}" for i, doc in enumerate(documents)],
            "source_text": ["x" * (i % 23 + 10) for i in range(len(documents))],
            "document_id": documents, "domain": ["test"] * len(documents)}
    models = list(TARGETS) + [f"dev{i:02d}" for i in range(12)]
    for i, name in enumerate(models):
        data[name] = np.clip(95 - 3 * i + rng.normal(0, 2.0, len(documents)), 0, 100)
    return pd.DataFrame(data), {"language_pair": PRIMARY[0], "target_models": list(TARGETS),
                                 "development_models_by_coverage": models[6:]}


class CalibratedRolesTests(unittest.TestCase):
    def test_roles_derive_from_calibration_only(self):
        frame, panel = synthetic()
        scores = frame[panel["target_models"] + panel["development_models_by_coverage"]].to_numpy(float)
        cal = partition_documents(frame.document_id.to_numpy(), seed=91, split=0)
        target, dev, _ = calibrated_systems(scores, cal)
        self.assertEqual(len(target), 6)
        self.assertEqual(len(dev), 6)
        self.assertFalse(set(target) & set(dev))
        changed = scores.copy()
        changed[~cal] = np.flip(changed[~cal], axis=1)
        new_target, new_dev, _ = calibrated_systems(changed, cal)
        np.testing.assert_array_equal(target, new_target)
        np.testing.assert_array_equal(dev, new_dev)

    def test_new_selectors_are_dev_only_and_deterministic(self):
        frame, panel = synthetic()
        data = frame[panel["target_models"] + panel["development_models_by_coverage"]].to_numpy(float)
        cal = partition_documents(frame.document_id.to_numpy(), seed=25, split=0)
        _, low, _ = calibrated_systems(data, cal)
        eval_dev = data[~cal][:, low]
        normalized = standardize_development(eval_dev, data[cal][:, low])
        first = vector_kmeans(normalized, seed=123)
        second = vector_kmeans(normalized, seed=123)
        np.testing.assert_array_equal(first, second)
        pick = mean_balance_indices(normalized, 25)
        self.assertEqual(len(np.unique(pick)), 25)
        idx, weight = cluster_proportional_indices(first, 25, np.random.default_rng(4))
        self.assertEqual(len(np.unique(idx)), 25)
        self.assertAlmostEqual(weight.sum(), 1, places=12)
        self.assertTrue((weight > 0).all())
        for method in METHODS:
            idx, weight = choose(method, eval_dev, np.arange(len(eval_dev), dtype=float),
                                 data[cal][:, low].mean(axis=0), normalized, first,
                                 10, np.random.default_rng(7))
            self.assertEqual(len(set(idx)), 10)
            self.assertAlmostEqual(weight.sum(), 1, places=12)


class FullSyntheticRunTests(unittest.TestCase):
    def test_eval_only_target_change_leaves_selections_unchanged(self):
        frame, panel = synthetic()
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "one", Path(temp) / "two"
            run_language(frame, panel, seed=39, splits=1, trials=2, budgets=[10, 25], output=first)
            changed = frame.copy()
            cal = partition_documents(frame.document_id.to_numpy(), seed=39, split=0)
            names = panel["target_models"] + panel["development_models_by_coverage"]
            high, low, _ = calibrated_systems(frame[names].to_numpy(float), cal)
            for index in high:
                changed.loc[~cal, names[index]] = 77 + index * 0.04
            run_language(changed, panel, seed=39, splits=1, trials=2, budgets=[10, 25], output=second)
            a = pd.read_csv(first / "trials.csv")
            b = pd.read_csv(second / "trials.csv")
            self.assertEqual(len(a), 24)
            self.assertEqual(a.selected_indices.tolist(), b.selected_indices.tolist())
            self.assertEqual(a.weights.tolist(), b.weights.tolist())
            self.assertNotAlmostEqual(a.mae.mean(), b.mae.mean())
            manifests = json.loads((first / "split_manifest.json").read_text())
            self.assertEqual(len(manifests), 1)
            self.assertEqual(len(manifests[0]["targets_selected_from_calibration"]), 6)
            self.assertFalse(set(frame.document_id[cal]) & set(frame.document_id[~cal]))
            self.assertEqual(len(pd.read_csv(first / "quality_gap_audit.csv")), 1)
            self.assertEqual(len(pd.read_csv(first / "summary.csv")), 16)

    def test_independent_metric_audit_rejects_tampering(self):
        frame, panel = synthetic()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "run"
            run_language(frame, panel, seed=42, splits=2, trials=2,
                         budgets=[10, 25], output=folder)
            evidence = audit_language(frame, panel, folder, seed=42, splits=2,
                                      trials=2, budgets=[10, 25])
            self.assertEqual(evidence["status"], "passed")
            self.assertEqual(evidence["checked_selections"], 48)
            corrupted = pd.read_csv(folder / "trials.csv")
            corrupted.loc[0, "pairwise_accuracy"] = 0.123
            corrupted.to_csv(folder / "trials.csv", index=False)
            with self.assertRaisesRegex(ValueError, "independent pairwise_accuracy"):
                audit_language(frame, panel, folder, seed=42, splits=2,
                               trials=2, budgets=[10, 25])


if __name__ == "__main__":
    unittest.main()
