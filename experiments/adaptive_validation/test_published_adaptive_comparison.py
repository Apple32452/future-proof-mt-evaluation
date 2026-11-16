"""Ten checks for the adapted published allocation policies and shared engine."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from published_adaptive_comparison import (published_action, rank_weights,
                                           run_one, run_published_policy)
from pairwise_racing import RaceState
from run_comparison import load_matrix


class TestComparisons(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2026)
        self.y = np.clip(np.linspace(.8, .3, 6)[None, :]
                         + rng.normal(0, .07, (60, 6)), 0, 1)

    def test_rank_weights(self):
        state = RaceState(60, 6)
        for i in range(15):
            for j in range(6):
                state.observe_next(j, float(self.y[i, j]))
        weights = rank_weights(state)
        self.assertEqual(weights.shape, (6,))
        self.assertGreater(weights[0], weights[-1])

    def test_published_policies_deterministic(self):
        for policy in ("published_confusion", "published_inv_rank2"):
            self.assertEqual(run_one(self.y, policy, seed=101),
                             run_one(self.y, policy, seed=101))

    def test_all_policies_on_separated_matrix(self):
        for policy in ("published_confusion", "published_inv_rank2",
                       "common_active", "random_pair", "certificate_racing_v2"):
            out = run_one(self.y, policy, seed=555)
            self.assertEqual(out["wrong_certificates"], 0)
            self.assertGreaterEqual(out["cost_90"], 15 * 6)
            self.assertLessEqual(out["cost_90"], self.y.size)

    def test_query_exactly_once(self):
        out = run_published_policy(self.y, "published_inv_rank2", seed=9)
        self.assertEqual(out["cost_cells"], out["queried_unique"])

    def test_score_bounds_enforced(self):
        bad = self.y.copy()
        bad[0, 0] = 1.2
        with self.assertRaises(ValueError):
            run_one(bad, "published_inv_rank2", seed=8)

    def test_warmup_limit(self):
        with self.assertRaises(ValueError):
            run_one(self.y[:10], "published_confusion", seed=2)

    def test_unknown_policy(self):
        with self.assertRaises(ValueError):
            run_one(self.y, "nonexistent", seed=8)

    def test_tied_matrix_census(self):
        matrix = np.full((35, 5), .5)
        for policy in ("published_confusion", "published_inv_rank2",
                       "certificate_racing_v2"):
            self.assertEqual(run_one(matrix, policy, seed=33)["wrong_certificates"], 0)

    def test_csv_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matrix.csv"
            with path.open("w") as handle:
                writer = csv.writer(handle)
                writer.writerow(["item_id", "doc_id", "a", "b"])
                for i in range(40):
                    writer.writerow([str(i), str(i // 10), .8, .4])
            y, meta = load_matrix(path)
            self.assertEqual(y.shape, (40, 2))
            self.assertEqual(meta["n_items"], 40)
            with path.open("a") as handle:
                handle.write("1,d,0.2,0.4\n")
            with self.assertRaises(ValueError):
                load_matrix(path)

    def test_actions_only_from_observed_prefix(self):
        rng = np.random.default_rng(5)
        state = RaceState(25, 4)
        for _ in range(15):
            for system in range(4):
                state.observe_next(system, float(rng.random()))
        for policy in ("published_confusion", "published_inv_rank2"):
            chosen = published_action(state, np.random.default_rng(12), policy)
            self.assertIn(chosen, range(4))


if __name__ == "__main__":
    unittest.main()
