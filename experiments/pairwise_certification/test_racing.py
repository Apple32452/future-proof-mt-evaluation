"""Independent contract and mathematical sanity tests for pairwise racing.

Synthetic fixtures validate implementation behavior, not real MT efficiency or
the novelty of the scheduling heuristic.
"""
from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from pairwise_racing import (
    BET_FRACTIONS, POLICIES, UNKNOWN, RaceState,
    betting_factors, log_mixture, run_race,
)


class BettingPrimitiveTests(unittest.TestCase):
    def test_factors_at_half_null(self) -> None:
        actual = betting_factors(np.array([0.0, 0.5, 1.0]), np.array(0.5))
        expected = np.stack((1 - BET_FRACTIONS,
                             np.ones(len(BET_FRACTIONS)),
                             1 + BET_FRACTIONS))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
        self.assertTrue(np.all(actual > 0))

    def test_neutral_infeasible_null_boundary(self) -> None:
        result = betting_factors(np.array([0.2, 0.0, 0.8]),
                                 np.array([0.0, -0.1, 1e-15]))
        np.testing.assert_array_equal(result, np.ones((3, len(BET_FRACTIONS))))

    def test_log_mixture_matches_direct_calculation(self) -> None:
        x = np.array([[0.2, 1.0, 4.0], [0.01, 0.1, 1.0]])
        np.testing.assert_allclose(log_mixture(np.log(x)), np.log(x.mean(axis=-1)))

    def test_log_mixture_stable_large_positive_and_negative(self) -> None:
        logs = np.array([[1000.0, 999.0, 998.0], [-1000.0, -1001.0, -1002.0]])
        expected = np.array([1000.0, -1000.0]) + math.log((1 + math.exp(-1) + math.exp(-2)) / 3)
        actual = log_mixture(logs)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected)

    def test_exact_null_population_eprocess_expectation(self) -> None:
        # Four fixed values have finite-population mean exactly 1/2. Exhaust all
        # 4! uniform orders; each fixed-bet product must have expectation one at
        # every fixed prefix. This is a sanity check, not a general proof.
        population = [0.0, 0.25, 0.75, 1.0]
        products_by_order = []
        for order in itertools.permutations(population):
            products = np.ones(len(BET_FRACTIONS))
            prefix = []
            observed_sum = 0.0
            for n, value in enumerate(order):
                conditional_null = (2.0 - observed_sum) / (4 - n)
                products *= betting_factors(np.array(value), np.array(conditional_null))
                prefix.append(products.copy())
                observed_sum += value
            products_by_order.append(prefix)
        all_products = np.asarray(products_by_order)
        np.testing.assert_allclose(all_products.mean(axis=0), np.ones((4, len(BET_FRACTIONS))),
                                   rtol=0, atol=2e-15)
        mixture_paths = all_products.mean(axis=-1)
        for threshold in (1.5, 2.0, 3.0):
            crossing_probability = np.mean(mixture_paths.max(axis=1) >= threshold)
            self.assertLessEqual(crossing_probability, 1 / threshold + 1e-15)


class RaceStateTests(unittest.TestCase):
    def test_config_validation(self) -> None:
        for n, k, alpha in [(0, 2, .05), (-1, 2, .05), (3.5, 2, .05),
                            (3, 1, .05), (3, 0, .05), (3, 2, 0),
                            (3, 2, 1), (3, 2, float("nan"))]:
            with self.subTest(n=n, k=k, alpha=alpha), self.assertRaises(ValueError):
                RaceState(n, k, alpha)

    def test_observation_range_rejection_does_not_charge_cost(self) -> None:
        state = RaceState(5, 2)
        for value in (-0.1, 1.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                state.observe_next(0, value)
            self.assertEqual(state.cost, 0)
        for system in (-1, 2):
            with self.subTest(system=system), self.assertRaises(ValueError):
                state.observe_next(system, .5)

    def test_rejects_query_after_system_census(self) -> None:
        state = RaceState(1, 2)
        state.observe_next(0, .5)
        with self.assertRaisesRegex(ValueError, "fully observed"):
            state.observe_next(0, .5)
        self.assertEqual(state.cost, 1)

    def test_completion_bounds_cover_true_differences_at_every_state(self) -> None:
        rng = np.random.default_rng(709)
        y = rng.uniform(size=(13, 4))
        state = RaceState(*y.shape)
        true_differences = y.mean(axis=0)[state.pair_a] - y.mean(axis=0)[state.pair_b]
        for _ in range(y.size):
            available = np.flatnonzero(state.counts < state.n_items)
            system = int(rng.choice(available))
            state.observe_next(system, float(y[state.counts[system], system]))
            lower, upper = state.completion_bounds()
            self.assertTrue(np.all(lower <= true_differences + 1e-14))
            self.assertTrue(np.all(upper >= true_differences - 1e-14))
            np.testing.assert_array_equal(state.pair_n,
                                          np.minimum(state.counts[state.pair_a], state.counts[state.pair_b]))
        lower, upper = state.completion_bounds()
        np.testing.assert_allclose(lower, true_differences, atol=1e-14)
        np.testing.assert_allclose(upper, true_differences, atol=1e-14)

    def test_reuse_updates_multiple_pairs_without_extra_cost(self) -> None:
        state = RaceState(10, 3)
        state.observe_next(0, .6)
        state.observe_next(1, .5)
        np.testing.assert_array_equal(state.pair_n, [1, 0, 0])
        state.observe_next(2, .4)
        self.assertEqual(state.cost, 3)
        np.testing.assert_array_equal(state.pair_n, [1, 1, 1])
        np.testing.assert_allclose(state.sum_d, [.1, .2, .1])

    def test_exact_ties_are_not_certified_before_pair_census(self) -> None:
        state = RaceState(7, 3)
        for row in range(7):
            for system in range(3):
                state.observe_next(system, .5)
                for p in np.flatnonzero(state.decisions != UNKNOWN):
                    self.assertEqual(state.decisions[p], 0)
                    self.assertEqual(state.reasons[p], "exact_census")
                    self.assertEqual(state.counts[state.pair_a[p]], 7)
                    self.assertEqual(state.counts[state.pair_b[p]], 7)
        self.assertTrue(state.complete)
        np.testing.assert_array_equal(state.decisions, [0, 0, 0])

    def test_near_zero_census_gap_is_not_silently_a_tie(self) -> None:
        state = RaceState(2, 2)
        for _ in range(2):
            state.observe_next(0, float(np.nextafter(.5, 1.0)))
            state.observe_next(1, .5)
        self.assertEqual(state.decisions.tolist(), [1])
        self.assertEqual(state.reasons, ["exact_census"])

    def test_complement_symmetry_for_fixed_observation_schedule(self) -> None:
        y = np.array([[.125, .625, .25], [.75, .25, .5],
                      [.875, .375, .625], [.5, .0, .125]])
        left, right = RaceState(4, 3), RaceState(4, 3)
        for row in range(4):
            for system in (2, 0, 1):
                left.observe_next(system, y[row, system])
                right.observe_next(system, 1.0 - y[row, system])
                np.testing.assert_allclose(left.sum_d, -right.sum_d)
                np.testing.assert_allclose(left.log_products[:, 0], right.log_products[:, 1])
                np.testing.assert_allclose(left.log_products[:, 1], right.log_products[:, 0])
        np.testing.assert_array_equal(left.decisions, -right.decisions)


class RacingControllerTests(unittest.TestCase):
    def test_invalid_run_configuration(self) -> None:
        cases = [{"policy": "not-a-policy"}, {"seed": -1}, {"seed": 1.5},
                 {"warmup_rows": -1}, {"warmup_rows": 1.5},
                 {"budget": -1}, {"budget": 7}, {"budget": 1.5}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_race(3, 2, lambda row, system: .5, **kwargs)

    def test_zero_budget_never_calls_oracle(self) -> None:
        def fail_query(row, system):
            raise AssertionError("zero-budget race must not query")
        result = run_race(10, 3, fail_query, budget=0, checkpoints=(0, 5), record_trace=True)
        self.assertEqual(result["cost_cells"], 0)
        self.assertEqual(result["trace"], [])
        self.assertEqual(result["resolved_pairs"], 0)
        self.assertEqual(result["snapshots"]["0"]["cost_cells"], 0)
        self.assertFalse(result["complete"])

    def test_constant_ordered_systems_all_policies(self) -> None:
        y = np.tile([1.0, .5, 0.0], (80, 1))
        for policy in POLICIES:
            with self.subTest(policy=policy):
                result = run_race(80, 3, lambda row, system: y[row, system],
                                  policy=policy, seed=7)
                self.assertTrue(result["complete"])
                self.assertEqual(result["decisions"], [1, 1, 1])
                self.assertLess(result["cost_cells"], y.size)

    def test_all_policies_reproducible_prefix_only_and_cost_consistent(self) -> None:
        y = np.random.default_rng(891).uniform(size=(15, 4))
        for policy in POLICIES:
            with self.subTest(policy=policy):
                queried = []
                def query(row, system):
                    self.assertIsInstance(row, int)
                    self.assertIsInstance(system, int)
                    self.assertTrue(0 <= row < 15)
                    self.assertTrue(0 <= system < 4)
                    queried.append((row, system))
                    return y[row, system]
                first = run_race(15, 4, query, policy=policy, seed=42,
                                 budget=43, record_trace=True)
                second = run_race(15, 4, lambda row, system: y[row, system],
                                  policy=policy, seed=42, budget=43, record_trace=True)
                self.assertEqual(first, second)
                self.assertEqual(len(queried), first["cost_cells"])
                self.assertEqual(len(set(queried)), len(queried))
                self.assertEqual(sum(first["system_counts"]), first["cost_cells"])
                self.assertLessEqual(first["cost_cells"], 43)
                order = np.random.default_rng(np.random.SeedSequence([42, 1])).permutation(15)
                for system in range(4):
                    seen = [row for row, s in queried if s == system]
                    self.assertEqual(seen, order[:len(seen)].tolist())
                for cost, (entry, (row, system)) in enumerate(zip(first["trace"], queried), start=1):
                    self.assertEqual(entry, {"cost": cost, "row": row,
                                             "system": system, "value": y[row, system]})

    def test_snapshots_hit_requested_budgets_and_decisions_are_monotone(self) -> None:
        # Identical .5 labels prevent early completion before the full census.
        for policy in POLICIES:
            with self.subTest(policy=policy):
                result = run_race(8, 3, lambda row, system: .5, policy=policy,
                                  seed=4, budget=23, checkpoints=range(24))
                previous = [UNKNOWN] * 3
                for budget in range(24):
                    snap = result["snapshots"][str(budget)]
                    self.assertEqual(snap["cost_cells"], budget)
                    for before, after in zip(previous, snap["decisions"]):
                        if before != UNKNOWN:
                            self.assertEqual(after, before)
                    previous = snap["decisions"]
                self.assertEqual(result["cost_cells"], 23)
                self.assertFalse(result["complete"])

    def test_early_stop_snapshots_do_not_fabricate_extra_labels(self) -> None:
        y = np.tile([1.0, 0.0], (100, 1))
        result = run_race(100, 2, lambda row, system: y[row, system],
                          seed=5, checkpoints=(0, 100, 200), record_trace=True)
        self.assertTrue(result["complete"])
        self.assertLess(result["cost_cells"], 100)
        self.assertEqual(result["snapshots"]["100"]["cost_cells"], result["cost_cells"])
        self.assertEqual(result["snapshots"]["200"], result["snapshots"]["100"])
        self.assertEqual(result["trace"][-1]["cost"], result["cost_cells"])

    def test_full_runs_finish_without_exceeding_census_budget(self) -> None:
        y = np.random.default_rng(28).uniform(size=(9, 4))
        true_signs = np.sign(y.mean(axis=0)[np.triu_indices(4, 1)[0]] -
                             y.mean(axis=0)[np.triu_indices(4, 1)[1]]).astype(int).tolist()
        for policy in POLICIES:
            with self.subTest(policy=policy):
                result = run_race(9, 4, lambda row, system: y[row, system], policy=policy, seed=6)
                self.assertTrue(result["complete"])
                self.assertLessEqual(result["cost_cells"], y.size)
                self.assertEqual(result["decisions"], true_signs)
                self.assertTrue(all(n <= 9 for n in result["system_counts"]))


if __name__ == "__main__":
    unittest.main()
