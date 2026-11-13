"""Independent correctness tests for the acquisition-only v2 revision.

The scalar forecast below independently implements the frozen mathematical
rule; it does not call the prototype's factor or mixture functions. These are
implementation tests, not performance experiments or an MT validation set.
"""
from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np

import certificate_racing_v2 as v2
from pairwise_racing import RaceState, UNKNOWN


REFERENCE_C = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95)
REFERENCE_TOL = 1e-12


def scalar_log_mixture(logs):
    peak = max(logs)
    return peak + math.log(math.fsum(math.exp(value - peak) for value in logs) / len(logs))


def scalar_bet_update(logs, peaks, n_items, prior_n, prior_sum_d, next_d):
    x_values = ((next_d + 1) / 2, (1 - next_d) / 2)
    prefix_sums = ((prior_n + prior_sum_d) / 2, (prior_n - prior_sum_d) / 2)
    for side in range(2):
        q = (n_items / 2 - prefix_sums[side]) / (n_items - prior_n)
        ratio = x_values[side] / q if q > REFERENCE_TOL else 1.0
        for j, c in enumerate(REFERENCE_C):
            logs[side][j] += math.log(1.0 + c * (ratio - 1.0))
        peaks[side] = max(peaks[side], scalar_log_mixture(logs[side]))


def scalar_forecast(n_items, pair_n, sum_d, log_products, log_e_peak, log_threshold):
    """Independent step-by-step forecast including pair-only exact bounds."""
    n = int(pair_n)
    mean_d = float(sum_d) / n if n else 0.0
    mean_d = min(1.0, max(-1.0, mean_d))
    logs = [list(map(float, row)) for row in log_products]
    peaks = list(map(float, log_e_peak))
    for h in range(1, n_items - n + 1):
        scalar_bet_update(logs, peaks, n_items, n + h - 1,
                          float(sum_d) + (h - 1) * mean_d, mean_d)
        projected_sum = float(sum_d) + h * mean_d
        remaining = n_items - n - h
        lower = (projected_sum - remaining) / n_items
        upper = (projected_sum + remaining) / n_items
        crossed = [peak > log_threshold + 1e-10 for peak in peaks]
        if (lower > REFERENCE_TOL or upper < -REFERENCE_TOL
                or remaining == 0 or (crossed[0] != crossed[1])):
            return h
    raise AssertionError("a non-census forecast must terminate by census")


def make_prefix_state(n_items, differences):
    logs = [[0.0] * len(REFERENCE_C) for _ in range(2)]
    peaks = [0.0, 0.0]
    total = 0.0
    for n, d in enumerate(differences):
        scalar_bet_update(logs, peaks, n_items, n, total, float(d))
        total += float(d)
    return total, np.asarray(logs), np.asarray(peaks)


def state_arrays(state):
    names = ("counts", "sums", "cache", "pair_n", "sum_d", "sum_d2",
             "log_products", "log_e_peak", "decisions", "decision_costs")
    return {name: getattr(state, name).copy() for name in names}


class ForecastTests(unittest.TestCase):
    def test_vectorized_agrees_with_independent_scalar_forecast(self):
        rng = np.random.default_rng(401)
        n_items = 19
        counts = np.asarray([0, 1, 4, 9, 13, 18], dtype=int)
        states = [make_prefix_state(n_items, rng.uniform(-0.7, 0.7, size=int(n)))
                  for n in counts]
        sums = np.asarray([state[0] for state in states])
        logs = np.asarray([state[1] for state in states])
        peaks = np.asarray([state[2] for state in states])
        threshold = math.log(2 * len(counts) / 0.05)
        expected = [scalar_forecast(n_items, n, total, capital, peak, threshold)
                    for n, total, capital, peak in zip(counts, sums, logs, peaks)]
        actual = v2.forecast_paired_steps(n_items, counts, sums, logs, peaks, threshold)
        np.testing.assert_array_equal(actual, expected)

    def test_zero_mean_forecasts_census(self):
        counts = np.asarray([0, 4, 10, 12])
        actual = v2.forecast_paired_steps(13, counts, np.zeros(4),
                                          np.zeros((4, 2, 7)), np.zeros((4, 2)),
                                          math.log(1000))
        np.testing.assert_array_equal(actual, 13 - counts)

    def test_forecast_inputs_are_not_mutated(self):
        counts = np.asarray([2, 7])
        sums = np.asarray([0.2, -1.4])
        logs = np.full((2, 2, 7), -0.3)
        peaks = np.zeros((2, 2))
        arrays = (counts, sums, logs, peaks)
        originals = [array.copy() for array in arrays]
        for array in arrays:
            array.setflags(write=False)
        v2.forecast_paired_steps(11, *arrays, math.log(80))
        for array, original in zip(arrays, originals):
            np.testing.assert_array_equal(array, original)

    def test_forecast_bounds_across_small_populations(self):
        rng = np.random.default_rng(931)
        for n_items in (1, 2, 7, 25):
            counts = np.arange(n_items)
            states = [make_prefix_state(n_items, rng.uniform(-0.95, 0.95, size=int(n)))
                      for n in counts]
            sums = np.asarray([state[0] for state in states])
            logs = np.asarray([state[1] for state in states])
            peaks = np.asarray([state[2] for state in states])
            actual = np.asarray(v2.forecast_paired_steps(n_items, counts, sums, logs, peaks,
                                                         math.log(2 * n_items / 0.05)))
            self.assertEqual(actual.shape, counts.shape)
            self.assertTrue(np.all(actual >= 1))
            self.assertTrue(np.all(actual <= n_items - counts))
            self.assertTrue(np.all(actual == actual.astype(int)))

    def test_two_crossed_directions_do_not_create_a_certificate(self):
        threshold = math.log(40)
        actual = v2.forecast_paired_steps(10, np.asarray([4]), np.asarray([0.0]),
                                          np.full((1, 2, 7), threshold + 1),
                                          np.full((1, 2), threshold + 1), threshold)
        np.testing.assert_array_equal(actual, [6])

    def test_deterministic_completion_can_resolve_contradictory_bets(self):
        threshold = math.log(40)
        actual = v2.forecast_paired_steps(10, np.asarray([4]), np.asarray([4.0]),
                                          np.full((1, 2, 7), threshold + 1),
                                          np.full((1, 2), threshold + 1), threshold)
        # With all hypothetical differences equal to one, at m=5 the lower
        # bound is zero; at m=6 it is positive. Historical peak contradiction
        # must not block the deterministic certificate.
        np.testing.assert_array_equal(actual, [2])

    def test_uses_current_capital_and_retains_historical_peak(self):
        threshold = math.log(50)
        counts = np.asarray([5])
        sums = np.asarray([0.6])
        logs = np.full((1, 2, 7), -5.0)
        peaks = np.asarray([[threshold - 0.5, 0.0]])
        expected = scalar_forecast(30, 5, 0.6, logs[0], peaks[0], threshold)
        actual = v2.forecast_paired_steps(30, counts, sums, logs, peaks, threshold)
        np.testing.assert_array_equal(actual, [expected])


class StateTests(unittest.TestCase):
    def test_inference_methods_are_inherited_without_replacement(self):
        self.assertIs(v2.CertificateRacingState.observe_next, RaceState.observe_next)
        self.assertIs(v2.CertificateRacingState.completion_bounds, RaceState.completion_bounds)
        self.assertIs(v2.CertificateRacingState._certify, RaceState._certify)
        self.assertIs(v2.CertificateRacingState._census_sign, RaceState._census_sign)

    def test_same_observation_sequence_has_identical_inference(self):
        rng = np.random.default_rng(723)
        matrix = rng.uniform(0.1, 0.9, size=(12, 4))
        base = RaceState(12, 4)
        revised = v2.CertificateRacingState(12, 4)
        for _ in range(matrix.size):
            available = np.flatnonzero(base.counts < 12)
            system = int(rng.choice(available))
            value = float(matrix[base.counts[system], system])
            base.observe_next(system, value)
            revised.observe_next(system, value)
            for name, expected in state_arrays(base).items():
                np.testing.assert_array_equal(getattr(revised, name), expected,
                                              err_msg=f"inference diverged in {name}")
            self.assertEqual(revised.cost, base.cost)
            self.assertEqual(revised.reasons, base.reasons)
            if not revised.complete:
                before = state_arrays(revised)
                revised.adaptive_action(np.random.default_rng(55))
                for name, expected in before.items():
                    np.testing.assert_array_equal(getattr(revised, name), expected,
                                                  err_msg=f"forecast mutated inference {name}")

    def test_cache_reuse_and_prefix_based_invalidation(self):
        history = [(system, 0.4 + 0.04 * system)
                   for _ in range(4) for system in range(3)]
        state = v2.CertificateRacingState(20, 3)
        for observation in history:
            state.observe_next(*observation)
        with mock.patch.object(v2, "forecast_paired_steps", wraps=v2.forecast_paired_steps) as spy:
            state.adaptive_action(np.random.default_rng(11))
            self.assertGreater(spy.call_count, 0)
            spy.reset_mock()
            state.adaptive_action(np.random.default_rng(11))
            self.assertEqual(spy.call_count, 0)
            # No pair's minimum prefix advances when only one system moves
            # ahead of a previously equal set of prefix lengths.
            history.append((0, 0.4))
            state.observe_next(*history[-1])
            state.adaptive_action(np.random.default_rng(11))
            self.assertEqual(spy.call_count, 0)
            # Now the (0,1) paired prefix advances and its cached forecast must
            # be invalidated, even if other pair forecasts remain reusable.
            history.append((1, 0.44))
            state.observe_next(*history[-1])
            actual = state.adaptive_action(np.random.default_rng(38))
            self.assertGreater(spy.call_count, 0)
        fresh = v2.CertificateRacingState(20, 3)
        for observation in history:
            fresh.observe_next(*observation)
        expected = fresh.adaptive_action(np.random.default_rng(38))
        self.assertEqual(tuple(actual), tuple(expected))

    def test_forecast_never_becomes_actual_evidence(self):
        state = v2.CertificateRacingState(50, 2)
        for _ in range(4):
            state.observe_next(0, 0.9)
            state.observe_next(1, 0.1)
        self.assertEqual(state.decisions.tolist(), [UNKNOWN])
        state.adaptive_action(np.random.default_rng(1))
        self.assertEqual(state.decisions.tolist(), [UNKNOWN])
        self.assertEqual(state.cost, 8)


class RunnerTests(unittest.TestCase):
    def test_prefix_order_cost_unique_queries_and_checkpoints(self):
        n_items, n_systems, seed, budget = 18, 4, 818, 21
        matrix = np.repeat(np.linspace(0.2, 0.7, n_items)[:, None], n_systems, axis=1)
        order = np.random.default_rng(np.random.SeedSequence([seed, 1])).permutation(n_items)
        counts = [0] * n_systems
        seen = set()

        def query(row, system):
            self.assertEqual(row, int(order[counts[system]]))
            self.assertNotIn((row, system), seen)
            seen.add((row, system))
            counts[system] += 1
            self.assertLessEqual(len(seen), budget)
            return float(matrix[row, system])

        result = v2.run_certificate_race(n_items, n_systems, query, seed=seed,
                                          budget=budget, warmup_rows=2,
                                          checkpoints=(0, 1, 7, 21), record_trace=True)
        self.assertEqual(result["policy"], "certificate_racing_v2")
        self.assertEqual(result["cost_cells"], budget)
        self.assertEqual(result["cost_cells"], len(seen))
        self.assertEqual(result["system_counts"], counts)
        self.assertEqual(len(result["trace"]), budget)
        for checkpoint in (0, 1, 7, 21):
            self.assertEqual(result["snapshots"][str(checkpoint)]["cost_cells"], checkpoint)

    def test_seeded_run_is_reproducible(self):
        matrix = np.random.default_rng(191).uniform(0.1, 0.9, size=(17, 4))

        def run():
            return v2.run_certificate_race(17, 4, lambda i, s: float(matrix[i, s]),
                                            seed=561, budget=39, warmup_rows=3,
                                            checkpoints=(0, 10, 20, 39), record_trace=True)

        self.assertEqual(run(), run())

    def test_zero_budget_never_queries(self):
        def forbidden_query(*args):
            raise AssertionError("zero budget must not reveal a score")

        result = v2.run_certificate_race(10, 4, forbidden_query, budget=0,
                                          checkpoints=(0,), record_trace=True)
        self.assertEqual(result["cost_cells"], 0)
        self.assertEqual(result["system_counts"], [0] * 4)
        self.assertEqual(result["decisions"], [UNKNOWN] * 6)
        self.assertEqual(result["trace"], [])
        self.assertEqual(result["stop_reason"], "budget_exhausted")

    def test_odd_final_budget_allows_only_one_cell_of_bundle(self):
        calls = []

        def query(row, system):
            calls.append((row, system))
            return 0.5

        result = v2.run_certificate_race(11, 2, query, seed=87, budget=5,
                                          warmup_rows=0, record_trace=True)
        self.assertEqual(result["cost_cells"], 5)
        self.assertEqual(len(calls), 5)
        self.assertEqual(sorted(result["system_counts"]), [2, 3])
        self.assertEqual(result["paired_prefix_counts"], [2])

    def test_exact_ties_resolve_at_census_without_starvation(self):
        result = v2.run_certificate_race(7, 3, lambda i, s: 0.4, seed=96,
                                          warmup_rows=2, record_trace=True)
        self.assertTrue(result["complete"])
        self.assertEqual(result["cost_cells"], 21)
        self.assertEqual(result["decisions"], [0, 0, 0])
        self.assertEqual(result["reasons"], ["exact_census"] * 3)

    def test_score_range_validation_remains_active(self):
        with self.assertRaises(ValueError):
            v2.run_certificate_race(3, 2, lambda i, s: 1.01, seed=7)


if __name__ == "__main__":
    unittest.main()
