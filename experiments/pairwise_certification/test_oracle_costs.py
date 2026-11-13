"""Independent tiny-case tests for the hindsight count-allocation oracle.

These are finite correctness fixtures, not new MT performance experiments.
Brute-force reference minima enumerate every final count vector. Independent
scalar betting arithmetic and actual RaceState replays check the geometry.
"""
from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from oracle_costs import build_pair_geometry, solve_oracle
from pairwise_racing import RaceState, UNKNOWN


BET_C = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95)
TOL = 1e-12


def scalar_first_betting_crossing(differences, n_pairs, alpha):
    """First XOR crossing, persisted even if both peaks cross later."""
    n_items = len(differences)
    capital = [[0.0] * len(BET_C) for _ in range(2)]
    peaks = [0.0, 0.0]
    difference_sum = 0.0
    threshold = math.log(2 * n_pairs / alpha) + 1e-10
    first = n_items + 1
    for n, difference in enumerate(differences):
        xs = ((1 + float(difference)) / 2, (1 - float(difference)) / 2)
        observed_x = ((n + difference_sum) / 2, (n - difference_sum) / 2)
        for direction in range(2):
            q = (n_items / 2 - observed_x[direction]) / (n_items - n)
            ratio = xs[direction] / q if q > TOL else 1.0
            for j, fraction in enumerate(BET_C):
                capital[direction][j] += math.log(1 + fraction * (ratio - 1))
            maximum = max(capital[direction])
            mixture = maximum + math.log(math.fsum(
                math.exp(value - maximum) for value in capital[direction]) / len(BET_C))
            peaks[direction] = max(peaks[direction], mixture)
        if first == n_items + 1 and ((peaks[0] > threshold) != (peaks[1] > threshold)):
            first = n + 1
        difference_sum += float(difference)
    return first


def independent_geometry(y, alpha=0.05):
    """Direct pair-by-count definition, without calling oracle helpers."""
    n_items, n_systems = y.shape
    pairs = list(itertools.combinations(range(n_systems), 2))
    tau = [scalar_first_betting_crossing(y[:, a] - y[:, b], len(pairs), alpha)
           for a, b in pairs]
    result = np.zeros((len(pairs), n_items + 1, n_items + 1), dtype=bool)
    for p, (a, b) in enumerate(pairs):
        for ca, cb in itertools.product(range(n_items + 1), repeat=2):
            sa = math.fsum(float(x) for x in y[:ca, a])
            sb = math.fsum(float(x) for x in y[:cb, b])
            lower = sa / n_items - (sb + n_items - cb) / n_items
            upper = (sa + n_items - ca) / n_items - sb / n_items
            result[p, ca, cb] = (min(ca, cb) >= tau[p]
                                or lower > TOL or upper < -TOL
                                or (ca == n_items and cb == n_items))
    return result, np.asarray(tau)


def brute_minimum(y, target_pairs, warmup_min=0, alpha=0.05):
    matrices, _ = independent_geometry(y, alpha)
    pairs = list(itertools.combinations(range(y.shape[1]), 2))
    feasible = []
    for counts in itertools.product(range(warmup_min, y.shape[0] + 1), repeat=y.shape[1]):
        decided = sum(bool(matrices[p, counts[a], counts[b]]) for p, (a, b) in enumerate(pairs))
        if decided >= target_pairs:
            feasible.append((sum(counts), counts))
    return min(feasible)


def fixture_matrices():
    return [
        np.array([[1.0, 0.0]]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.array([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]]),
        np.array([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0], [0.5, 0.5, 0.5]]),
        np.array([[1.0, 0.25, 0.0], [0.75, 0.5, 0.25],
                  [1.0, 0.25, 0.5], [0.5, 0.75, 0.0]]),
        np.full((4, 3), 0.5),
    ]


class GeometryTests(unittest.TestCase):
    def test_prefix_sums_and_independent_full_geometry(self):
        for y in fixture_matrices():
            with self.subTest(shape=y.shape, first_row=y[0].tolist()):
                geometry = build_pair_geometry(y)
                expected, tau = independent_geometry(y)
                np.testing.assert_array_equal(geometry["matrices"], expected)
                np.testing.assert_array_equal(geometry["tau"], tau)
                np.testing.assert_allclose(geometry["prefix_sums"],
                                           np.vstack((np.zeros(y.shape[1]), np.cumsum(y, axis=0))),
                                           rtol=0, atol=1e-14)

    def test_betting_crossing_matches_independent_scalar_reference(self):
        # Larger than brute-force fixtures to exercise a non-census crossing.
        y = np.tile([1.0, 0.5, 0.0], (18, 1))
        geometry = build_pair_geometry(y)
        expected, tau = independent_geometry(y)
        np.testing.assert_array_equal(geometry["tau"], tau)
        self.assertTrue(np.any(tau < len(y)))
        np.testing.assert_array_equal(geometry["matrices"], expected)

    def test_decision_regions_are_monotone_and_include_census(self):
        for y in fixture_matrices():
            matrices = np.asarray(build_pair_geometry(y)["matrices"])
            self.assertTrue(np.all(~matrices[:, :-1, :] | matrices[:, 1:, :]))
            self.assertTrue(np.all(~matrices[:, :, :-1] | matrices[:, :, 1:]))
            self.assertTrue(np.all(matrices[:, -1, -1]))
            self.assertFalse(np.any(matrices[:, 0, 0]))

    def test_maximal_forbidden_points_and_clauses_are_exact(self):
        for y in fixture_matrices():
            geometry = build_pair_geometry(y)
            n = len(y)
            for p, matrix in enumerate(geometry["matrices"]):
                frontier = [tuple(map(int, point)) for point in geometry["forbidden_frontiers"][p]]
                self.assertEqual(len(set(frontier)), len(frontier))
                for u, v in frontier:
                    self.assertFalse(matrix[u, v])
                    if u < n:
                        self.assertTrue(matrix[u + 1, v])
                    if v < n:
                        self.assertTrue(matrix[u, v + 1])
                for ca, cb in itertools.product(range(n + 1), repeat=2):
                    # x_(a,u+1) OR x_(b,v+1), including u=N/v=N boundaries.
                    predicate = all(ca > u or cb > v for u, v in frontier)
                    self.assertEqual(predicate, bool(matrix[ca, cb]),
                                     (p, ca, cb, frontier))

    def test_unequal_prefix_completion_is_not_discarded(self):
        y = np.tile([1.0, 0.0], (4, 1))
        geometry = build_pair_geometry(y)
        self.assertTrue(geometry["matrices"][0, 4, 1])
        self.assertTrue(geometry["matrices"][0, 1, 4])
        self.assertLess(1, geometry["tau"][0])
        # A paired-prefix-only bound would still be [-.5, 1] at min count 1.
        self.assertLessEqual((1.0 - 3) / 4, 0)
        state = RaceState(4, 2)
        for value in y[:, 0]:
            state.observe_next(0, float(value))
        state.observe_next(1, 0.0)
        self.assertEqual(state.decisions.tolist(), [1])
        self.assertEqual(state.reasons, ["deterministic_completion"])

    def test_exact_tie_geometry_requires_two_system_census(self):
        y = np.full((4, 2), 0.5)
        geometry = build_pair_geometry(y)
        expected = np.zeros((5, 5), dtype=bool)
        expected[4, 4] = True
        np.testing.assert_array_equal(geometry["matrices"][0], expected)
        self.assertEqual(set(map(tuple, geometry["forbidden_frontiers"][0])), {(3, 4), (4, 3)})

    def test_actual_inference_replay_agrees_for_every_tiny_count_vector(self):
        y = fixture_matrices()[4]
        geometry = build_pair_geometry(y)
        pairs = list(itertools.combinations(range(y.shape[1]), 2))
        for counts in itertools.product(range(len(y) + 1), repeat=y.shape[1]):
            expected = [bool(geometry["matrices"][p, counts[a], counts[b]])
                        for p, (a, b) in enumerate(pairs)]
            for system_order in ((0, 1, 2), (2, 0, 1)):
                state = RaceState(*y.shape)
                # Fully advance one chosen system at a time, an intentionally
                # unbalanced path that exercises one-sided completion bounds.
                for system in system_order:
                    for row in range(counts[system]):
                        state.observe_next(system, float(y[row, system]))
                np.testing.assert_array_equal(state.decisions != UNKNOWN, expected,
                                              err_msg=f"counts={counts}; order={system_order}")


class OptimizationTests(unittest.TestCase):
    def test_tiny_milp_optimum_matches_exhaustive_count_search(self):
        for y in fixture_matrices():
            geometry = build_pair_geometry(y)
            p = y.shape[1] * (y.shape[1] - 1) // 2
            for warmup in sorted({0, min(1, len(y))}):
                for target in sorted({1, max(1, p - 1), p, math.ceil(0.9 * p)}):
                    with self.subTest(shape=y.shape, first_row=y[0].tolist(),
                                      target=target, warmup=warmup):
                        expected_cost, _ = brute_minimum(y, target, warmup)
                        result = solve_oracle(geometry, target, warmup_min=warmup, time_limit=10)
                        self.assertTrue(result["optimal"], result)
                        self.assertAlmostEqual(result["incumbent_cost"], expected_cost, places=6)
                        self.assertLessEqual(result["lower_bound"], expected_cost + 1e-6)
                        # The implementation intentionally floors the numerical
                        # dual bound minus a small guard: an integer optimum
                        # may therefore yield a conservative bound one below it.
                        self.assertGreaterEqual(result["lower_bound"], expected_cost - 1 - 1e-6)
                        counts = list(map(int, result["counts"]))
                        self.assertEqual(sum(counts), expected_cost)
                        self.assertTrue(all(warmup <= c <= len(y) for c in counts))
                        decided = sum(bool(geometry["matrices"][q, counts[a], counts[b]])
                                      for q, (a, b) in enumerate(zip(geometry["pair_a"], geometry["pair_b"])))
                        self.assertGreaterEqual(decided, target)

    def test_tied_pair_known_exact_cost_and_positive_warmup(self):
        y = np.full((4, 3), 0.5)
        geometry = build_pair_geometry(y)
        # One pair requires two full systems. A compulsory warmup on the third
        # is a real cost and must not be dropped from the objective.
        for warmup, expected in ((0, 8), (1, 9), (2, 10), (4, 12)):
            result = solve_oracle(geometry, 1, warmup_min=warmup, time_limit=10)
            self.assertTrue(result["optimal"])
            self.assertEqual(result["incumbent_cost"], expected)


if __name__ == "__main__":
    unittest.main()
