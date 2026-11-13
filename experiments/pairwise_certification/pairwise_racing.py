"""Shared-prefix finite-population pairwise racing, research prototype v1.

The controller sees only (N, K, a query callback). It never receives the complete
score matrix. Statistical inference is separate from the heuristic scheduler.
See METHOD.md for the precise fixed-matrix, randomization-based guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Callable

import numpy as np


BET_FRACTIONS = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95])
UNKNOWN = 2
POLICIES = ("shared_racing", "random_pair", "common_all", "common_active")
NUMERIC_TOL = 1e-12


def log_mixture(log_products: np.ndarray) -> np.ndarray:
    """Log of the uniform mixture over the last axis, stable for large bets."""
    peak = np.max(log_products, axis=-1)
    return peak + np.log(np.exp(log_products - peak[..., None]).mean(axis=-1))


def betting_factors(x: np.ndarray, conditional_null: np.ndarray) -> np.ndarray:
    """Fixed nonnegative betting factors; neutral near an infeasible boundary.

    A neutral factor preserves the supermartingale property. Deterministic
    score-completion bounds handle infeasible nulls without dividing by zero.
    """
    x, conditional_null = np.broadcast_arrays(x, conditional_null)
    ratio = np.ones_like(x, dtype=float)
    good = conditional_null > NUMERIC_TOL
    np.divide(x, conditional_null, out=ratio, where=good)
    return 1.0 + (ratio[..., None] - 1.0) * BET_FRACTIONS


@dataclass
class RaceState:
    n_items: int
    n_systems: int
    alpha: float = 0.05
    counts: np.ndarray = field(init=False)
    sums: np.ndarray = field(init=False)
    cache: np.ndarray = field(init=False)
    pair_a: np.ndarray = field(init=False)
    pair_b: np.ndarray = field(init=False)
    pair_n: np.ndarray = field(init=False)
    sum_d: np.ndarray = field(init=False)
    sum_d2: np.ndarray = field(init=False)
    log_products: np.ndarray = field(init=False)
    log_e_peak: np.ndarray = field(init=False)
    decisions: np.ndarray = field(init=False)
    decision_costs: np.ndarray = field(init=False)
    reasons: list[str] = field(init=False)
    cost: int = 0

    def __post_init__(self):
        if not isinstance(self.n_items, int) or self.n_items < 1:
            raise ValueError("n_items must be a positive integer")
        if not isinstance(self.n_systems, int) or self.n_systems < 2:
            raise ValueError("n_systems must be at least two")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        self.counts = np.zeros(self.n_systems, dtype=int)
        self.sums = np.zeros(self.n_systems)
        # Rows of cache are positions in the ONE precommitted random order.
        self.cache = np.full((self.n_items, self.n_systems), np.nan)
        self.pair_a, self.pair_b = np.triu_indices(self.n_systems, 1)
        p = len(self.pair_a)
        self.pair_n = np.zeros(p, dtype=int)
        self.sum_d = np.zeros(p)
        self.sum_d2 = np.zeros(p)
        self.log_products = np.zeros((p, 2, len(BET_FRACTIONS)))
        self.log_e_peak = np.zeros((p, 2))
        self.decisions = np.full(p, UNKNOWN, dtype=int)
        self.decision_costs = np.full(p, -1, dtype=int)
        self.reasons = [""] * p

    @property
    def n_pairs(self):
        return len(self.pair_a)

    @property
    def log_threshold(self):
        return math.log(2 * self.n_pairs / self.alpha)

    @property
    def complete(self):
        return bool(np.all(self.decisions != UNKNOWN))

    def _certify(self, indices, signs, reason):
        for p, sign in zip(np.atleast_1d(indices), np.atleast_1d(signs)):
            p = int(p)
            if self.decisions[p] == UNKNOWN:
                self.decisions[p] = int(sign)
                self.decision_costs[p] = self.cost
                self.reasons[p] = reason

    def completion_bounds(self):
        lo = self.sums / self.n_items
        hi = (self.sums + self.n_items - self.counts) / self.n_items
        return lo[self.pair_a] - hi[self.pair_b], hi[self.pair_a] - lo[self.pair_b]

    def _census_sign(self, a, b):
        av, bv = self.cache[:, a], self.cache[:, b]
        rough = math.fsum(av.tolist()) - math.fsum(bv.tolist())
        if abs(rough) > self.n_items * NUMERIC_TOL:
            return 1 if rough > 0 else -1
        # Exact rational fallback on represented floats: near zero is NOT
        # silently relabeled an exact tie.
        exact = sum((Fraction.from_float(float(x)) for x in av), Fraction())
        exact -= sum((Fraction.from_float(float(x)) for x in bv), Fraction())
        return int(exact > 0) - int(exact < 0)

    def observe_next(self, system: int, value: float):
        """Acquire one system's next prefix cell, then reuse it for all pairs."""
        if system < 0 or system >= self.n_systems:
            raise ValueError("invalid system index")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("scores must be finite and lie in the KNOWN [0,1] range")
        pos = int(self.counts[system])
        if pos >= self.n_items:
            raise ValueError("system is already fully observed")
        self.cache[pos, system] = value
        self.counts[system] += 1
        self.sums[system] += value
        self.cost += 1
        shared_n = np.minimum(self.counts[self.pair_a], self.counts[self.pair_b])
        changed = np.flatnonzero(shared_n > self.pair_n)
        if changed.size:
            old_n = self.pair_n[changed].copy()
            if not np.all(shared_n[changed] == old_n + 1):
                raise AssertionError("a pair prefix may advance by only one per new cell")
            d = self.cache[old_n, self.pair_a[changed]] - self.cache[old_n, self.pair_b[changed]]
            # Statistical updates use ONLY this pair's own paired prefix.
            active = self.decisions[changed] == UNKNOWN
            idx = changed[active]
            if idx.size:
                nd = old_n[active]
                dd = d[active]
                x = np.column_stack(((dd + 1) / 2, (1 - dd) / 2))
                prior_sums = np.column_stack(((nd + self.sum_d[idx]) / 2,
                                              (nd - self.sum_d[idx]) / 2))
                conditional_null = (self.n_items / 2 - prior_sums) / (self.n_items - nd[:, None])
                factors = betting_factors(x, conditional_null)
                if not np.all(np.isfinite(factors)) or not np.all(factors > 0):
                    raise FloatingPointError("invalid betting factor")
                self.log_products[idx] += np.log(factors)
                self.log_e_peak[idx] = np.maximum(self.log_e_peak[idx], log_mixture(self.log_products[idx]))
            self.pair_n[changed] += 1
            self.sum_d[changed] += d
            self.sum_d2[changed] += d * d

        # Deterministic constraints do not spend statistical error probability.
        lo, hi = self.completion_bounds()
        idx = np.flatnonzero((self.decisions == UNKNOWN) & (lo > NUMERIC_TOL))
        self._certify(idx, np.ones(len(idx), dtype=int), "deterministic_completion")
        idx = np.flatnonzero((self.decisions == UNKNOWN) & (hi < -NUMERIC_TOL))
        self._certify(idx, -np.ones(len(idx), dtype=int), "deterministic_completion")
        full = (self.counts[self.pair_a] == self.n_items) & (self.counts[self.pair_b] == self.n_items)
        for p in np.flatnonzero((self.decisions == UNKNOWN) & full):
            self._certify([p], [self._census_sign(self.pair_a[p], self.pair_b[p])], "exact_census")

        crossed = self.log_e_peak > self.log_threshold + 1e-10
        for p in np.flatnonzero(self.decisions == UNKNOWN):
            if crossed[p, 0] and crossed[p, 1]:
                # Do not manufacture an order from contradictory evidence.
                continue
            if crossed[p, 0]:
                self._certify([p], [1], "mixture_betting")
            elif crossed[p, 1]:
                self._certify([p], [-1], "mixture_betting")

    def candidate_pairs(self):
        return np.flatnonzero(self.decisions == UNKNOWN)

    def action_for_pair(self, p):
        a, b = int(self.pair_a[p]), int(self.pair_b[p])
        if self.counts[a] < self.counts[b]:
            return (a,)
        if self.counts[b] < self.counts[a]:
            return (b,)
        return (a, b)

    def adaptive_action(self, rng):
        """Heuristic expected reduction in variance/gap risk per new cell.

        This is a frozen acquisition heuristic, not an optimal-information theorem.
        Changing this scheduler does not change the inference guarantee.
        """
        unresolved = self.decisions == UNKNOWN
        n = self.pair_n.astype(float)
        means = np.divide(self.sum_d, n, out=np.zeros_like(n), where=n > 0)
        var = np.ones_like(n)
        ok = n > 1
        var[ok] = np.maximum(0.0, (self.sum_d2[ok] - self.sum_d[ok] ** 2 / n[ok]) / (n[ok] - 1))
        difficulty = (var + 1e-4) / (means * means + 0.01 ** 2)
        gain = difficulty / ((n + 1) * (n + 2))
        gain[~unresolved] = 0
        ca, cb = self.counts[self.pair_a], self.counts[self.pair_b]
        benefit = np.zeros(self.n_systems)
        np.add.at(benefit, self.pair_a[ca < cb], gain[ca < cb])
        np.add.at(benefit, self.pair_b[cb < ca], gain[cb < ca])
        utility = np.where(ca < cb, benefit[self.pair_a],
                           np.where(cb < ca, benefit[self.pair_b],
                                    (benefit[self.pair_a] + benefit[self.pair_b] + gain) / 2))
        candidates = self.candidate_pairs()
        best = np.max(utility[candidates])
        ties = candidates[np.isclose(utility[candidates], best, rtol=1e-12, atol=1e-15)]
        return self.action_for_pair(int(rng.choice(ties)))

    def summary(self):
        means = np.divide(self.sums, self.counts, out=np.zeros_like(self.sums), where=self.counts > 0)
        return {"cost_cells": self.cost,
                "resolved_pairs": int(np.count_nonzero(self.decisions != UNKNOWN)),
                "n_pairs": self.n_pairs, "decisions": self.decisions.tolist(),
                "decision_costs": self.decision_costs.tolist(), "reasons": self.reasons.copy(),
                "system_counts": self.counts.tolist(), "sample_means": means.tolist(),
                "paired_prefix_counts": self.pair_n.tolist()}


def run_race(n_items: int, n_systems: int, query: Callable[[int, int], float], *,
             policy="shared_racing", seed=0, alpha=0.05, budget=None,
             warmup_rows=4, checkpoints=(), record_trace=False):
    """Run a frozen policy. Query(row,system) is the only score-access channel."""
    if policy not in POLICIES:
        raise ValueError("unknown policy")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(warmup_rows, int) or warmup_rows < 0:
        raise ValueError("warmup_rows must be a nonnegative integer")
    state = RaceState(n_items, n_systems, alpha)
    cap = n_items * n_systems if budget is None else budget
    if not isinstance(cap, int) or not 0 <= cap <= n_items * n_systems:
        raise ValueError("budget must be an integer in [0,N*K]")
    # Item permutation and policy randomization are separate streams.
    order = np.random.default_rng(np.random.SeedSequence([int(seed), 1])).permutation(n_items)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2]))
    system_order = rng.permutation(n_systems)
    tie_rank = np.argsort(system_order)
    wanted = sorted(set(int(x) for x in checkpoints if 0 <= x <= cap))
    snapshots = {}
    trace = []
    resolution_costs = {str(f): None for f in (0.5, 0.9, 0.95, 1.0)}
    resolved_area = 0.0

    def acquire(s):
        nonlocal resolved_area
        resolved_area += np.count_nonzero(state.decisions != UNKNOWN) / state.n_pairs
        row = int(order[state.counts[s]])
        value = float(query(row, int(s)))
        state.observe_next(int(s), value)
        if record_trace:
            trace.append({"cost": state.cost, "row": row, "system": int(s), "value": value})
        fraction = np.count_nonzero(state.decisions != UNKNOWN) / state.n_pairs
        for f in resolution_costs:
            if resolution_costs[f] is None and fraction >= float(f):
                resolution_costs[f] = state.cost
        if state.cost in wanted:
            snapshots[str(state.cost)] = state.summary()

    if 0 in wanted:
        snapshots["0"] = state.summary()
    for _ in range(min(warmup_rows, n_items)):
        for s in system_order:
            if state.cost >= cap or state.complete:
                break
            acquire(int(s))
        if state.cost >= cap or state.complete:
            break

    while state.cost < cap and not state.complete:
        if policy == "shared_racing":
            action = state.adaptive_action(rng)
        elif policy == "random_pair":
            p = int(rng.choice(state.candidate_pairs()))
            action = state.action_for_pair(p)
        else:
            if policy == "common_all":
                active = np.arange(n_systems)
            else:
                unresolved = state.candidate_pairs()
                active = np.unique(np.r_[state.pair_a[unresolved], state.pair_b[unresolved]])
            available = active[state.counts[active] < n_items]
            if not available.size:
                raise AssertionError("no acquisition available while pairs remain unresolved")
            key = state.counts[available] * n_systems + tie_rank[available]
            action = (int(available[np.argmin(key)]),)
        # Randomize order inside a two-cell bundle, independent of hidden scores.
        for s in rng.permutation(action):
            if state.cost >= cap or state.complete:
                break
            acquire(int(s))

    result = state.summary()
    # When early stopping occurs, the same certificates persist at later budgets;
    # no further labels are charged and point estimates remain unchanged.
    for checkpoint in wanted:
        if str(checkpoint) not in snapshots:
            snapshots[str(checkpoint)] = state.summary()
    final_fraction = result["resolved_pairs"] / state.n_pairs
    result.update({"policy": policy, "seed": int(seed), "alpha": alpha,
                   "complete": state.complete, "resolution_costs": resolution_costs,
                   "certified_fraction_auc": (resolved_area + (cap - state.cost) * final_fraction) / cap if cap else 0.0,
                   "snapshots": snapshots, "trace": trace,
                   "stop_reason": "all_pairs_decided" if state.complete else "budget_exhausted"})
    return result
