"""One development revision: acquisition by plug-in distance to certification.

Actual certificates are issued ONLY by the unmodified v1 RaceState. Forecasts
are fictional constant-mean continuations; they never become observations.
See METHOD_V2.md and protocol_v2.json for limitations and frozen choices.
"""
from __future__ import annotations

import numpy as np

from pairwise_racing import RaceState, UNKNOWN, NUMERIC_TOL, betting_factors, log_mixture


POLICY = "certificate_racing_v2"


def forecast_paired_steps(n_items: int, pair_n: np.ndarray, sum_d: np.ndarray,
                         log_products: np.ndarray, log_e_peak: np.ndarray,
                         log_threshold: float) -> np.ndarray:
    """Forecast remaining paired steps, NOT expected cost or a confidence bound.

    Copies of current pair capitals are advanced with constant future differences
    equal to the current paired mean. Ignore one-sided cached observations in
    this acquisition heuristic. Return first predicted XOR betting crossing,
    paired-only deterministic completion, or census, always in [1, N-n].
    """
    ns = np.asarray(pair_n, dtype=int)
    sums = np.asarray(sum_d, dtype=float)
    if ns.ndim != 1 or sums.shape != ns.shape:
        raise ValueError("pair counts and sums must be same-length vectors")
    if np.any(ns < 0) or np.any(ns >= n_items):
        raise ValueError("forecast requires an unresolved, non-census prefix")
    if len(ns) == 0:
        return np.zeros(0, dtype=int)
    products = np.asarray(log_products, dtype=float)
    peaks = np.asarray(log_e_peak, dtype=float)
    if products.shape != (len(ns), 2, 7) or peaks.shape != (len(ns), 2):
        raise ValueError("unexpected directional betting-state shape")
    if not all(np.isfinite(x).all() for x in (sums, products, peaks)) or not np.isfinite(log_threshold):
        raise ValueError("forecast state must be finite")

    remaining = n_items - ns
    steps = np.arange(1, int(remaining.max()) + 1)
    valid = steps[None, :] <= remaining[:, None]
    dbar = np.divide(sums, ns, out=np.zeros_like(sums), where=ns > 0)
    # This clips only floating-roundoff in the fictional forecast, not data.
    dbar = np.clip(dbar, -1.0, 1.0)
    xhat = np.column_stack(((1 + dbar) / 2, (1 - dbar) / 2))
    past_x = np.column_stack(((ns + sums) / 2, (ns - sums) / 2))
    denominator = n_items - ns[:, None] - steps[None, :] + 1
    denominator = np.where(valid, denominator, 1)
    q = (n_items / 2 - past_x[:, None, :]
         - (steps[None, :, None] - 1) * xhat[:, None, :]) / denominator[:, :, None]
    factors = betting_factors(xhat[:, None, :], q)
    increments = np.where(valid[:, :, None, None], np.log(factors), 0.0)
    projected_products = products[:, None, :, :] + np.cumsum(increments, axis=1)
    projected_peaks = np.maximum(peaks[:, None, :],
                                 np.maximum.accumulate(log_mixture(projected_products), axis=1))
    crossed = projected_peaks > log_threshold + 1e-10
    betting_hit = np.logical_xor(crossed[:, :, 0], crossed[:, :, 1])

    projected_sum = sums[:, None] + steps[None, :] * dbar[:, None]
    unseen = remaining[:, None] - steps[None, :]
    lower = (projected_sum - unseen) / n_items
    upper = (projected_sum + unseen) / n_items
    completion_hit = (lower > NUMERIC_TOL) | (upper < -NUMERIC_TOL)
    census_hit = steps[None, :] == remaining[:, None]
    hit = (betting_hit | completion_hit | census_hit) & valid
    if not np.all(np.any(hit, axis=1)):
        raise AssertionError("the census fallback must be reachable")
    return np.argmax(hit, axis=1).astype(int) + 1


class CertificateRacingState(RaceState):
    """Inherits all observation, inference, and certification code unchanged."""

    def __post_init__(self):
        super().__post_init__()
        self._forecast_n = np.full(self.n_pairs, -1, dtype=int)
        self._forecast_h = np.full(self.n_pairs, self.n_items, dtype=int)

    def adaptive_action(self, rng):
        unresolved = self.decisions == UNKNOWN
        dirty = np.flatnonzero(unresolved & (self._forecast_n != self.pair_n))
        if dirty.size:
            self._forecast_h[dirty] = forecast_paired_steps(
                self.n_items, self.pair_n[dirty], self.sum_d[dirty],
                self.log_products[dirty], self.log_e_peak[dirty], self.log_threshold)
            self._forecast_n[dirty] = self.pair_n[dirty]
        gain = np.where(unresolved, 1.0 / self._forecast_h, 0.0)
        ca, cb = self.counts[self.pair_a], self.counts[self.pair_b]
        benefit = np.zeros(self.n_systems)
        np.add.at(benefit, self.pair_a[ca < cb], gain[ca < cb])
        np.add.at(benefit, self.pair_b[cb < ca], gain[cb < ca])
        utility = np.where(ca < cb, benefit[self.pair_a],
                           np.where(cb < ca, benefit[self.pair_b],
                                    (benefit[self.pair_a] + benefit[self.pair_b] + gain) / 2))
        candidates = self.candidate_pairs()
        if not candidates.size:
            raise ValueError("no unresolved pair remains")
        best = np.max(utility[candidates])
        ties = candidates[np.isclose(utility[candidates], best, rtol=1e-12, atol=1e-15)]
        return self.action_for_pair(int(rng.choice(ties)))


def run_certificate_race(n_items: int, n_systems: int, query, *, seed=0,
                         alpha=0.05, budget=None, warmup_rows=4,
                         checkpoints=(), record_trace=False):
    """The same v1 driver conventions with one changed acquisition policy.

    Kept separate so the frozen v1 source hashes and results remain untouched.
    The only score access is query(row,system), once per acquired cell.
    """
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(warmup_rows, int) or warmup_rows < 0:
        raise ValueError("warmup_rows must be a nonnegative integer")
    state = CertificateRacingState(n_items, n_systems, alpha)
    cap = n_items * n_systems if budget is None else budget
    if not isinstance(cap, int) or not 0 <= cap <= n_items * n_systems:
        raise ValueError("budget must be an integer in [0,N*K]")
    order = np.random.default_rng(np.random.SeedSequence([int(seed), 1])).permutation(n_items)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2]))
    system_order = rng.permutation(n_systems)
    wanted = sorted(set(int(x) for x in checkpoints if 0 <= x <= cap))
    snapshots, trace = {}, []
    resolution_costs = {str(f): None for f in (0.5, 0.9, 0.95, 1.0)}
    resolved_area = 0.0

    def acquire(system):
        nonlocal resolved_area
        resolved_area += np.count_nonzero(state.decisions != UNKNOWN) / state.n_pairs
        row = int(order[state.counts[system]])
        value = float(query(row, int(system)))
        state.observe_next(int(system), value)
        if record_trace:
            trace.append({"cost": state.cost, "row": row, "system": int(system), "value": value})
        fraction = np.count_nonzero(state.decisions != UNKNOWN) / state.n_pairs
        for f in resolution_costs:
            if resolution_costs[f] is None and fraction >= float(f):
                resolution_costs[f] = state.cost
        if state.cost in wanted:
            snapshots[str(state.cost)] = state.summary()

    if 0 in wanted:
        snapshots["0"] = state.summary()
    for _ in range(min(warmup_rows, n_items)):
        for system in system_order:
            if state.cost >= cap or state.complete:
                break
            acquire(int(system))
        if state.cost >= cap or state.complete:
            break

    while state.cost < cap and not state.complete:
        action = state.adaptive_action(rng)
        for system in rng.permutation(action):
            if state.cost >= cap or state.complete:
                break
            acquire(int(system))

    result = state.summary()
    for checkpoint in wanted:
        if str(checkpoint) not in snapshots:
            snapshots[str(checkpoint)] = state.summary()
    final_fraction = result["resolved_pairs"] / state.n_pairs
    result.update({"policy": POLICY, "seed": int(seed), "alpha": alpha,
                   "complete": state.complete, "resolution_costs": resolution_costs,
                   "certified_fraction_auc": (resolved_area + (cap - state.cost) * final_fraction) / cap if cap else 0.0,
                   "snapshots": snapshots, "trace": trace,
                   "stop_reason": "all_pairs_decided" if state.complete else "budget_exhausted"})
    return result
