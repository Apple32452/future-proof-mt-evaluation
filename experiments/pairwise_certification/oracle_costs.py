"""Hindsight final-prefix cost optimization, NEVER a deployable acquisition rule.

Full scores may enter this diagnostic only on exposed development data. The
model includes both paired-prefix betting and unequal-prefix completion.
Its arbitrary per-system acquisition class relaxes v1/v2 pair-bundle actions.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from pairwise_racing import BET_FRACTIONS, NUMERIC_TOL, RaceState, UNKNOWN, betting_factors, log_mixture


def build_pair_geometry(y_ordered, alpha=0.05):
    y = np.asarray(y_ordered, dtype=float)
    if y.ndim != 2 or y.shape[0] < 1 or y.shape[1] < 2:
        raise ValueError("a nonempty item-by-system matrix is required")
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("scores must be finite and bounded by [0,1]")
    if not 0 < alpha < 1:
        raise ValueError("invalid alpha")
    n_items, k = y.shape
    a, b = np.triu_indices(k, 1)
    pairs = len(a)
    threshold = math.log(2 * pairs / alpha)
    sums = np.zeros(pairs)
    products = np.zeros((pairs, 2, len(BET_FRACTIONS)))
    peaks = np.zeros((pairs, 2))
    tau = np.full(pairs, n_items + 1, dtype=int)
    # Independently advance every pair's statistical path, without stopping it
    # at a deterministic completion produced by a particular global schedule.
    for n in range(n_items):
        d = y[n, a] - y[n, b]
        x = np.column_stack(((d + 1) / 2, (1 - d) / 2))
        previous_x = np.column_stack(((n + sums) / 2, (n - sums) / 2))
        q = (n_items / 2 - previous_x) / (n_items - n)
        products += np.log(betting_factors(x, q))
        peaks = np.maximum(peaks, log_mixture(products))
        crossed = peaks > threshold + 1e-10
        first = (tau == n_items + 1) & np.logical_xor(crossed[:, 0], crossed[:, 1])
        tau[first] = n + 1
        sums += d

    prefix = np.vstack((np.zeros((1, k)), np.cumsum(y, axis=0)))
    counts = np.arange(n_items + 1)
    lower = prefix / n_items
    upper = (prefix + n_items - counts[:, None]) / n_items
    matrices = np.zeros((pairs, n_items + 1, n_items + 1), dtype=bool)
    frontiers = []
    paired_counts = np.minimum(counts[:, None], counts[None, :])
    for p, (left, right) in enumerate(zip(a, b)):
        lo = lower[:, left, None] - upper[None, :, right]
        hi = upper[:, left, None] - lower[None, :, right]
        decided = (paired_counts >= tau[p]) | (lo > NUMERIC_TOL) | (hi < -NUMERIC_TOL)
        decided[-1, -1] = True  # Exact census includes exact ties.
        if np.any(decided[:-1, :] & ~decided[1:, :]) or np.any(decided[:, :-1] & ~decided[:, 1:]):
            raise AssertionError("numerical decision geometry is not monotone")
        matrices[p] = decided
        # Maximal elements of the downward-closed undecided count set.
        max_v = np.array([np.flatnonzero(~row)[-1] if np.any(~row) else -1 for row in decided])
        maximal = [(u, int(v)) for u, v in enumerate(max_v)
                   if v >= 0 and (u == n_items or v > max_v[u + 1])]
        frontiers.append(maximal)
    return {"n_items": n_items, "n_systems": k, "alpha": alpha,
            "pair_a": a, "pair_b": b, "prefix_sums": prefix, "tau": tau,
            "matrices": matrices, "forbidden_frontiers": frontiers}


def resolution_count(geometry, counts):
    counts = np.asarray(counts, dtype=int)
    p = np.arange(len(geometry["pair_a"]))
    return int(geometry["matrices"][p, counts[geometry["pair_a"]], counts[geometry["pair_b"]]].sum())


def solve_oracle(geometry, target_pairs, warmup_min=0, time_limit=30):
    n, k = geometry["n_items"], geometry["n_systems"]
    a, b = geometry["pair_a"], geometry["pair_b"]
    pairs = len(a)
    if not 1 <= target_pairs <= pairs or not 0 <= warmup_min <= n:
        raise ValueError("invalid target or warmup")
    variables = k * n + pairs
    objective = np.r_[np.ones(k * n), np.zeros(pairs)]
    lb, ub = np.zeros(variables), np.ones(variables)
    for s in range(k):
        lb[s * n:s * n + warmup_min] = 1
    rows, columns, values, limits = [], [], [], []

    def constraint(terms, upper):
        index = len(limits)
        for col, value in terms:
            rows.append(index)
            columns.append(col)
            values.append(value)
        limits.append(upper)

    # x[s,r] = 1 exactly when the prefix count for system s is at least r.
    for s in range(k):
        for r in range(1, n):
            constraint([(s * n + r, 1), (s * n + r - 1, -1)], 0)
    for p, forbidden in enumerate(geometry["forbidden_frontiers"]):
        for u, v in forbidden:
            terms = [(k * n + p, 1)]
            if u < n:
                terms.append((int(a[p]) * n + u, -1))
            if v < n:
                terms.append((int(b[p]) * n + v, -1))
            constraint(terms, 0)
    constraint([(k * n + p, -1) for p in range(pairs)], -target_pairs)
    matrix = coo_matrix((values, (rows, columns)), shape=(len(limits), variables)).tocsc()
    result = milp(objective, integrality=np.ones(variables), bounds=Bounds(lb, ub),
                  constraints=LinearConstraint(matrix, -np.inf, np.array(limits)),
                  options={"time_limit": float(time_limit), "mip_rel_gap": 0.0})
    raw_lower = getattr(result, "mip_dual_bound", None)
    valid_lower = float(raw_lower) if raw_lower is not None and np.isfinite(raw_lower) else 0.0
    # Conservatively round DOWN. This is a numerical solver certificate, not
    # an independently checked exact-rational branch-and-bound proof.
    lower = max(k * warmup_min, int(math.floor(max(0.0, valid_lower) - 1e-7)))
    answer = {"solver_status": int(result.status), "solver_message": str(result.message),
              "incumbent_cost": None, "lower_bound": lower, "raw_solver_dual_bound": valid_lower,
              "counts": None, "optimal": bool(result.status == 0),
              "gap": None, "variables": variables, "constraints": len(limits),
              "mip_node_count": int(getattr(result, "mip_node_count", 0) or 0),
              "target_pairs": int(target_pairs), "warmup_min": warmup_min,
              "time_limit_seconds": float(time_limit)}
    gap = getattr(result, "mip_gap", None)
    if gap is not None and np.isfinite(gap):
        answer["gap"] = float(gap)
    if result.x is not None and np.isfinite(result.x).all():
        rounded = np.rint(result.x)
        if np.max(np.abs(result.x - rounded)) > 1e-5:
            raise AssertionError("solver incumbent is not integral")
        x = rounded[:k * n].reshape(k, n).astype(int)
        if np.any(x[:, 1:] > x[:, :-1]):
            raise AssertionError("non-prefix incumbent")
        counts = x.sum(axis=1)
        if np.any(counts < warmup_min) or resolution_count(geometry, counts) < target_pairs:
            raise AssertionError("solver incumbent fails independent geometry check")
        cost = int(counts.sum())
        if abs(cost - float(result.fun)) > 1e-5 or lower > cost:
            raise AssertionError("inconsistent optimization cost/bound")
        answer.update({"incumbent_cost": cost, "counts": counts.tolist(),
                       "resolved_pairs": resolution_count(geometry, counts)})
    return answer


def replay_allocation(y_ordered, counts, alpha=0.05, target_pairs=None):
    """Check an oracle count vector using actual v1 inference, charging all cells.

    A common-row schedule with per-system terminal counts belongs to the relaxed
    oracle class, not necessarily the pair-bundle action class of our methods.
    """
    y = np.asarray(y_ordered)
    counts = np.asarray(counts, dtype=int)
    n, k = y.shape
    if counts.shape != (k,) or np.any(counts < 0) or np.any(counts > n):
        raise ValueError("invalid final counts")
    state = RaceState(n, k, alpha)
    target = state.n_pairs if target_pairs is None else target_pairs
    first_cost = None
    first_counts = None
    for row in range(int(counts.max(initial=0))):
        for system in range(k):
            if row < counts[system]:
                state.observe_next(system, float(y[row, system]))
                if first_cost is None and np.count_nonzero(state.decisions != UNKNOWN) >= target:
                    first_cost, first_counts = state.cost, state.counts.tolist()
    return {"charged_cells": state.cost, "first_target_cost": first_cost,
            "first_target_counts": first_counts, "resolved_pairs": int(np.count_nonzero(state.decisions != UNKNOWN)),
            "decisions": state.decisions.tolist(), "counts": state.counts.tolist()}
