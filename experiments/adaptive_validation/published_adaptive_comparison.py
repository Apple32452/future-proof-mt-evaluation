"""Published adaptive acquisition rules adapted to common MT pair certificates.

Zouhar et al., Findings of EMNLP 2026:
https://aclanthology.org/2026.findings-emnlp.360/
https://github.com/zouharvi/evaluation-bandit/blob/main/evaluation_bandit/algorithms.py

NOT an exact replication of their original experiment. The policies here share
our finite-population, simultaneous pairwise certification rule and cell costs.
The path shim below reuses the already-audited sibling pairwise engine.
"""
from __future__ import annotations
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pairwise_certification"))
from pairwise_racing import RaceState, UNKNOWN, run_race
from certificate_racing_v2 import run_certificate_race

POLICIES = ("common_active", "random_pair", "certificate_racing_v2",
            "published_inv_rank2", "published_confusion")


def rank_weights(state: RaceState) -> np.ndarray:
    """Inverse squared observed-mean rank (higher scores have rank one)."""
    means = np.divide(state.sums, state.counts,
                      out=np.zeros_like(state.sums), where=state.counts > 0)
    order = np.argsort(-means, kind="stable")
    ranks = np.empty(state.n_systems, dtype=np.int64)
    ranks[order] = np.arange(1, state.n_systems + 1)
    return 1.0 / (ranks.astype(float) ** 2)


def published_action(state: RaceState, rng: np.random.Generator, policy: str) -> int:
    """Select one system using observed prefixes only; never query unseen scores.

    Confusion minimization forecasts 5 more points, but actually purchases
    batches of 3. Small observed variances receive a fixed 1e-6 floor.
    """
    viable = np.flatnonzero(state.counts < state.n_items)
    if not len(viable):
        raise RuntimeError("no valid system to acquire")
    weights = rank_weights(state)
    if policy == "published_inv_rank2":
        probs = weights[viable] / np.sum(weights[viable])
        return int(rng.choice(viable, p=probs))
    if policy != "published_confusion":
        raise ValueError("unknown published policy")
    counts = state.counts.astype(float)
    means = state.sums / counts
    var = np.empty(state.n_systems)
    for s in range(state.n_systems):
        prefix = state.cache[:state.counts[s], s]
        var[s] = max(float(np.var(prefix, ddof=1)), 1e-6)
    delta = np.zeros(state.n_systems)
    for a in viable:
        for b in range(state.n_systems):
            if a == b:
                continue
            gap = abs(means[a] - means[b])
            old = math.sqrt(var[a] / counts[a] + var[b] / counts[b])
            new = math.sqrt(var[a] / (counts[a] + 5) + var[b] / counts[b])
            delta[a] += weights[b] * (ndtr(gap / old) - ndtr(gap / new))
        delta[a] *= weights[a]
    minimum = min(delta[viable])
    tied = viable[np.isclose(delta[viable], minimum, atol=1e-14)]
    return int(rng.choice(tied))


def run_published_policy(y: np.ndarray, policy: str, *, seed: int,
                         alpha=.05, warmup_rows=15):
    """Same randomized fixed item order and same RaceState inference as v2."""
    if policy not in ("published_inv_rank2", "published_confusion"):
        raise ValueError("not a supported published acquisition rule")
    n, k = y.shape
    if warmup_rows < 2 or warmup_rows > n:
        raise ValueError("published baseline needs >=2 warmup rows per system")
    order = np.random.default_rng(np.random.SeedSequence([seed, 1])).permutation(n)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    state = RaceState(n, k, alpha)
    asked = set()
    snapshots = {}
    costs = {"0.5": None, "0.9": None, "0.95": None, "1.0": None}
    check_budget = (n * k) // 2
    system_order = rng.permutation(k)

    def acquire(s: int):
        row = int(order[state.counts[s]])
        if (row, s) in asked:
            raise AssertionError("duplicate score cell charged")
        asked.add((row, s))
        state.observe_next(s, float(y[row, s]))
        frac = np.mean(state.decisions != UNKNOWN)
        for cutoff in costs:
            if costs[cutoff] is None and frac >= float(cutoff):
                costs[cutoff] = state.cost
        if state.cost == check_budget:
            snapshots["half"] = state.summary()

    for _ in range(warmup_rows):
        for s in system_order:
            if state.complete:
                break
            acquire(int(s))
    while not state.complete and state.cost < n * k:
        s = published_action(state, rng, policy)
        batch = min(3, n - state.counts[s]) if policy == "published_confusion" else 1
        for _ in range(batch):
            if state.complete:
                break
            acquire(s)
    if state.cost != len(asked):
        raise AssertionError("score cache/cost mismatch")
    return {
        "policy": policy, "seed": seed, "alpha": alpha,
        "cost_cells": state.cost, "complete": state.complete,
        "decisions": state.decisions.tolist(),
        "decision_costs": state.decision_costs.tolist(),
        "resolution_costs": costs,
        "half_snapshot": snapshots.get("half", state.summary()),
        "system_counts": state.counts.tolist(),
        "queried_unique": len(asked),
    }


def run_one(y: np.ndarray, policy: str, *, seed: int,
            alpha=.05, warmup_rows=15):
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or y.shape[0] < warmup_rows or y.shape[1] < 2:
        raise ValueError("expected rectangular item-by-system matrix")
    if not np.all(np.isfinite(y)) or np.any((y < 0) | (y > 1)):
        raise ValueError("scores must have PREDECLARED bounds [0,1]")
    n, k = y.shape
    if policy in ("published_inv_rank2", "published_confusion"):
        rec = run_published_policy(y, policy, seed=seed,
                                   alpha=alpha, warmup_rows=warmup_rows)
    else:
        def query(r, s):
            return float(y[r, s])
        if policy == "certificate_racing_v2":
            rec = run_certificate_race(n, k, query, seed=seed, alpha=alpha,
                                       warmup_rows=warmup_rows,
                                       checkpoints=(n*k//2,))
        else:
            rec = run_race(n, k, query, policy=policy, seed=seed, alpha=alpha,
                           warmup_rows=warmup_rows, checkpoints=(n*k//2,))
    truth_diff = y.mean(axis=0)
    pa, pb = np.triu_indices(k, 1)
    truth = np.sign(truth_diff[pa] - truth_diff[pb]).astype(int)
    dec = np.asarray(rec["decisions"], dtype=int)
    wrong = (dec != UNKNOWN) & (dec != truth)
    cost90 = rec["resolution_costs"]["0.9"]
    costall = rec["resolution_costs"]["1.0"]
    half = rec.get("half_snapshot",
                   rec.get("snapshots", {}).get(str(n*k//2), {}))
    approx = np.asarray(half.get("sample_means", []), dtype=float)
    if approx.shape != (k,):
        raise AssertionError("missing half-budget sample means")
    # Exploratory posthoc ranking; TRUE means enter evaluation ONLY.
    ranks = np.empty(k, dtype=int)
    ranks[np.argsort(-truth_diff, kind="stable")] = np.arange(1, k + 1)
    pairweights = 1.0 / (ranks[pa]**2 * ranks[pb]**2)
    approxsign = np.sign(approx[pa] - approx[pb])
    wtau = float(np.sum(pairweights * truth * approxsign) / np.sum(pairweights))
    true_winner = int(np.argmax(truth_diff))
    top1 = int(np.argmax(approx) == true_winner)
    return {
        "policy": policy, "seed": seed, "items": n, "systems": k,
        "posthoc_top_weighted_tau_half": wtau,
        "posthoc_top1_half": top1,
        "cost_90": int(cost90) if cost90 is not None else n*k,
        "reached_90": cost90 is not None,
        "cost_all": int(costall) if costall is not None else n*k,
        "reached_all": costall is not None,
        "cost_spent": rec["cost_cells"],
        "resolved_half": np.mean(np.asarray(half.get("decisions", rec["decisions"])) != UNKNOWN),
        "wrong_certificates": int(wrong.sum()),
        "any_wrong": bool(wrong.any()),
        "resolved_end": float(np.mean(dec != UNKNOWN)),
    }
