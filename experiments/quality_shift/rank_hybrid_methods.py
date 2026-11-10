"""Target-blind, weighted rank-informative stratified MT sampling.

The allocation rule is declared before evaluating the 12 additional WMT26
directions. It does not use target human scores, true rankings, or outcomes.
"""
from __future__ import annotations

import numpy as np


N_BINS = 5
REPRESENTATIVE_FRACTION = 0.40  # At budget 25, 10 representative + 15 informative.


def empirical_percentiles(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Need a finite vector of development-derived utilities")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = (np.arange(len(values)) + 0.5) / len(values)
    return ranks


def rank_information_bins(dev_eval: np.ndarray, bins: int = N_BINS):
    """Five equal-size groups, from least to most development-derived utility.

    Utility combines item difficulty (-mean dev cESA) and inter-system dev
    disagreement (std over model scores), using each signal's percentile rank.
    """
    if dev_eval.ndim != 2 or dev_eval.shape[1] < 2 or len(dev_eval) < bins:
        raise ValueError("Need an item-by-development-system score matrix")
    if not np.isfinite(dev_eval).all():
        raise ValueError("Development scores must be finite")
    difficulty = empirical_percentiles(-dev_eval.mean(axis=1))
    disagreement = empirical_percentiles(dev_eval.std(axis=1, ddof=1))
    utility = (difficulty + disagreement) / 2
    order = np.argsort(utility, kind="stable")
    labels = np.empty(len(order), dtype=int)
    for bin_index, group in enumerate(np.array_split(order, bins)):
        labels[group] = bin_index
    return labels, utility


def hybrid_allocation(counts: np.ndarray, budget: int):
    """40% representative across bins, remaining toward two highest-utility bins.

    A minimum of one draw per nonempty bin is imposed. Budget 25 produces
    a 10/15 representative/informative split when the two informative bins
    have adequate capacity. All assignments have documented inclusion weights.
    """
    counts = np.asarray(counts, dtype=int)
    if len(counts) != N_BINS or np.any(counts <= 0) or not len(counts):
        raise ValueError("Expected five nonempty strata")
    if budget < len(counts) or budget > counts.sum():
        raise ValueError("Budget must cover all strata without duplicates")
    representatives = max(len(counts), int(round(REPRESENTATIVE_FRACTION * budget)))
    representatives = min(representatives, budget)
    allocation = np.full(len(counts), representatives // len(counts), dtype=int)
    allocation[: representatives % len(counts)] += 1
    # In very small populations, redistribute any over-capacity base draws.
    excess = int(np.maximum(allocation - counts, 0).sum())
    allocation = np.minimum(allocation, counts)
    for _ in range(excess):
        candidates = np.flatnonzero(allocation < counts)
        allocation[int(candidates[0])] += 1
    remaining = budget - int(allocation.sum())
    for t in range(remaining):
        # Alternate upper-utility strata first. Fall back when they fill.
        preferred = [4, 3] if t % 2 == 0 else [3, 4]
        preferred += [2, 1, 0]
        chosen = next((i for i in preferred if allocation[i] < counts[i]), None)
        if chosen is None:
            raise ValueError("No available stratum for additional sampling")
        allocation[chosen] += 1
    if int(allocation.sum()) != budget or (allocation <= 0).any() or (allocation > counts).any():
        raise AssertionError("Invalid hybrid stratification allocation")
    return allocation


def rank_aware_hybrid_sample(dev_eval: np.ndarray, budget: int,
                             rng: np.random.Generator):
    labels, utility = rank_information_bins(dev_eval)
    strata = [np.flatnonzero(labels == i) for i in range(N_BINS)]
    counts = np.asarray([len(group) for group in strata], dtype=int)
    take = hybrid_allocation(counts, budget)
    selected = np.concatenate([rng.choice(group, size=int(n), replace=False)
                               for group, n in zip(strata, take)])
    weights = np.concatenate([np.full(int(n), len(group) / (len(dev_eval) * int(n)))
                              for group, n in zip(strata, take)])
    if len(set(selected)) != budget or not np.isclose(weights.sum(), 1, atol=1e-12):
        raise AssertionError("Hybrid must produce unique valid weighted samples")
    return selected.astype(int), weights, labels, utility
