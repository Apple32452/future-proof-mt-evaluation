#!/usr/bin/env python3
"""Frozen calibration-driven pairwise-difference-variance stratified design.

Inputs to this module contain ONLY role/variance-calibration target labels
and development-system labels. Evaluation target labels never enter the API.
"""
from __future__ import annotations

import numpy as np


N_STRATA = 3
SHRINKAGE = 5.0
PAIR_GAP_OFFSET = 5.0


def role_weights(role_target: np.ndarray) -> np.ndarray:
    """Importance of the six pairs, using role-calibration target means."""
    if role_target.ndim != 2 or role_target.shape[1] != 4 or not len(role_target):
        raise ValueError("Expected four role-calibration target systems")
    means = role_target.mean(axis=0)
    aa, bb = np.triu_indices(4, 1)
    weights = (PAIR_GAP_OFFSET + np.abs(means[aa] - means[bb])) ** -2
    return weights / weights.sum()


def make_strata(dev_eval: np.ndarray, dev_variance_cal: np.ndarray):
    """Equal-count evaluation tertiles and calibration bin mapping.

    Ties are broken stably by item order in evaluation. Calibration items
    are mapped by numerical tertile boundaries, never by target labels.
    """
    if dev_eval.ndim != 2 or dev_eval.shape[1] != 4 or len(dev_eval) < N_STRATA:
        raise ValueError("Three strata require four development systems")
    if dev_variance_cal.ndim != 2 or dev_variance_cal.shape[1] != 4:
        raise ValueError("Invalid variance-calibration development scores")
    difficulty = -dev_eval.mean(axis=1)
    ordered = np.argsort(difficulty, kind="stable")
    groups = np.array_split(ordered, N_STRATA)
    labels = np.empty(len(dev_eval), dtype=np.int8)
    for h, ids in enumerate(groups):
        labels[ids] = h
    boundaries = np.array([difficulty[group[-1]] for group in groups[:-1]])
    cal_diff = -dev_variance_cal.mean(axis=1)
    cal_labels = np.searchsorted(boundaries, cal_diff, side="right").astype(np.int8)
    return labels, cal_labels, boundaries


def pairwise_dispersion(variance_target: np.ndarray, cal_labels: np.ndarray,
                        pair_importance: np.ndarray) -> np.ndarray:
    """Global-shrunk within-stratum difference variance for all six pairs."""
    if variance_target.ndim != 2 or variance_target.shape[1] != 4 \
            or len(variance_target) != len(cal_labels) or len(variance_target) < 2:
        raise ValueError("Need at least two variance-calibration target items")
    if pair_importance.shape != (6,) or not np.isclose(pair_importance.sum(), 1):
        raise ValueError("Invalid six-pair weights")
    aa, bb = np.triu_indices(4, 1)
    differences = variance_target[:, aa] - variance_target[:, bb]
    global_var = differences.var(axis=0, ddof=1)
    dispersions = []
    for h in range(N_STRATA):
        sub = differences[cal_labels == h]
        if len(sub) >= 2:
            local = sub.var(axis=0, ddof=1)
            empirical_df = len(sub) - 1
        else:
            local = np.zeros(6, dtype=float)
            empirical_df = 0
        shrunk = (empirical_df * local + SHRINKAGE * global_var) / (empirical_df + SHRINKAGE)
        dispersions.append(float(np.sqrt(pair_importance @ np.maximum(shrunk, 0))))
    result = np.asarray(dispersions, dtype=float)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Nonfinite or negative pairwise stratum dispersion")
    return result


def allocated_counts(populations: np.ndarray, budget: int,
                     dispersion: np.ndarray | None = None) -> np.ndarray:
    """Capped integer Neyman allocation, with one per nonempty stratum."""
    sizes = np.asarray(populations, dtype=int)
    if sizes.ndim != 1 or not len(sizes) or (sizes < 0).any():
        raise ValueError("Invalid stratum population sizes")
    active = sizes > 0
    if budget < active.sum() or budget > sizes.sum():
        raise ValueError("Budget cannot cover active strata or exceeds population")
    weight = sizes.astype(float)
    if dispersion is not None:
        spread = np.asarray(dispersion, dtype=float)
        if spread.shape != sizes.shape or not np.isfinite(spread).all() or (spread < 0).any():
            raise ValueError("Invalid dispersion estimates")
        neyman = weight * spread
        if neyman.sum() > 1e-15:
            weight = neyman
    wanted = budget * weight / weight.sum()
    allocated = active.astype(int)
    for _ in range(budget - int(active.sum())):
        deficits = np.where(allocated < sizes, wanted - allocated, -np.inf)
        allocated[int(np.argmax(deficits))] += 1
    if allocated.sum() != budget or (allocated > sizes).any() \
            or ((allocated == 0) & active).any():
        raise AssertionError("Invalid bounded Neyman allocation")
    return allocated


def draw_strata(labels: np.ndarray, allocation: np.ndarray, rng: np.random.Generator):
    """Draw without replacement; return indices and unbiased mean weights."""
    population = len(labels)
    if population == 0 or not len(allocation):
        raise ValueError("No evaluation item or stratum allocation")
    selected, weights = [], []
    for h, take in enumerate(allocation):
        members = np.flatnonzero(labels == h)
        if take <= 0 or take > len(members):
            raise ValueError("Empty or oversubscribed stratum")
        indices = rng.choice(members, size=int(take), replace=False)
        selected.append(indices)
        weights.append(np.full(int(take), len(members) / (population * int(take))))
    picks, item_weights = np.concatenate(selected), np.concatenate(weights)
    if len(np.unique(picks)) != len(picks) or not np.isclose(item_weights.sum(), 1, atol=1e-12):
        raise AssertionError("Stratified weighted estimator is invalid")
    return picks.astype(int), item_weights


def calibration_design(role_target: np.ndarray, variance_target: np.ndarray,
                       dev_eval: np.ndarray, dev_variance_cal: np.ndarray):
    """Produce a fixed selection design before revealing evaluation targets."""
    weights = role_weights(role_target)
    labels, calibration_labels, boundaries = make_strata(dev_eval, dev_variance_cal)
    dispersion = pairwise_dispersion(variance_target, calibration_labels, weights)
    sizes = np.bincount(labels, minlength=N_STRATA)
    return {"labels": labels, "calibration_labels": calibration_labels,
            "boundaries": boundaries, "pair_weights": weights,
            "dispersion": dispersion, "sizes": sizes}
