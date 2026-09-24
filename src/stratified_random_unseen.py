#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


LANGUAGE_PAIR = "cs-de_DE"

BUDGETS = [10, 25, 50, 100]
N_BINS = 5
N_TRIALS = 1000
BASE_SEED = 42


ROOT = Path(__file__).resolve().parents[1]

MATRIX_FILE = (
    ROOT
    / "results"
    / "score_matrices"
    / f"{LANGUAGE_PAIR}_human_scores.csv"
)

SPLIT_FILE = (
    ROOT
    / "results"
    / "splits"
    / f"{LANGUAGE_PAIR}_pilot_system_split.json"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "stratified_random_unseen"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRIAL_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_stratified_random_trials.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_stratified_random_summary.csv"
)


# ============================================================
# Load
# ============================================================

df = pd.read_csv(MATRIX_FILE)

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split = json.load(f)

dev_systems = split["dev_systems"]
test_systems = split["test_systems"]

dev_scores = df[dev_systems]
test_scores = df[test_systems]

N = len(df)


# ============================================================
# Development-only difficulty strata
# ============================================================

df["dev_mean_score"] = dev_scores.mean(axis=1)

df["difficulty_bin"] = pd.qcut(
    df["dev_mean_score"],
    q=N_BINS,
    labels=False,
    duplicates="drop",
)

bins = sorted(
    df["difficulty_bin"]
    .dropna()
    .unique()
)


# ============================================================
# Full unseen-system reference
# ============================================================

full_test_scores = test_scores.mean(axis=0)


# ============================================================
# Pairwise ranking accuracy
# ============================================================

def pairwise_ranking_accuracy(reference_scores, estimated_scores):

    ref = reference_scores.to_numpy()
    est = estimated_scores.to_numpy()

    correct = 0
    total = 0

    for i in range(len(ref)):
        for j in range(i + 1, len(ref)):

            ref_diff = ref[i] - ref[j]
            est_diff = est[i] - est[j]

            if ref_diff == 0 or est_diff == 0:
                continue

            total += 1

            if np.sign(ref_diff) == np.sign(est_diff):
                correct += 1

    return correct / total if total else np.nan


# ============================================================
# Stratified random selector
# ============================================================

def select_stratified_random(dataframe, budget, rng):

    n_bins = len(bins)

    base = budget // n_bins
    remainder = budget % n_bins

    selected = []

    for position, bin_id in enumerate(bins):

        local_indices = (
            dataframe[
                dataframe["difficulty_bin"] == bin_id
            ]
            .index
            .to_numpy()
        )

        n_select = base

        if position < remainder:
            n_select += 1

        chosen = rng.choice(
            local_indices,
            size=n_select,
            replace=False,
        )

        selected.extend(
            chosen.tolist()
        )

    return np.array(selected, dtype=int)


# ============================================================
# Run
# ============================================================

print("=" * 80)
print("STRATIFIED RANDOM — UNSEEN SYSTEM EVALUATION")
print("=" * 80)

print(f"\nLanguage pair: {LANGUAGE_PAIR}")
print(f"Sentences:     {N}")
print(f"Dev systems:   {len(dev_systems)}")
print(f"Test systems:  {len(test_systems)}")
print(f"Difficulty bins: {len(bins)}")
print(f"Trials:        {N_TRIALS}")
print(f"Budgets:       {BUDGETS}")


trial_rows = []

for budget in BUDGETS:

    print(f"\nRunning budget m={budget}...")

    for trial in range(N_TRIALS):

        rng = np.random.default_rng(
            BASE_SEED
            + budget * 10000
            + trial
        )

        selected_indices = (
            select_stratified_random(
                df,
                budget,
                rng,
            )
        )

        subset_scores = (
            test_scores
            .iloc[selected_indices]
            .mean(axis=0)
        )

        mae = np.mean(
            np.abs(
                subset_scores.to_numpy()
                -
                full_test_scores.to_numpy()
            )
        )

        spearman = spearmanr(
            full_test_scores,
            subset_scores,
        ).statistic

        kendall = kendalltau(
            full_test_scores,
            subset_scores,
        ).statistic

        pairwise = pairwise_ranking_accuracy(
            full_test_scores,
            subset_scores,
        )

        selected_dev_mean = (
            df
            .iloc[selected_indices][
                "dev_mean_score"
            ]
            .mean()
        )

        trial_rows.append(
            {
                "budget": budget,
                "trial": trial,
                "mae": mae,
                "spearman": spearman,
                "kendall": kendall,
                "pairwise_accuracy": pairwise,
                "selected_dev_mean_score":
                    selected_dev_mean,
            }
        )


trial_df = pd.DataFrame(trial_rows)

trial_df.to_csv(
    TRIAL_FILE,
    index=False,
)


# ============================================================
# Summary
# ============================================================

summary_rows = []

for budget in BUDGETS:

    local = trial_df[
        trial_df["budget"] == budget
    ]

    summary_rows.append(
        {
            "budget": budget,

            "mae_mean":
                local["mae"].mean(),

            "mae_std":
                local["mae"].std(),

            "spearman_mean":
                local["spearman"].mean(),

            "spearman_std":
                local["spearman"].std(),

            "kendall_mean":
                local["kendall"].mean(),

            "kendall_std":
                local["kendall"].std(),

            "pairwise_accuracy_mean":
                local[
                    "pairwise_accuracy"
                ].mean(),

            "pairwise_accuracy_std":
                local[
                    "pairwise_accuracy"
                ].std(),

            "selected_dev_mean_score_mean":
                local[
                    "selected_dev_mean_score"
                ].mean(),
        }
    )


summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)

print("\n" + "=" * 80)
print("STRATIFIED RANDOM SUMMARY")
print("=" * 80)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nSaved:")
print(TRIAL_FILE)
print(SUMMARY_FILE)
