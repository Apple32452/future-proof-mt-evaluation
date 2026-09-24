#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


# ============================================================
# Configuration
# ============================================================

LANGUAGE_PAIR = "cs-de_DE"

BUDGETS = [10, 25, 50, 100]

N_TRIALS = 1000
BASE_SEED = 42


# ============================================================
# Paths
# ============================================================

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
    / "random_baseline_unseen"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRIAL_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_random_unseen_trials.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_random_unseen_summary.csv"
)


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(MATRIX_FILE)

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split = json.load(f)

dev_systems = split["dev_systems"]
test_systems = split["test_systems"]

test_scores = df[test_systems]

N = len(df)
K_TEST = len(test_systems)


# ============================================================
# Full-test-set reference scores
# ============================================================

# IMPORTANT:
# These scores use ALL 231 sentences, but ONLY the unseen systems.
#
# The subset methods later must try to reproduce these values
# without using unseen-system scores during sentence selection.

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

            # Ignore exact ties
            if ref_diff == 0 or est_diff == 0:
                continue

            total += 1

            if np.sign(ref_diff) == np.sign(est_diff):
                correct += 1

    if total == 0:
        return np.nan

    return correct / total


# ============================================================
# Print experiment setup
# ============================================================

print("=" * 80)
print("RANDOM BASELINE — UNSEEN SYSTEM EVALUATION")
print("=" * 80)

print(f"\nLanguage pair: {LANGUAGE_PAIR}")
print(f"Sentences:     {N}")
print(f"Dev systems:   {len(dev_systems)}")
print(f"Test systems:  {K_TEST}")
print(f"Trials:        {N_TRIALS}")
print(f"Budgets:       {BUDGETS}")

print("\nUNSEEN TEST SYSTEMS")

for system in test_systems:
    print(f"  {system}")

print("\nFULL-DATASET REFERENCE SCORES — UNSEEN SYSTEMS")

ranking = (
    full_test_scores
    .sort_values(ascending=False)
)

for rank, (system, score) in enumerate(
    ranking.items(),
    start=1,
):
    print(
        f"{rank:2d}. "
        f"{system:25s} "
        f"{score:.3f}"
    )


# ============================================================
# Run random subset experiments
# ============================================================

trial_results = []

for budget in BUDGETS:

    if budget > N:
        raise ValueError(
            f"Budget {budget} exceeds N={N}"
        )

    print(f"\nRunning budget m={budget}...")

    for trial in range(N_TRIALS):

        rng = np.random.default_rng(
            BASE_SEED
            + budget * 10000
            + trial
        )

        selected_indices = rng.choice(
            N,
            size=budget,
            replace=False,
        )

        subset_scores = (
            test_scores
            .iloc[selected_indices]
            .mean(axis=0)
        )

        # --------------------------------------------
        # Score preservation
        # --------------------------------------------

        mae = np.mean(
            np.abs(
                subset_scores.to_numpy()
                - full_test_scores.to_numpy()
            )
        )

        # --------------------------------------------
        # Ranking preservation
        # --------------------------------------------

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

        trial_results.append(
            {
                "budget": budget,
                "trial": trial,
                "mae": mae,
                "spearman": spearman,
                "kendall": kendall,
                "pairwise_accuracy": pairwise,
            }
        )


# ============================================================
# Trial-level dataframe
# ============================================================

trial_df = pd.DataFrame(
    trial_results
)

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

            "mae_p05":
                local["mae"].quantile(0.05),

            "mae_p95":
                local["mae"].quantile(0.95),

            "spearman_mean":
                local["spearman"].mean(),

            "spearman_std":
                local["spearman"].std(),

            "kendall_mean":
                local["kendall"].mean(),

            "kendall_std":
                local["kendall"].std(),

            "pairwise_accuracy_mean":
                local["pairwise_accuracy"].mean(),

            "pairwise_accuracy_std":
                local["pairwise_accuracy"].std(),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 80)
print("UNSEEN-SYSTEM RANDOM BASELINE SUMMARY")
print("=" * 80)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nLower MAE is better.")
print("Higher Spearman is better.")
print("Higher Kendall tau is better.")
print("Higher pairwise accuracy is better.")

print("\nSaved:")
print(TRIAL_FILE)
print(SUMMARY_FILE)
