#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    spearmanr,
    kendalltau,
    wasserstein_distance,
)


# ============================================================
# Configuration
# ============================================================

LANGUAGE_PAIR = "cs-de_DE"

BUDGETS = [10, 25, 50, 100]

N_BINS = 5


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
    / "representative_disagreement_unseen"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_representative_disagreement_summary.csv"
)


# ============================================================
# Load data
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
# Development-only sentence statistics
# ============================================================

df["dev_mean_score"] = (
    dev_scores.mean(axis=1)
)

df["disagreement"] = (
    dev_scores.var(
        axis=1,
        ddof=0,
    )
)


# ============================================================
# Difficulty strata
#
# qcut divides sentences into approximately equally sized
# groups according to development-system mean score.
#
# Bin 0 = hardest
# Bin 4 = easiest
# ============================================================

df["difficulty_bin"] = pd.qcut(
    df["dev_mean_score"],
    q=N_BINS,
    labels=False,
    duplicates="drop",
)

num_actual_bins = (
    df["difficulty_bin"]
    .nunique()
)


# ============================================================
# Full unseen-system reference scores
# ============================================================

full_test_scores = (
    test_scores.mean(axis=0)
)


# ============================================================
# Pairwise ranking accuracy
# ============================================================

def pairwise_ranking_accuracy(
    reference_scores,
    estimated_scores,
):

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

    if total == 0:
        return np.nan

    return correct / total


# ============================================================
# Representative selection
# ============================================================

def select_representative_disagreement(
    dataframe,
    budget,
):

    bins = sorted(
        dataframe["difficulty_bin"]
        .dropna()
        .unique()
    )

    n_bins = len(bins)

    # Base number selected from each bin
    base = budget // n_bins

    # Remaining slots
    remainder = budget % n_bins

    selected = []

    for position, bin_id in enumerate(bins):

        local = dataframe[
            dataframe["difficulty_bin"] == bin_id
        ]

        # Equal allocation across quantile bins.
        n_select = base

        if position < remainder:
            n_select += 1

        # Within this difficulty region,
        # select sentences with highest disagreement.
        local_selected = (
            local
            .sort_values(
                "disagreement",
                ascending=False,
            )
            .head(n_select)
        )

        selected.extend(
            local_selected.index.tolist()
        )

    return np.array(
        selected,
        dtype=int,
    )


# ============================================================
# Experiment information
# ============================================================

print("=" * 80)
print("REPRESENTATIVE DISAGREEMENT — UNSEEN SYSTEM EVALUATION")
print("=" * 80)

print(f"\nLanguage pair: {LANGUAGE_PAIR}")
print(f"Sentences:     {N}")
print(f"Dev systems:   {len(dev_systems)}")
print(f"Test systems:  {len(test_systems)}")
print(f"Difficulty bins: {num_actual_bins}")
print(f"Budgets:       {BUDGETS}")

print("\nSelection rule:")
print(
    "  1. Bin sentences by development-system mean score."
)
print(
    "  2. Allocate selections across all difficulty bins."
)
print(
    "  3. Inside each bin, choose highest-disagreement sentences."
)

print("\nIMPORTANT:")
print(
    "  Unseen-system scores are never used for selection."
)


# ============================================================
# Bin summary
# ============================================================

print("\nDIFFICULTY-BIN SUMMARY")
print("-" * 80)

bin_summary = (
    df.groupby("difficulty_bin")
    .agg(
        sentences=("sentence_id", "count"),
        mean_dev_score=("dev_mean_score", "mean"),
        min_dev_score=("dev_mean_score", "min"),
        max_dev_score=("dev_mean_score", "max"),
        mean_disagreement=("disagreement", "mean"),
    )
)

print(
    bin_summary.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# Evaluate budgets
# ============================================================

summary_rows = []

full_dev_distribution = (
    df["dev_mean_score"]
    .to_numpy()
)

for budget in BUDGETS:

    selected_indices = (
        select_representative_disagreement(
            df,
            budget,
        )
    )

    if len(selected_indices) != budget:
        raise RuntimeError(
            f"Expected {budget} selected sentences, "
            f"got {len(selected_indices)}."
        )

    # ========================================================
    # Evaluate ONLY unseen systems
    # ========================================================

    subset_scores = (
        test_scores
        .iloc[selected_indices]
        .mean(axis=0)
    )

    # --------------------------------------------------------
    # Score preservation
    # --------------------------------------------------------

    mae = np.mean(
        np.abs(
            subset_scores.to_numpy()
            -
            full_test_scores.to_numpy()
        )
    )

    # --------------------------------------------------------
    # Ranking preservation
    # --------------------------------------------------------

    spearman = spearmanr(
        full_test_scores,
        subset_scores,
    ).statistic

    kendall = kendalltau(
        full_test_scores,
        subset_scores,
    ).statistic

    pairwise = (
        pairwise_ranking_accuracy(
            full_test_scores,
            subset_scores,
        )
    )

    # --------------------------------------------------------
    # Representativeness diagnostics
    # --------------------------------------------------------

    selected_dev_distribution = (
        df
        .iloc[selected_indices][
            "dev_mean_score"
        ]
        .to_numpy()
    )

    wasserstein = wasserstein_distance(
        full_dev_distribution,
        selected_dev_distribution,
    )

    full_dev_mean = (
        df["dev_mean_score"].mean()
    )

    selected_dev_mean = (
        df
        .iloc[selected_indices][
            "dev_mean_score"
        ]
        .mean()
    )

    mean_shift = (
        selected_dev_mean
        -
        full_dev_mean
    )

    selected_disagreement = (
        df
        .iloc[selected_indices][
            "disagreement"
        ]
        .mean()
    )

    summary_rows.append(
        {
            "budget": budget,
            "mae": mae,
            "spearman": spearman,
            "kendall": kendall,
            "pairwise_accuracy": pairwise,
            "selected_dev_mean_score":
                selected_dev_mean,
            "dev_mean_shift":
                mean_shift,
            "wasserstein_dev_score":
                wasserstein,
            "selected_mean_disagreement":
                selected_disagreement,
        }
    )

    # ========================================================
    # Save selected sentences
    # ========================================================

    selected_df = (
        df.iloc[selected_indices][
            [
                "sentence_id",
                "source_text",
                "difficulty_bin",
                "dev_mean_score",
                "disagreement",
            ]
        ]
        .copy()
    )

    selected_df = (
        selected_df
        .sort_values(
            [
                "difficulty_bin",
                "disagreement",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    selected_file = (
        OUTPUT_DIR
        / (
            f"{LANGUAGE_PAIR}_"
            f"representative_disagreement_m{budget}.csv"
        )
    )

    selected_df.to_csv(
        selected_file,
        index=False,
    )


# ============================================================
# Save results
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# Print
# ============================================================

print("\n" + "=" * 80)
print("REPRESENTATIVE DISAGREEMENT SUMMARY")
print("=" * 80)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nMetrics:")
print("  Lower MAE is better.")
print("  Higher ranking metrics are better.")
print("  dev_mean_shift closer to 0 is better.")
print("  Lower Wasserstein distance means better difficulty-distribution coverage.")

print("\nSaved:")
print(SUMMARY_FILE)
