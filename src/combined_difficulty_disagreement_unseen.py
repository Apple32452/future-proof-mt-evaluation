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

ALPHA = 0.5   # difficulty weight
BETA = 0.5    # disagreement weight


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
    / "combined_dd_unseen"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_combined_dd_summary.csv"
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
# Helper: min-max normalization
# ============================================================

def minmax(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index,
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


# ============================================================
# Difficulty
# ============================================================

df["dev_mean_score"] = (
    dev_scores.mean(axis=1)
)

# Lower translation quality = greater difficulty
df["difficulty_raw"] = (
    -df["dev_mean_score"]
)

df["difficulty_norm"] = minmax(
    df["difficulty_raw"]
)


# ============================================================
# Disagreement
# ============================================================

df["disagreement_raw"] = (
    dev_scores.var(
        axis=1,
        ddof=0,
    )
)

df["disagreement_norm"] = minmax(
    df["disagreement_raw"]
)


# ============================================================
# Combined score
# ============================================================

df["combined_score"] = (
    ALPHA * df["difficulty_norm"]
    +
    BETA * df["disagreement_norm"]
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
# Experiment information
# ============================================================

print("=" * 80)
print("DIFFICULTY + DISAGREEMENT — UNSEEN SYSTEM EVALUATION")
print("=" * 80)

print(f"\nLanguage pair: {LANGUAGE_PAIR}")
print(f"Sentences:     {N}")
print(f"Dev systems:   {len(dev_systems)}")
print(f"Test systems:  {len(test_systems)}")
print(f"Budgets:       {BUDGETS}")

print(
    f"\nCombined score = "
    f"{ALPHA:.2f} * difficulty_norm "
    f"+ {BETA:.2f} * disagreement_norm"
)

print("\nIMPORTANT:")
print(
    "  All selection statistics use ONLY development systems."
)


# ============================================================
# Preview highest-scoring sentences
# ============================================================

preview = (
    df[
        [
            "sentence_id",
            "source_text",
            "dev_mean_score",
            "difficulty_norm",
            "disagreement_raw",
            "disagreement_norm",
            "combined_score",
        ]
    ]
    .sort_values(
        "combined_score",
        ascending=False,
    )
    .head(10)
)

print("\nTOP 10 COMBINED-SCORE SENTENCES")
print("-" * 80)

for rank, (_, row) in enumerate(
    preview.iterrows(),
    start=1,
):

    print(
        f"\n{rank:2d}. "
        f"combined={row['combined_score']:.4f} "
        f"D={row['difficulty_norm']:.4f} "
        f"V={row['disagreement_norm']:.4f}"
    )

    print(
        f"    dev_mean={row['dev_mean_score']:.3f}"
    )

    print(
        f"    {row['source_text'][:150]}"
    )


# ============================================================
# Rank sentences
# ============================================================

ranked_indices = (
    df["combined_score"]
    .sort_values(
        ascending=False
    )
    .index
    .to_numpy()
)


# ============================================================
# Evaluate budgets
# ============================================================

summary_rows = []

for budget in BUDGETS:

    selected_indices = (
        ranked_indices[:budget]
    )

    subset_scores = (
        test_scores
        .iloc[selected_indices]
        .mean(axis=0)
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = np.mean(
        np.abs(
            subset_scores.to_numpy()
            -
            full_test_scores.to_numpy()
        )
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

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

    summary_rows.append(
        {
            "budget": budget,
            "mae": mae,
            "spearman": spearman,
            "kendall": kendall,
            "pairwise_accuracy": pairwise,

            "selected_dev_mean_score":
                df.iloc[selected_indices][
                    "dev_mean_score"
                ].mean(),

            "selected_mean_difficulty_norm":
                df.iloc[selected_indices][
                    "difficulty_norm"
                ].mean(),

            "selected_mean_disagreement_norm":
                df.iloc[selected_indices][
                    "disagreement_norm"
                ].mean(),
        }
    )

    # --------------------------------------------------------
    # Save selected sentences
    # --------------------------------------------------------

    selected_df = (
        df.iloc[selected_indices][
            [
                "sentence_id",
                "source_text",
                "dev_mean_score",
                "difficulty_norm",
                "disagreement_raw",
                "disagreement_norm",
                "combined_score",
            ]
        ]
        .copy()
    )

    selected_df.insert(
        0,
        "selection_rank",
        range(1, budget + 1),
    )

    selected_file = (
        OUTPUT_DIR
        / f"{LANGUAGE_PAIR}_combined_dd_selected_m{budget}.csv"
    )

    selected_df.to_csv(
        selected_file,
        index=False,
    )


# ============================================================
# Save + print summary
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)

print("\n" + "=" * 80)
print("COMBINED DIFFICULTY + DISAGREEMENT SUMMARY")
print("=" * 80)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nLower MAE is better.")
print("Higher ranking metrics are better.")

print("\nSaved:")
print(SUMMARY_FILE)
