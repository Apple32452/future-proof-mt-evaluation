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
    / "difficulty_baseline_unseen"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_difficulty_unseen_summary.csv"
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
# Compute sentence difficulty
# ============================================================

# A sentence is considered difficult if the development systems
# receive a low average human score on that sentence.
#
# Higher difficulty = harder sentence.

df["dev_mean_score"] = dev_scores.mean(axis=1)

df["difficulty"] = -df["dev_mean_score"]


# ============================================================
# Full-test-set reference scores for unseen systems
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

    if total == 0:
        return np.nan

    return correct / total


# ============================================================
# Experiment information
# ============================================================

print("=" * 80)
print("DIFFICULTY BASELINE — UNSEEN SYSTEM EVALUATION")
print("=" * 80)

print(f"\nLanguage pair: {LANGUAGE_PAIR}")
print(f"Sentences:     {N}")
print(f"Dev systems:   {len(dev_systems)}")
print(f"Test systems:  {len(test_systems)}")
print(f"Budgets:       {BUDGETS}")

print("\nDifficulty definition:")

print(
    "  difficulty(i) = "
    "- mean human score of development systems on sentence i"
)

print("\nIMPORTANT:")
print(
    "  Unseen test-system scores are NOT used "
    "for sentence selection."
)


# ============================================================
# Show hardest sentences
# ============================================================

hardest_preview = (
    df[
        [
            "sentence_id",
            "source_text",
            "dev_mean_score",
            "difficulty",
        ]
    ]
    .sort_values(
        "difficulty",
        ascending=False,
    )
    .head(10)
)

print("\nTOP 10 HARDEST SENTENCES ON DEVELOPMENT SYSTEMS")
print("-" * 80)

for rank, (_, row) in enumerate(
    hardest_preview.iterrows(),
    start=1,
):
    print(
        f"\n{rank:2d}. "
        f"dev_mean={row['dev_mean_score']:.3f}"
    )

    print(
        f"    {row['source_text'][:150]}"
    )


# ============================================================
# Evaluate each budget
# ============================================================

summary_rows = []

# Ranking is determined ONLY from development systems.
ranked_indices = (
    df["difficulty"]
    .sort_values(ascending=False)
    .index
    .to_numpy()
)

for budget in BUDGETS:

    if budget > N:
        raise ValueError(
            f"Budget {budget} exceeds N={N}"
        )

    selected_indices = ranked_indices[:budget]

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We now evaluate those selected sentence indices using
    # ONLY the unseen systems.
    # --------------------------------------------------------

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
            - full_test_scores.to_numpy()
        )
    )

    # --------------------------------------------------------
    # Ranking metrics
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

    # --------------------------------------------------------
    # Mean difficulty characteristics
    # --------------------------------------------------------

    selected_dev_mean = (
        df
        .iloc[selected_indices]["dev_mean_score"]
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
        }
    )

    # Save selected examples.
    selected_df = df.iloc[selected_indices][
        [
            "sentence_id",
            "source_text",
            "dev_mean_score",
            "difficulty",
        ]
    ].copy()

    selected_df.insert(
        0,
        "selection_rank",
        range(1, budget + 1),
    )

    selected_file = (
        OUTPUT_DIR
        / f"{LANGUAGE_PAIR}_difficulty_selected_m{budget}.csv"
    )

    selected_df.to_csv(
        selected_file,
        index=False,
    )


# ============================================================
# Save summary
# ============================================================

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 80)
print("DIFFICULTY BASELINE SUMMARY")
print("=" * 80)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nInterpretation:")
print("  Lower MAE is better.")
print("  Higher Spearman is better.")
print("  Higher Kendall tau is better.")
print("  Higher pairwise accuracy is better.")

print("\nSaved:")
print(SUMMARY_FILE)
