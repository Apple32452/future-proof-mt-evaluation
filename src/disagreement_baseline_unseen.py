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
    / "disagreement_baseline_unseen"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_disagreement_unseen_summary.csv"
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
# Compute system disagreement
# ============================================================

# Population variance across DEVELOPMENT systems only.
#
# High variance means the systems disagree strongly about
# this sentence.

df["dev_mean_score"] = dev_scores.mean(axis=1)

df["disagreement"] = dev_scores.var(
    axis=1,
    ddof=0,
)

df["dev_score_std"] = np.sqrt(
    df["disagreement"]
)


# ============================================================
# Full reference scores for unseen systems
# ============================================================

full_test_scores = test_scores.mean(axis=0)


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
# Print experiment information
# ============================================================

print("=" * 80)
print("DISAGREEMENT BASELINE — UNSEEN SYSTEM EVALUATION")
print("=" * 80)

print(f"\nLanguage pair: {LANGUAGE_PAIR}")
print(f"Sentences:     {N}")
print(f"Dev systems:   {len(dev_systems)}")
print(f"Test systems:  {len(test_systems)}")
print(f"Budgets:       {BUDGETS}")

print("\nDisagreement definition:")
print(
    "  disagreement(i) = variance of development-system "
    "scores on sentence i"
)

print("\nIMPORTANT:")
print(
    "  Unseen test-system scores are NOT used "
    "for sentence selection."
)


# ============================================================
# Preview highest-disagreement sentences
# ============================================================

preview = (
    df[
        [
            "sentence_id",
            "source_text",
            "dev_mean_score",
            "dev_score_std",
            "disagreement",
        ]
    ]
    .sort_values(
        "disagreement",
        ascending=False,
    )
    .head(10)
)

print("\nTOP 10 HIGHEST-DISAGREEMENT SENTENCES")
print("-" * 80)

for rank, (_, row) in enumerate(
    preview.iterrows(),
    start=1,
):

    print(
        f"\n{rank:2d}. "
        f"mean={row['dev_mean_score']:.3f} "
        f"std={row['dev_score_std']:.3f} "
        f"variance={row['disagreement']:.3f}"
    )

    print(
        f"    {row['source_text'][:150]}"
    )


# ============================================================
# Rank sentences using DEVELOPMENT systems only
# ============================================================

ranked_indices = (
    df["disagreement"]
    .sort_values(ascending=False)
    .index
    .to_numpy()
)


# ============================================================
# Evaluate budgets on unseen systems
# ============================================================

summary_rows = []

for budget in BUDGETS:

    if budget > N:
        raise ValueError(
            f"Budget {budget} exceeds N={N}"
        )

    selected_indices = ranked_indices[:budget]

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
            - full_test_scores.to_numpy()
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

    pairwise = pairwise_ranking_accuracy(
        full_test_scores,
        subset_scores,
    )

    selected_mean_disagreement = (
        df
        .iloc[selected_indices]["disagreement"]
        .mean()
    )

    selected_mean_dev_score = (
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
            "selected_mean_disagreement":
                selected_mean_disagreement,
            "selected_dev_mean_score":
                selected_mean_dev_score,
        }
    )

    # Save selected sentence list
    selected_df = df.iloc[selected_indices][
        [
            "sentence_id",
            "source_text",
            "dev_mean_score",
            "dev_score_std",
            "disagreement",
        ]
    ].copy()

    selected_df.insert(
        0,
        "selection_rank",
        range(1, budget + 1),
    )

    selected_file = (
        OUTPUT_DIR
        / f"{LANGUAGE_PAIR}_disagreement_selected_m{budget}.csv"
    )

    selected_df.to_csv(
        selected_file,
        index=False,
    )


# ============================================================
# Save summary
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# Print summary
# ============================================================

print("\n" + "=" * 80)
print("DISAGREEMENT BASELINE SUMMARY")
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
