#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


# ============================================================
# Configuration
# ============================================================

LANGUAGE_PAIR = "cs-de_DE"

N_TEST_SYSTEMS = 6
N_SYSTEM_SPLITS = 100

# Random and stratified-random are stochastic.
# Average each one over several sentence samples per system split.
N_SENTENCE_TRIALS = 50

BUDGETS = [10, 25, 50, 100]

N_BINS = 5

SYSTEM_SPLIT_SEED = 2026


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

OUTPUT_DIR = (
    ROOT
    / "results"
    / "multisplit_benchmark"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DETAIL_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_multisplit_detail.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_multisplit_summary.csv"
)


# ============================================================
# Load matrix
# ============================================================

df = pd.read_csv(MATRIX_FILE)

metadata_columns = [
    "sentence_id",
    "source_text",
]

all_systems = sorted(
    [
        column
        for column in df.columns
        if column not in metadata_columns
    ]
)

N = len(df)

print("=" * 80)
print("MULTI-SPLIT UNSEEN-SYSTEM BENCHMARK")
print("=" * 80)

print(f"\nLanguage pair:        {LANGUAGE_PAIR}")
print(f"Sentences:            {N}")
print(f"Total systems:        {len(all_systems)}")
print(f"Development systems:  {len(all_systems) - N_TEST_SYSTEMS}")
print(f"Unseen systems:       {N_TEST_SYSTEMS}")
print(f"System splits:        {N_SYSTEM_SPLITS}")
print(f"Sentence trials:      {N_SENTENCE_TRIALS}")
print(f"Budgets:              {BUDGETS}")


# ============================================================
# Metrics
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

    return correct / total if total else np.nan


def evaluate_subset(
    test_scores,
    full_scores,
    selected_indices,
):

    subset_scores = (
        test_scores
        .iloc[selected_indices]
        .mean(axis=0)
    )

    mae = np.mean(
        np.abs(
            subset_scores.to_numpy()
            - full_scores.to_numpy()
        )
    )

    spearman = spearmanr(
        full_scores,
        subset_scores,
    ).statistic

    kendall = kendalltau(
        full_scores,
        subset_scores,
    ).statistic

    pairwise = pairwise_ranking_accuracy(
        full_scores,
        subset_scores,
    )

    return {
        "mae": mae,
        "spearman": spearman,
        "kendall": kendall,
        "pairwise_accuracy": pairwise,
    }


# ============================================================
# Generate UNIQUE random system splits
# ============================================================

rng_split = np.random.default_rng(
    SYSTEM_SPLIT_SEED
)

test_sets = set()

while len(test_sets) < N_SYSTEM_SPLITS:

    test_systems = tuple(
        sorted(
            rng_split.choice(
                all_systems,
                size=N_TEST_SYSTEMS,
                replace=False,
            ).tolist()
        )
    )

    test_sets.add(test_systems)

test_sets = sorted(test_sets)


# ============================================================
# Main benchmark
# ============================================================

rows = []

for split_index, test_tuple in enumerate(
    test_sets
):

    test_systems = list(test_tuple)

    dev_systems = [
        system
        for system in all_systems
        if system not in test_systems
    ]

    dev_scores = df[dev_systems]
    test_scores = df[test_systems]

    full_test_scores = (
        test_scores.mean(axis=0)
    )

    # --------------------------------------------------------
    # Development-only statistics
    # --------------------------------------------------------

    dev_mean_score = (
        dev_scores.mean(axis=1)
    )

    disagreement = (
        dev_scores.var(
            axis=1,
            ddof=0,
        )
    )

    difficulty_bin = pd.qcut(
        dev_mean_score,
        q=N_BINS,
        labels=False,
        duplicates="drop",
    )

    working = pd.DataFrame(
        {
            "dev_mean_score":
                dev_mean_score,

            "disagreement":
                disagreement,

            "difficulty_bin":
                difficulty_bin,
        }
    )

    bins = sorted(
        working["difficulty_bin"]
        .dropna()
        .unique()
    )

    # --------------------------------------------------------
    # Representative-disagreement ranking
    # --------------------------------------------------------

    def select_representative_disagreement(
        budget,
    ):

        base = budget // len(bins)
        remainder = budget % len(bins)

        selected = []

        for position, bin_id in enumerate(bins):

            local = working[
                working["difficulty_bin"] == bin_id
            ]

            n_select = base

            if position < remainder:
                n_select += 1

            chosen = (
                local
                .sort_values(
                    "disagreement",
                    ascending=False,
                )
                .head(n_select)
                .index
                .tolist()
            )

            selected.extend(chosen)

        return np.asarray(
            selected,
            dtype=int,
        )

    # --------------------------------------------------------
    # Stratified random selector
    # --------------------------------------------------------

    def select_stratified_random(
        budget,
        rng,
    ):

        base = budget // len(bins)
        remainder = budget % len(bins)

        selected = []

        for position, bin_id in enumerate(bins):

            candidates = (
                working[
                    working["difficulty_bin"]
                    == bin_id
                ]
                .index
                .to_numpy()
            )

            n_select = base

            if position < remainder:
                n_select += 1

            chosen = rng.choice(
                candidates,
                size=n_select,
                replace=False,
            )

            selected.extend(
                chosen.tolist()
            )

        return np.asarray(
            selected,
            dtype=int,
        )

    # --------------------------------------------------------
    # Budgets
    # --------------------------------------------------------

    for budget in BUDGETS:

        # ====================================================
        # 1. Ordinary random
        # ====================================================

        random_metrics = []

        for trial in range(N_SENTENCE_TRIALS):

            rng = np.random.default_rng(
                10000000
                + split_index * 100000
                + budget * 100
                + trial
            )

            selected = rng.choice(
                N,
                size=budget,
                replace=False,
            )

            random_metrics.append(
                evaluate_subset(
                    test_scores,
                    full_test_scores,
                    selected,
                )
            )

        random_df = pd.DataFrame(
            random_metrics
        )

        rows.append(
            {
                "split": split_index,
                "budget": budget,
                "method": "random",

                "mae":
                    random_df["mae"].mean(),

                "spearman":
                    random_df["spearman"].mean(),

                "kendall":
                    random_df["kendall"].mean(),

                "pairwise_accuracy":
                    random_df[
                        "pairwise_accuracy"
                    ].mean(),
            }
        )

        # ====================================================
        # 2. Stratified random
        # ====================================================

        strat_metrics = []

        for trial in range(N_SENTENCE_TRIALS):

            rng = np.random.default_rng(
                20000000
                + split_index * 100000
                + budget * 100
                + trial
            )

            selected = (
                select_stratified_random(
                    budget,
                    rng,
                )
            )

            strat_metrics.append(
                evaluate_subset(
                    test_scores,
                    full_test_scores,
                    selected,
                )
            )

        strat_df = pd.DataFrame(
            strat_metrics
        )

        rows.append(
            {
                "split": split_index,
                "budget": budget,
                "method": "stratified_random",

                "mae":
                    strat_df["mae"].mean(),

                "spearman":
                    strat_df["spearman"].mean(),

                "kendall":
                    strat_df["kendall"].mean(),

                "pairwise_accuracy":
                    strat_df[
                        "pairwise_accuracy"
                    ].mean(),
            }
        )

        # ====================================================
        # 3. Representative disagreement
        # ====================================================

        selected = (
            select_representative_disagreement(
                budget
            )
        )

        metrics = evaluate_subset(
            test_scores,
            full_test_scores,
            selected,
        )

        rows.append(
            {
                "split": split_index,
                "budget": budget,
                "method":
                    "representative_disagreement",

                **metrics,
            }
        )

    if (
        (split_index + 1) % 10
        == 0
    ):
        print(
            f"Completed "
            f"{split_index + 1}/"
            f"{N_SYSTEM_SPLITS} system splits"
        )


# ============================================================
# Save detail
# ============================================================

detail_df = pd.DataFrame(rows)

detail_df.to_csv(
    DETAIL_FILE,
    index=False,
)


# ============================================================
# Aggregate across SYSTEM SPLITS
# ============================================================

summary = (
    detail_df
    .groupby(
        [
            "method",
            "budget",
        ]
    )
    .agg(
        mae_mean=(
            "mae",
            "mean",
        ),
        mae_std=(
            "mae",
            "std",
        ),

        spearman_mean=(
            "spearman",
            "mean",
        ),
        spearman_std=(
            "spearman",
            "std",
        ),

        kendall_mean=(
            "kendall",
            "mean",
        ),
        kendall_std=(
            "kendall",
            "std",
        ),

        pairwise_mean=(
            "pairwise_accuracy",
            "mean",
        ),
        pairwise_std=(
            "pairwise_accuracy",
            "std",
        ),
    )
    .reset_index()
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# Print
# ============================================================

print("\n" + "=" * 100)
print("MULTI-SPLIT SUMMARY")
print("=" * 100)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nSaved detail:")
print(DETAIL_FILE)

print("\nSaved summary:")
print(SUMMARY_FILE)
