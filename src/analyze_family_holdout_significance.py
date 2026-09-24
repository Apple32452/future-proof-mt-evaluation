#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

N_BOOTSTRAP = 20000
SEED = 42


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results"
    / "family_holdout_benchmark"
    / "family_holdout_case_summary.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "family_holdout_significance"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "family_holdout_significance_summary.csv"
)

FAMILY_FILE = (
    OUTPUT_DIR
    / "family_holdout_per_family_effects.csv"
)


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Recompute explicitly for safety.
df["mae_improvement"] = (
    df["random_mae"]
    - df["stratified_mae"]
)

df["relative_improvement_pct"] = (
    100.0
    * df["mae_improvement"]
    / df["random_mae"]
)


# ============================================================
# Family-level effects
# ============================================================

family_effects = (
    df
    .groupby(
        [
            "budget",
            "family",
        ]
    )
    .agg(
        num_cases=(
            "language_pair",
            "count",
        ),

        mean_improvement=(
            "mae_improvement",
            "mean",
        ),

        median_improvement=(
            "mae_improvement",
            "median",
        ),

        mean_relative_improvement_pct=(
            "relative_improvement_pct",
            "mean",
        ),

        win_rate=(
            "stratified_better_mae",
            "mean",
        ),
    )
    .reset_index()
)

family_effects.to_csv(
    FAMILY_FILE,
    index=False,
)


# ============================================================
# Clustered / family-balanced bootstrap
# ============================================================

def bootstrap_family_balanced(local, seed):

    rng = np.random.default_rng(seed)

    languages = sorted(
        local["language_pair"].unique()
    )

    boot_values = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    for b in range(N_BOOTSTRAP):

        # ----------------------------------------------------
        # Resample LANGUAGE PAIRS as clusters.
        #
        # If a language is sampled, all of its family-holdout
        # cases travel together.
        # ----------------------------------------------------

        sampled_languages = rng.choice(
            languages,
            size=len(languages),
            replace=True,
        )

        pieces = []

        for draw_id, language in enumerate(sampled_languages):

            temp = local[
                local["language_pair"] == language
            ].copy()

            # Keep repeated bootstrap draws distinct.
            temp["bootstrap_draw"] = draw_id

            pieces.append(temp)

        boot = pd.concat(
            pieces,
            ignore_index=True,
        )

        # ----------------------------------------------------
        # First average cases WITHIN each family.
        #
        # Then average the family means.
        #
        # This prevents Gemma, which has many more cases,
        # from dominating the macro result.
        # ----------------------------------------------------

        family_means = (
            boot
            .groupby("family")[
                "mae_improvement"
            ]
            .mean()
        )

        boot_values[b] = (
            family_means.mean()
        )

    return boot_values


# ============================================================
# Summary by budget
# ============================================================

summary_rows = []

for budget in sorted(
    df["budget"].unique()
):

    local = df[
        df["budget"] == budget
    ].copy()

    improvements = (
        local["mae_improvement"]
        .to_numpy()
    )

    relative = (
        local["relative_improvement_pct"]
        .to_numpy()
    )

    wins = int(
        (improvements > 0).sum()
    )

    losses = int(
        (improvements < 0).sum()
    )

    ties = int(
        (improvements == 0).sum()
    )

    # --------------------------------------------------------
    # Family-balanced observed effect
    # --------------------------------------------------------

    observed_family_means = (
        local
        .groupby("family")[
            "mae_improvement"
        ]
        .mean()
    )

    observed_family_balanced = (
        observed_family_means.mean()
    )

    # Relative improvement, also family balanced
    observed_relative_family_means = (
        local
        .groupby("family")[
            "relative_improvement_pct"
        ]
        .mean()
    )

    observed_relative_family_balanced = (
        observed_relative_family_means.mean()
    )

    # --------------------------------------------------------
    # Cluster bootstrap
    # --------------------------------------------------------

    boot = bootstrap_family_balanced(
        local,
        SEED + int(budget),
    )

    ci_low = np.quantile(
        boot,
        0.025,
    )

    ci_high = np.quantile(
        boot,
        0.975,
    )

    # Fraction of bootstrap samples where improvement <= 0.
    # Useful as an uncertainty diagnostic, not a classical
    # hypothesis-test p-value.
    bootstrap_nonpositive_rate = (
        np.mean(boot <= 0)
    )

    positive_families = int(
        (observed_family_means > 0).sum()
    )

    negative_families = int(
        (observed_family_means < 0).sum()
    )

    summary_rows.append(
        {
            "budget": budget,

            "num_cases":
                len(local),

            "num_languages":
                local[
                    "language_pair"
                ].nunique(),

            "num_families":
                local[
                    "family"
                ].nunique(),

            "case_mean_mae_improvement":
                improvements.mean(),

            "case_median_mae_improvement":
                np.median(improvements),

            "case_mean_relative_improvement_pct":
                relative.mean(),

            "case_wins":
                wins,

            "case_losses":
                losses,

            "case_ties":
                ties,

            "case_win_rate":
                wins / len(local),

            "positive_family_means":
                positive_families,

            "negative_family_means":
                negative_families,

            "family_balanced_mae_improvement":
                observed_family_balanced,

            "family_balanced_relative_improvement_pct":
                observed_relative_family_balanced,

            "cluster_bootstrap_ci95_low":
                ci_low,

            "cluster_bootstrap_ci95_high":
                ci_high,

            "bootstrap_nonpositive_rate":
                bootstrap_nonpositive_rate,
        }
    )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# Print
# ============================================================

print("=" * 110)
print("PER-FAMILY EFFECTS")
print("=" * 110)

print(
    family_effects.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n" + "=" * 130)
print("FAMILY-HOLDOUT SIGNIFICANCE / ROBUSTNESS SUMMARY")
print("=" * 130)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print(
    "\nPositive improvement means stratified sampling "
    "has lower MAE than random sampling."
)

print(
    "The primary bootstrap resamples language pairs as clusters "
    "and gives each model family equal weight."
)

print("\nSaved:")
print(FAMILY_FILE)
print(SUMMARY_FILE)
