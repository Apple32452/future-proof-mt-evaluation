#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# Configuration
# ============================================================

N_BOOTSTRAP = 20000
BOOTSTRAP_SEED = 42


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results"
    / "multilanguage_benchmark"
    / "multilanguage_pair_summary.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "multilanguage_significance"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DELTA_FILE = (
    OUTPUT_DIR
    / "per_language_differences.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "paired_significance_summary.csv"
)


# ============================================================
# Load results
# ============================================================

df = pd.read_csv(INPUT_FILE)

random_df = (
    df[df["method"] == "random"]
    [
        [
            "language_pair",
            "budget",
            "mae_mean",
            "spearman_mean",
            "kendall_mean",
            "pairwise_mean",
        ]
    ]
    .rename(
        columns={
            "mae_mean": "random_mae",
            "spearman_mean": "random_spearman",
            "kendall_mean": "random_kendall",
            "pairwise_mean": "random_pairwise",
        }
    )
)

strat_df = (
    df[df["method"] == "stratified_random"]
    [
        [
            "language_pair",
            "budget",
            "mae_mean",
            "spearman_mean",
            "kendall_mean",
            "pairwise_mean",
        ]
    ]
    .rename(
        columns={
            "mae_mean": "stratified_mae",
            "spearman_mean": "stratified_spearman",
            "kendall_mean": "stratified_kendall",
            "pairwise_mean": "stratified_pairwise",
        }
    )
)


# ============================================================
# Pair methods
# ============================================================

paired = random_df.merge(
    strat_df,
    on=[
        "language_pair",
        "budget",
    ],
    how="inner",
)


# Positive = stratified is better for MAE
paired["mae_improvement"] = (
    paired["random_mae"]
    - paired["stratified_mae"]
)

paired["relative_mae_improvement_pct"] = (
    100.0
    * paired["mae_improvement"]
    / paired["random_mae"]
)

# Positive = stratified is better
paired["spearman_change"] = (
    paired["stratified_spearman"]
    - paired["random_spearman"]
)

paired["kendall_change"] = (
    paired["stratified_kendall"]
    - paired["random_kendall"]
)

paired["pairwise_change"] = (
    paired["stratified_pairwise"]
    - paired["random_pairwise"]
)


paired.to_csv(
    DELTA_FILE,
    index=False,
)


# ============================================================
# Bootstrap helper
# ============================================================

def bootstrap_mean_ci(
    values,
    rng,
    n_bootstrap=N_BOOTSTRAP,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    boot_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(n_bootstrap):

        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        boot_means[i] = sample.mean()

    lower = np.quantile(
        boot_means,
        0.025,
    )

    upper = np.quantile(
        boot_means,
        0.975,
    )

    return lower, upper


# ============================================================
# Paired analysis across LANGUAGE PAIRS
# ============================================================

summary_rows = []

for budget in sorted(
    paired["budget"].unique()
):

    local = paired[
        paired["budget"] == budget
    ].copy()

    random_mae = (
        local["random_mae"]
        .to_numpy()
    )

    strat_mae = (
        local["stratified_mae"]
        .to_numpy()
    )

    improvements = (
        local["mae_improvement"]
        .to_numpy()
    )

    relative_improvements = (
        local[
            "relative_mae_improvement_pct"
        ]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Win rate
    # --------------------------------------------------------

    wins = int(
        np.sum(improvements > 0)
    )

    losses = int(
        np.sum(improvements < 0)
    )

    ties = int(
        np.sum(improvements == 0)
    )

    win_rate = (
        wins / len(local)
    )

    # --------------------------------------------------------
    # Paired Wilcoxon
    #
    # H1:
    # random MAE > stratified MAE
    # --------------------------------------------------------

    test = wilcoxon(
        random_mae,
        strat_mae,
        alternative="greater",
    )

    # --------------------------------------------------------
    # Bootstrap CI across LANGUAGE PAIRS
    # --------------------------------------------------------

    rng = np.random.default_rng(
        BOOTSTRAP_SEED + int(budget)
    )

    ci_low, ci_high = (
        bootstrap_mean_ci(
            improvements,
            rng,
        )
    )

    rel_low, rel_high = (
        bootstrap_mean_ci(
            relative_improvements,
            rng,
        )
    )

    summary_rows.append(
        {
            "budget": budget,
            "num_languages":
                len(local),

            "random_mae_mean":
                random_mae.mean(),

            "stratified_mae_mean":
                strat_mae.mean(),

            "mean_mae_improvement":
                improvements.mean(),

            "median_mae_improvement":
                np.median(improvements),

            "mean_relative_improvement_pct":
                relative_improvements.mean(),

            "mae_improvement_ci95_low":
                ci_low,

            "mae_improvement_ci95_high":
                ci_high,

            "relative_improvement_ci95_low":
                rel_low,

            "relative_improvement_ci95_high":
                rel_high,

            "languages_stratified_better":
                wins,

            "languages_random_better":
                losses,

            "ties":
                ties,

            "stratified_win_rate":
                win_rate,

            "wilcoxon_statistic":
                test.statistic,

            "wilcoxon_pvalue":
                test.pvalue,

            "mean_spearman_change":
                local[
                    "spearman_change"
                ].mean(),

            "mean_kendall_change":
                local[
                    "kendall_change"
                ].mean(),

            "mean_pairwise_change":
                local[
                    "pairwise_change"
                ].mean(),
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
# Print per-language MAE effects
# ============================================================

for budget in sorted(
    paired["budget"].unique()
):

    local = (
        paired[
            paired["budget"] == budget
        ]
        .sort_values(
            "relative_mae_improvement_pct",
            ascending=False,
        )
    )

    print("\n" + "=" * 90)
    print(
        f"PER-LANGUAGE MAE RESULTS — BUDGET {budget}"
    )
    print("=" * 90)

    print(
        local[
            [
                "language_pair",
                "random_mae",
                "stratified_mae",
                "mae_improvement",
                "relative_mae_improvement_pct",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================
# Print statistical summary
# ============================================================

print("\n" + "=" * 120)
print("PAIRED CROSS-LANGUAGE SIGNIFICANCE SUMMARY")
print("=" * 120)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\nPositive MAE improvement means stratified sampling is better.")
print(
    "The bootstrap interval is over language pairs, "
    "not over the overlapping system splits."
)

print("\nSaved:")
print(DELTA_FILE)
print(SUMMARY_FILE)
