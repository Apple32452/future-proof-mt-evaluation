#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FIGURE_DIR = ROOT / "figures" / "paper"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Figure 1
# Multi-language random-system holdout
# ============================================================

multi_file = (
    ROOT
    / "results"
    / "multilanguage_benchmark"
    / "multilanguage_macro_summary.csv"
)

multi = pd.read_csv(multi_file)

plt.figure(figsize=(6.5, 4.5))

for method, label in [
    ("random", "Random"),
    ("stratified_random", "Difficulty-stratified"),
]:
    local = multi[
        multi["method"] == method
    ].sort_values("budget")

    plt.plot(
        local["budget"],
        local["mae_mean"],
        marker="o",
        label=label,
    )

plt.xlabel("Evaluation budget (sentences)")
plt.ylabel("MAE on unseen MT systems")
plt.title("Unseen-System Evaluation Across 14 Language Pairs")
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "fig1_multilanguage_mae.pdf"
)

plt.savefig(
    FIGURE_DIR / "fig1_multilanguage_mae.png",
    dpi=300,
)

plt.close()


# ============================================================
# Figure 2
# Per-language improvement at budget 25
# ============================================================

difference_file = (
    ROOT
    / "results"
    / "multilanguage_significance"
    / "per_language_differences.csv"
)

differences = pd.read_csv(
    difference_file
)

budget25 = (
    differences[
        differences["budget"] == 25
    ]
    .sort_values(
        "relative_mae_improvement_pct"
    )
)

plt.figure(figsize=(7.0, 5.2))

plt.barh(
    budget25["language_pair"],
    budget25["relative_mae_improvement_pct"],
)

plt.axvline(
    0,
    linewidth=1,
)

plt.xlabel(
    "Relative MAE reduction (%)"
)

plt.ylabel(
    "Language pair"
)

plt.title(
    "Difficulty-Stratified Sampling at Budget 25"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "fig2_per_language_improvement_m25.pdf"
)

plt.savefig(
    FIGURE_DIR
    / "fig2_per_language_improvement_m25.png",
    dpi=300,
)

plt.close()


# ============================================================
# Figure 3
# Entire model-family holdout
# ============================================================

family_file = (
    ROOT
    / "results"
    / "family_holdout_benchmark"
    / "family_holdout_macro_summary.csv"
)

family = pd.read_csv(
    family_file
).sort_values("budget")

plt.figure(figsize=(6.5, 4.5))

plt.plot(
    family["budget"],
    family["random_mae"],
    marker="o",
    label="Random",
)

plt.plot(
    family["budget"],
    family["stratified_mae"],
    marker="o",
    label="Difficulty-stratified",
)

plt.xlabel(
    "Evaluation budget (sentences)"
)

plt.ylabel(
    "Family-balanced MAE"
)

plt.title(
    "Transfer to Completely Unseen Model Families"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "fig3_family_holdout_mae.pdf"
)

plt.savefig(
    FIGURE_DIR
    / "fig3_family_holdout_mae.png",
    dpi=300,
)

plt.close()


# ============================================================
# Figure 4
# Difficulty-bin robustness
# ============================================================

robust_file = (
    ROOT
    / "results"
    / "family_holdout_bin_robustness"
    / "bin_robustness_family_balanced_summary.csv"
)

robust = pd.read_csv(
    robust_file
)

plt.figure(figsize=(6.5, 4.5))

for n_bins in sorted(
    robust["num_bins"].unique()
):

    local = robust[
        robust["num_bins"] == n_bins
    ].sort_values("budget")

    plt.plot(
        local["budget"],
        local["mean_relative_improvement_pct"],
        marker="o",
        label=f"{n_bins} bins",
    )

plt.axhline(
    0,
    linewidth=1,
)

plt.xlabel(
    "Evaluation budget (sentences)"
)

plt.ylabel(
    "Relative MAE reduction (%)"
)

plt.title(
    "Robustness to Number of Difficulty Strata"
)

plt.legend(
    title="Difficulty strata"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "fig4_bin_robustness.pdf"
)

plt.savefig(
    FIGURE_DIR
    / "fig4_bin_robustness.png",
    dpi=300,
)

plt.close()


print("Saved paper figures to:")
print(FIGURE_DIR)
