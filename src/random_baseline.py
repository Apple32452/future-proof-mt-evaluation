#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


# ============================================================
# Configuration
# ============================================================

LANGUAGE_PAIR = "cs-de_DE"

BUDGETS = [25, 50, 100]

# Repeat random selection many times so that the baseline
# is not dependent on one lucky random sample.
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

OUTPUT_DIR = ROOT / "results" / "random_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIAL_FILE = OUTPUT_DIR / f"{LANGUAGE_PAIR}_random_trials.csv"
SUMMARY_FILE = OUTPUT_DIR / f"{LANGUAGE_PAIR}_random_summary.csv"


# ============================================================
# Load score matrix
# ============================================================

df = pd.read_csv(MATRIX_FILE)

metadata_columns = [
    "sentence_id",
    "source_text",
]

system_columns = [
    col
    for col in df.columns
    if col not in metadata_columns
]

scores = df[system_columns]

N = len(scores)
K = len(system_columns)

print("=" * 80)
print("RANDOM SUBSET BASELINE")
print("=" * 80)

print(f"\nLanguage pair: {LANGUAGE_PAIR}")
print(f"Sentences:     {N}")
print(f"Systems:       {K}")
print(f"Trials:        {N_TRIALS}")
print(f"Budgets:       {BUDGETS}")


# ============================================================
# Full-test-set reference scores
# ============================================================

full_scores = scores.mean(axis=0)


# ============================================================
# Pairwise ranking accuracy
# ============================================================

def pairwise_ranking_accuracy(
    reference_scores,
    estimated_scores,
):
    """
    Fraction of system pairs whose relative ordering is
    preserved by the subset.

    Ties in either vector are skipped.
    """

    correct = 0
    total = 0

    values_ref = reference_scores.to_numpy()
    values_est = estimated_scores.to_numpy()

    for i in range(len(values_ref)):
        for j in range(i + 1, len(values_ref)):

            ref_diff = values_ref[i] - values_ref[j]
            est_diff = values_est[i] - values_est[j]

            # Ignore exact ties.
            if ref_diff == 0 or est_diff == 0:
                continue

            total += 1

            if np.sign(ref_diff) == np.sign(est_diff):
                correct += 1

    if total == 0:
        return np.nan

    return correct / total


# ============================================================
# Run experiments
# ============================================================

trial_results = []

for budget in BUDGETS:

    if budget > N:
        raise ValueError(
            f"Budget {budget} exceeds dataset size {N}"
        )

    print(f"\nRunning budget m={budget}...")

    for trial in range(N_TRIALS):

        rng = np.random.default_rng(
            BASE_SEED + budget * 10000 + trial
        )

        selected_indices = rng.choice(
            N,
            size=budget,
            replace=False,
        )

        subset_scores = (
            scores
            .iloc[selected_indices]
            .mean(axis=0)
        )

        # ----------------------------------------------------
        # Score preservation
        # ----------------------------------------------------

        mae = np.mean(
            np.abs(
                subset_scores.to_numpy()
                - full_scores.to_numpy()
            )
        )

        # ----------------------------------------------------
        # Ranking preservation
        # ----------------------------------------------------

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
# Save trial-level results
# ============================================================

trial_df = pd.DataFrame(trial_results)

trial_df.to_csv(
    TRIAL_FILE,
    index=False,
)


# ============================================================
# Summarize random baseline
# ============================================================

summary_rows = []

for budget in BUDGETS:

    local = trial_df[
        trial_df["budget"] == budget
    ]

    row = {
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

    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# Print summary
# ============================================================

print("\n" + "=" * 80)
print("RANDOM BASELINE SUMMARY")
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
print("  Higher pairwise ranking accuracy is better.")

print("\nSaved trial results:")
print(TRIAL_FILE)

print("\nSaved summary:")
print(SUMMARY_FILE)
