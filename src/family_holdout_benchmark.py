#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


# ============================================================
# Configuration
# ============================================================

BUDGETS = [10, 25, 50, 100]

N_BINS = 5
N_SENTENCE_TRIALS = 100
BASE_SEED = 2026


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

WMT_FILE = (
    ROOT.parent
    / "wmt25-general-mt"
    / "data"
    / "wmt25-genmt-humeval.jsonl"
)

AUDIT_FILE = (
    ROOT
    / "results"
    / "family_holdout"
    / "family_holdout_audit.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "family_holdout_benchmark"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DETAIL_FILE = (
    OUTPUT_DIR
    / "family_holdout_detail.csv"
)

CASE_SUMMARY_FILE = (
    OUTPUT_DIR
    / "family_holdout_case_summary.csv"
)

FAMILY_SUMMARY_FILE = (
    OUTPUT_DIR
    / "family_holdout_family_summary.csv"
)

MACRO_SUMMARY_FILE = (
    OUTPUT_DIR
    / "family_holdout_macro_summary.csv"
)


# ============================================================
# Metrics
# ============================================================

def pairwise_accuracy(reference_scores, estimated_scores):

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
    heldout_scores,
    full_scores,
    selected_indices,
):

    subset_scores = (
        heldout_scores
        .iloc[selected_indices]
        .mean(axis=0)
    )

    mae = np.mean(
        np.abs(
            subset_scores.to_numpy()
            - full_scores.to_numpy()
        )
    )

    # Ranking metrics become very coarse with only 2 systems,
    # but we save them as secondary diagnostics.
    if len(full_scores) >= 2:

        spearman = spearmanr(
            full_scores,
            subset_scores,
        ).statistic

        kendall = kendalltau(
            full_scores,
            subset_scores,
        ).statistic

        pairwise = pairwise_accuracy(
            full_scores,
            subset_scores,
        )

    else:
        spearman = np.nan
        kendall = np.nan
        pairwise = np.nan

    return {
        "mae": mae,
        "spearman": spearman,
        "kendall": kendall,
        "pairwise": pairwise,
    }


# ============================================================
# Load eligible family holdouts
# ============================================================

audit = pd.read_csv(AUDIT_FILE)

audit = audit[
    audit["eligible"] == True
].copy()

print("=" * 100)
print("MODEL-FAMILY HOLDOUT BENCHMARK")
print("=" * 100)

print(f"\nEligible family-language cases: {len(audit)}")
print(f"Budgets: {BUDGETS}")
print(f"Sentence trials per method: {N_SENTENCE_TRIALS}")


# ============================================================
# Load WMT data once
# ============================================================

needed_pairs = set(
    audit["language_pair"]
)

records_by_pair = {
    pair: []
    for pair in needed_pairs
}

with open(WMT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        row = json.loads(line)

        pair = (
            row["doc_id"]
            .split("_#_", 1)[0]
        )

        if pair not in needed_pairs:
            continue

        record = {
            "sentence_id": row["doc_id"],
            "source_text": row["src_text"],
        }

        for system, annotations in row["scores"].items():

            if system == "refA":
                continue

            values = [
                annotation["score"]
                for annotation in annotations
                if annotation.get("score") is not None
            ]

            if values:
                record[system] = np.mean(values)

        records_by_pair[pair].append(record)


# ============================================================
# Main experiment
# ============================================================

detail_rows = []

for case_index, audit_row in audit.reset_index(drop=True).iterrows():

    pair = audit_row["language_pair"]
    family = audit_row["family"]

    heldout_systems = [
        x.strip()
        for x in audit_row["systems"].split(",")
        if x.strip()
    ]

    data = pd.DataFrame(
        records_by_pair[pair]
    )

    metadata_columns = [
        "sentence_id",
        "source_text",
    ]

    system_columns = [
        c
        for c in data.columns
        if c not in metadata_columns
    ]

    # Keep only fully observed systems for safety.
    complete_systems = [
        system
        for system in system_columns
        if data[system].notna().all()
    ]

    heldout_systems = [
        system
        for system in heldout_systems
        if system in complete_systems
    ]

    dev_systems = [
        system
        for system in complete_systems
        if system not in heldout_systems
    ]

    N = len(data)

    if len(heldout_systems) < 2:
        print(
            f"SKIPPING {pair} / {family}: "
            "fewer than 2 complete held-out systems"
        )
        continue

    if len(dev_systems) < 5:
        print(
            f"SKIPPING {pair} / {family}: "
            "fewer than 5 development systems"
        )
        continue

    print(
        f"\n[{case_index + 1}/{len(audit)}] "
        f"{pair} / {family}: "
        f"N={N}, "
        f"dev={len(dev_systems)}, "
        f"heldout={len(heldout_systems)}"
    )

    dev_scores = data[dev_systems]
    heldout_scores = data[heldout_systems]

    full_scores = (
        heldout_scores.mean(axis=0)
    )

    # --------------------------------------------------------
    # Development-only difficulty estimate
    # --------------------------------------------------------

    dev_mean_score = (
        dev_scores.mean(axis=1)
    )

    difficulty_bin = pd.qcut(
        dev_mean_score,
        q=N_BINS,
        labels=False,
        duplicates="drop",
    )

    bins = sorted(
        difficulty_bin
        .dropna()
        .unique()
    )

    # --------------------------------------------------------
    # Run budgets
    # --------------------------------------------------------

    for budget in BUDGETS:

        if budget > N:
            continue

        random_trial_metrics = []
        stratified_trial_metrics = []

        for trial in range(N_SENTENCE_TRIALS):

            # =================================================
            # Ordinary random
            # =================================================

            rng_random = np.random.default_rng(
                BASE_SEED
                + case_index * 1_000_000
                + budget * 10_000
                + trial
            )

            random_indices = (
                rng_random.choice(
                    N,
                    size=budget,
                    replace=False,
                )
            )

            random_trial_metrics.append(
                evaluate_subset(
                    heldout_scores,
                    full_scores,
                    random_indices,
                )
            )

            # =================================================
            # Difficulty-stratified random
            # =================================================

            rng_stratified = np.random.default_rng(
                BASE_SEED
                + 100_000_000
                + case_index * 1_000_000
                + budget * 10_000
                + trial
            )

            base = (
                budget // len(bins)
            )

            remainder = (
                budget % len(bins)
            )

            selected = []

            for position, bin_id in enumerate(bins):

                candidates = (
                    data.index[
                        difficulty_bin == bin_id
                    ]
                    .to_numpy()
                )

                n_select = base

                if position < remainder:
                    n_select += 1

                chosen = (
                    rng_stratified.choice(
                        candidates,
                        size=n_select,
                        replace=False,
                    )
                )

                selected.extend(
                    chosen.tolist()
                )

            stratified_indices = np.asarray(
                selected,
                dtype=int,
            )

            stratified_trial_metrics.append(
                evaluate_subset(
                    heldout_scores,
                    full_scores,
                    stratified_indices,
                )
            )

        # =====================================================
        # Average stochastic sentence trials within this case
        # =====================================================

        for method, values in [
            ("random", random_trial_metrics),
            ("stratified_random", stratified_trial_metrics),
        ]:

            trial_df = pd.DataFrame(values)

            detail_rows.append(
                {
                    "language_pair": pair,
                    "family": family,
                    "budget": budget,
                    "method": method,

                    "num_sentences": N,
                    "num_dev_systems":
                        len(dev_systems),

                    "num_heldout_systems":
                        len(heldout_systems),

                    "heldout_system_names":
                        ", ".join(heldout_systems),

                    "mae":
                        trial_df["mae"].mean(),

                    "spearman":
                        trial_df["spearman"].mean(),

                    "kendall":
                        trial_df["kendall"].mean(),

                    "pairwise":
                        trial_df["pairwise"].mean(),
                }
            )


# ============================================================
# Save case-level detail
# ============================================================

detail = pd.DataFrame(
    detail_rows
)

detail.to_csv(
    DETAIL_FILE,
    index=False,
)


# ============================================================
# Pair Random vs Stratified within each family-language case
# ============================================================

random_results = (
    detail[
        detail["method"] == "random"
    ]
    .rename(
        columns={
            "mae": "random_mae",
            "spearman": "random_spearman",
            "kendall": "random_kendall",
            "pairwise": "random_pairwise",
        }
    )
)

strat_results = (
    detail[
        detail["method"] == "stratified_random"
    ]
    .rename(
        columns={
            "mae": "stratified_mae",
            "spearman": "stratified_spearman",
            "kendall": "stratified_kendall",
            "pairwise": "stratified_pairwise",
        }
    )
)

case_summary = random_results.merge(
    strat_results[
        [
            "language_pair",
            "family",
            "budget",
            "stratified_mae",
            "stratified_spearman",
            "stratified_kendall",
            "stratified_pairwise",
        ]
    ],
    on=[
        "language_pair",
        "family",
        "budget",
    ],
)

case_summary["mae_improvement"] = (
    case_summary["random_mae"]
    - case_summary["stratified_mae"]
)

case_summary["relative_mae_improvement_pct"] = (
    100.0
    * case_summary["mae_improvement"]
    / case_summary["random_mae"]
)

case_summary["stratified_better_mae"] = (
    case_summary["mae_improvement"] > 0
)

case_summary.to_csv(
    CASE_SUMMARY_FILE,
    index=False,
)


# ============================================================
# Family-level summary
#
# Average language-pair cases WITHIN each family first.
# ============================================================

family_summary = (
    case_summary
    .groupby(
        [
            "family",
            "budget",
        ]
    )
    .agg(
        num_cases=(
            "language_pair",
            "count",
        ),

        random_mae=(
            "random_mae",
            "mean",
        ),

        stratified_mae=(
            "stratified_mae",
            "mean",
        ),

        mae_improvement=(
            "mae_improvement",
            "mean",
        ),

        relative_improvement_pct=(
            "relative_mae_improvement_pct",
            "mean",
        ),

        win_rate=(
            "stratified_better_mae",
            "mean",
        ),

        random_kendall=(
            "random_kendall",
            "mean",
        ),

        stratified_kendall=(
            "stratified_kendall",
            "mean",
        ),
    )
    .reset_index()
)

family_summary.to_csv(
    FAMILY_SUMMARY_FILE,
    index=False,
)


# ============================================================
# FAMILY-BALANCED macro average
#
# Each family gets equal weight, instead of Gemma receiving
# 13/32 of the overall weight.
# ============================================================

macro_summary = (
    family_summary
    .groupby("budget")
    .agg(
        num_families=(
            "family",
            "nunique",
        ),

        random_mae=(
            "random_mae",
            "mean",
        ),

        stratified_mae=(
            "stratified_mae",
            "mean",
        ),

        mean_mae_improvement=(
            "mae_improvement",
            "mean",
        ),

        mean_relative_improvement_pct=(
            "relative_improvement_pct",
            "mean",
        ),

        mean_family_win_rate=(
            "win_rate",
            "mean",
        ),

        random_kendall=(
            "random_kendall",
            "mean",
        ),

        stratified_kendall=(
            "stratified_kendall",
            "mean",
        ),
    )
    .reset_index()
)

macro_summary.to_csv(
    MACRO_SUMMARY_FILE,
    index=False,
)


# ============================================================
# Print
# ============================================================

print("\n" + "=" * 110)
print("FAMILY-LEVEL SUMMARY")
print("=" * 110)

print(
    family_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\n" + "=" * 110)
print("FAMILY-BALANCED MACRO SUMMARY")
print("=" * 110)

print(
    macro_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nPrimary endpoint: MAE.")
print(
    "Ranking metrics are secondary because some holdouts "
    "contain only two systems."
)

print("\nSaved:")
print(DETAIL_FILE)
print(CASE_SUMMARY_FILE)
print(FAMILY_SUMMARY_FILE)
print(MACRO_SUMMARY_FILE)
