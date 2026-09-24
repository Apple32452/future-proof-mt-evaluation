#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

N_BINS_LIST = [2, 3, 5, 10]
BUDGETS = [10, 25, 50, 100]

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
    / "family_holdout_bin_robustness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DETAIL_FILE = (
    OUTPUT_DIR
    / "bin_robustness_detail.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "bin_robustness_family_balanced_summary.csv"
)


# ============================================================
# Load eligible cases
# ============================================================

audit = pd.read_csv(AUDIT_FILE)

audit = audit[
    audit["eligible"] == True
].reset_index(drop=True)

needed_pairs = set(
    audit["language_pair"]
)


# ============================================================
# Load WMT data
# ============================================================

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

rows = []

print("=" * 100)
print("FAMILY-HOLDOUT DIFFICULTY-BIN ROBUSTNESS")
print("=" * 100)

print(f"\nCases:   {len(audit)}")
print(f"Bins:    {N_BINS_LIST}")
print(f"Budgets: {BUDGETS}")


for case_index, audit_row in audit.iterrows():

    pair = audit_row["language_pair"]
    family = audit_row["family"]

    heldout_systems = [
        item.strip()
        for item in audit_row["systems"].split(",")
        if item.strip()
    ]

    data = pd.DataFrame(
        records_by_pair[pair]
    )

    metadata = [
        "sentence_id",
        "source_text",
    ]

    systems = [
        column
        for column in data.columns
        if column not in metadata
    ]

    complete_systems = [
        system
        for system in systems
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

    if len(heldout_systems) < 2:
        continue

    dev_scores = data[dev_systems]
    heldout_scores = data[heldout_systems]

    full_scores = (
        heldout_scores.mean(axis=0)
    )

    dev_mean_score = (
        dev_scores.mean(axis=1)
    )

    N = len(data)

    print(
        f"\n[{case_index + 1}/{len(audit)}] "
        f"{pair} / {family}"
    )

    # ========================================================
    # Random baseline ONCE per case/budget.
    #
    # It does not depend on number of bins.
    # ========================================================

    random_by_budget = {}

    for budget in BUDGETS:

        trial_maes = []

        for trial in range(N_SENTENCE_TRIALS):

            rng = np.random.default_rng(
                BASE_SEED
                + case_index * 1_000_000
                + budget * 10_000
                + trial
            )

            indices = rng.choice(
                N,
                size=budget,
                replace=False,
            )

            subset_scores = (
                heldout_scores
                .iloc[indices]
                .mean(axis=0)
            )

            mae = np.mean(
                np.abs(
                    subset_scores.to_numpy()
                    - full_scores.to_numpy()
                )
            )

            trial_maes.append(mae)

        random_by_budget[budget] = (
            np.mean(trial_maes)
        )

    # ========================================================
    # Stratification with different bin counts
    # ========================================================

    for n_bins in N_BINS_LIST:

        difficulty_bin = pd.qcut(
            dev_mean_score,
            q=n_bins,
            labels=False,
            duplicates="drop",
        )

        bins = sorted(
            difficulty_bin
            .dropna()
            .unique()
        )

        for budget in BUDGETS:

            trial_maes = []
            
            for trial in range(N_SENTENCE_TRIALS):

                rng = np.random.default_rng(
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

                    chosen = rng.choice(
                        candidates,
                        size=n_select,
                        replace=False,
                    )

                    selected.extend(
                        chosen.tolist()
                    )

                subset_scores = (
                    heldout_scores
                    .iloc[selected]
                    .mean(axis=0)
                )

                mae = np.mean(
                    np.abs(
                        subset_scores.to_numpy()
                        - full_scores.to_numpy()
                    )
                )

                trial_maes.append(mae)

            stratified_mae = (
                np.mean(trial_maes)
            )

            random_mae = (
                random_by_budget[budget]
            )

            improvement = (
                random_mae
                - stratified_mae
            )

            relative = (
                100.0
                * improvement
                / random_mae
            )

            rows.append(
                {
                    "language_pair": pair,
                    "family": family,
                    "num_bins": n_bins,
                    "budget": budget,

                    "random_mae":
                        random_mae,

                    "stratified_mae":
                        stratified_mae,

                    "mae_improvement":
                        improvement,

                    "relative_improvement_pct":
                        relative,

                    "stratified_better":
                        improvement > 0,
                }
            )


# ============================================================
# Save detail
# ============================================================

detail = pd.DataFrame(rows)

detail.to_csv(
    DETAIL_FILE,
    index=False,
)


# ============================================================
# First average within family
# ============================================================

family = (
    detail
    .groupby(
        [
            "family",
            "num_bins",
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
            "relative_improvement_pct",
            "mean",
        ),

        win_rate=(
            "stratified_better",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# Family-balanced macro result
# ============================================================

summary = (
    family
    .groupby(
        [
            "num_bins",
            "budget",
        ]
    )
    .agg(
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

print("\n" + "=" * 110)
print("FAMILY-BALANCED BIN ROBUSTNESS")
print("=" * 110)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nIMPORTANT:")
print(
    "5 bins remains the pre-existing primary configuration."
)

print(
    "This experiment tests sensitivity; "
    "it should not be used to select a new best bin count "
    "from the family-holdout test results."
)

print("\nSaved:")
print(DETAIL_FILE)
print(SUMMARY_FILE)
