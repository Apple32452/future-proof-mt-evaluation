#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


# ============================================================
# Configuration
# ============================================================

LANGUAGE_PAIRS = [
    "cs-de_DE",
    "en-cs_CZ",
    "cs-uk_UA",
    "en-sr_Cyrl_RS",
    "en-is_IS",
    "en-ru_RU",
    "en-ar_EG",
    "en-uk_UA",
    "en-zh_CN",
    "en-et_EE",
    "en-bho_IN",
    "en-ja_JP",
    "en-mas_KE",
    "en-it_IT",
]

BUDGETS = [10, 25, 50, 100]

N_SYSTEM_SPLITS = 100
N_SENTENCE_TRIALS = 30

N_TEST_SYSTEMS = 6
N_BINS = 5

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

OUTPUT_DIR = (
    ROOT
    / "results"
    / "multilanguage_benchmark"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DETAIL_FILE = (
    OUTPUT_DIR
    / "multilanguage_detail.csv"
)

PAIR_SUMMARY_FILE = (
    OUTPUT_DIR
    / "multilanguage_pair_summary.csv"
)

MACRO_SUMMARY_FILE = (
    OUTPUT_DIR
    / "multilanguage_macro_summary.csv"
)


# ============================================================
# Metrics
# ============================================================

def pairwise_accuracy(reference, estimated):

    ref = reference.to_numpy()
    est = estimated.to_numpy()

    correct = 0
    total = 0

    for i in range(len(ref)):
        for j in range(i + 1, len(ref)):

            dr = ref[i] - ref[j]
            de = est[i] - est[j]

            if dr == 0 or de == 0:
                continue

            total += 1

            if np.sign(dr) == np.sign(de):
                correct += 1

    return correct / total if total else np.nan


def evaluate(test_scores, full_scores, indices):

    subset = (
        test_scores
        .iloc[indices]
        .mean(axis=0)
    )

    return {
        "mae": np.mean(
            np.abs(
                subset.to_numpy()
                - full_scores.to_numpy()
            )
        ),

        "spearman": spearmanr(
            full_scores,
            subset,
        ).statistic,

        "kendall": kendalltau(
            full_scores,
            subset,
        ).statistic,

        "pairwise": pairwise_accuracy(
            full_scores,
            subset,
        ),
    }


# ============================================================
# Load WMT human evaluations
# ============================================================

print("=" * 90)
print("MULTI-LANGUAGE UNSEEN-SYSTEM BENCHMARK")
print("=" * 90)

pair_records = {
    pair: []
    for pair in LANGUAGE_PAIRS
}

with open(WMT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        row = json.loads(line)

        pair = (
            row["doc_id"]
            .split("_#_", 1)[0]
        )

        if pair not in pair_records:
            continue

        record = {
            "sentence_id": row["doc_id"],
            "source_text": row["src_text"],
        }

        for system, annotations in row["scores"].items():

            # refA is the human reference, not an MT system.
            if system == "refA":
                continue

            values = [
                x["score"]
                for x in annotations
                if x.get("score") is not None
            ]

            if values:
                record[system] = np.mean(values)

        pair_records[pair].append(record)


# ============================================================
# Main experiment
# ============================================================

all_rows = []

for pair_i, pair in enumerate(LANGUAGE_PAIRS):

    data = pd.DataFrame(
        pair_records[pair]
    )

    metadata = [
        "sentence_id",
        "source_text",
    ]

    systems = sorted(
        c
        for c in data.columns
        if c not in metadata
    )

    # Keep only systems complete over this language pair.
    complete_systems = [
        system
        for system in systems
        if data[system].notna().all()
    ]

    data = data[
        metadata + complete_systems
    ]

    N = len(data)
    K = len(complete_systems)

    print(
        f"\n[{pair_i + 1}/{len(LANGUAGE_PAIRS)}] "
        f"{pair}: "
        f"N={N}, systems={K}"
    )

    if K <= N_TEST_SYSTEMS:
        print("  SKIPPED: insufficient systems")
        continue

    if min(BUDGETS) > N:
        print("  SKIPPED: insufficient sentences")
        continue

    # --------------------------------------------------------
    # Generate unique held-out system sets
    # --------------------------------------------------------

    rng_split = np.random.default_rng(
        BASE_SEED + pair_i * 100000
    )

    test_sets = set()

    while len(test_sets) < N_SYSTEM_SPLITS:

        held_out = tuple(
            sorted(
                rng_split.choice(
                    complete_systems,
                    size=N_TEST_SYSTEMS,
                    replace=False,
                ).tolist()
            )
        )

        test_sets.add(held_out)

    test_sets = sorted(test_sets)

    # --------------------------------------------------------
    # System splits
    # --------------------------------------------------------

    for split_i, held_out_tuple in enumerate(test_sets):

        test_systems = list(
            held_out_tuple
        )

        dev_systems = [
            s
            for s in complete_systems
            if s not in test_systems
        ]

        dev_scores = data[dev_systems]
        test_scores = data[test_systems]

        full_scores = (
            test_scores.mean(axis=0)
        )

        # Development systems ONLY
        dev_mean = (
            dev_scores.mean(axis=1)
        )

        difficulty_bin = pd.qcut(
            dev_mean,
            q=N_BINS,
            labels=False,
            duplicates="drop",
        )

        bins = sorted(
            difficulty_bin
            .dropna()
            .unique()
        )

        # ====================================================
        # Budgets
        # ====================================================

        for budget in BUDGETS:

            if budget > N:
                continue

            # ================================================
            # RANDOM
            # ================================================

            random_metrics = []

            for trial in range(N_SENTENCE_TRIALS):

                rng = np.random.default_rng(
                    10_000_000
                    + pair_i * 1_000_000
                    + split_i * 10_000
                    + budget * 100
                    + trial
                )

                indices = rng.choice(
                    N,
                    size=budget,
                    replace=False,
                )

                random_metrics.append(
                    evaluate(
                        test_scores,
                        full_scores,
                        indices,
                    )
                )

            random_df = pd.DataFrame(
                random_metrics
            )

            all_rows.append(
                {
                    "language_pair": pair,
                    "split": split_i,
                    "budget": budget,
                    "method": "random",

                    "mae":
                        random_df["mae"].mean(),

                    "spearman":
                        random_df["spearman"].mean(),

                    "kendall":
                        random_df["kendall"].mean(),

                    "pairwise":
                        random_df["pairwise"].mean(),
                }
            )

            # ================================================
            # STRATIFIED RANDOM
            # ================================================

            strat_metrics = []

            for trial in range(N_SENTENCE_TRIALS):

                rng = np.random.default_rng(
                    20_000_000
                    + pair_i * 1_000_000
                    + split_i * 10_000
                    + budget * 100
                    + trial
                )

                base = budget // len(bins)
                remainder = budget % len(bins)

                selected = []

                for pos, bin_id in enumerate(bins):

                    candidates = (
                        data.index[
                            difficulty_bin == bin_id
                        ]
                        .to_numpy()
                    )

                    n_select = base

                    if pos < remainder:
                        n_select += 1

                    chosen = rng.choice(
                        candidates,
                        size=n_select,
                        replace=False,
                    )

                    selected.extend(
                        chosen.tolist()
                    )

                selected = np.asarray(
                    selected,
                    dtype=int,
                )

                strat_metrics.append(
                    evaluate(
                        test_scores,
                        full_scores,
                        selected,
                    )
                )

            strat_df = pd.DataFrame(
                strat_metrics
            )

            all_rows.append(
                {
                    "language_pair": pair,
                    "split": split_i,
                    "budget": budget,
                    "method": "stratified_random",

                    "mae":
                        strat_df["mae"].mean(),

                    "spearman":
                        strat_df["spearman"].mean(),

                    "kendall":
                        strat_df["kendall"].mean(),

                    "pairwise":
                        strat_df["pairwise"].mean(),
                }
            )

        if (split_i + 1) % 20 == 0:
            print(
                f"  completed "
                f"{split_i + 1}/"
                f"{N_SYSTEM_SPLITS} splits"
            )


# ============================================================
# Save detailed results
# ============================================================

detail = pd.DataFrame(
    all_rows
)

detail.to_csv(
    DETAIL_FILE,
    index=False,
)


# ============================================================
# Per-language summary
# ============================================================

pair_summary = (
    detail
    .groupby(
        [
            "language_pair",
            "method",
            "budget",
        ]
    )
    .agg(
        mae_mean=("mae", "mean"),
        spearman_mean=("spearman", "mean"),
        kendall_mean=("kendall", "mean"),
        pairwise_mean=("pairwise", "mean"),
    )
    .reset_index()
)

pair_summary.to_csv(
    PAIR_SUMMARY_FILE,
    index=False,
)


# ============================================================
# Macro average across language pairs
#
# Important:
# Each language pair gets equal weight here.
# ============================================================

macro = (
    pair_summary
    .groupby(
        [
            "method",
            "budget",
        ]
    )
    .agg(
        mae_mean=("mae_mean", "mean"),
        mae_std=("mae_mean", "std"),

        spearman_mean=("spearman_mean", "mean"),
        spearman_std=("spearman_mean", "std"),

        kendall_mean=("kendall_mean", "mean"),
        kendall_std=("kendall_mean", "std"),

        pairwise_mean=("pairwise_mean", "mean"),
        pairwise_std=("pairwise_mean", "std"),
    )
    .reset_index()
)

macro.to_csv(
    MACRO_SUMMARY_FILE,
    index=False,
)


# ============================================================
# Print macro results
# ============================================================

print("\n" + "=" * 100)
print("MACRO AVERAGE ACROSS LANGUAGE PAIRS")
print("=" * 100)

print(
    macro.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nSaved:")
print(DETAIL_FILE)
print(PAIR_SUMMARY_FILE)
print(MACRO_SUMMARY_FILE)
