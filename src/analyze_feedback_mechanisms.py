"""
Feedback analyses for the future-proof MT evaluation paper.

1. Difficulty-transfer analysis
   Does sentence difficulty estimated from development systems
   correlate with difficulty of unseen held-out systems?

2. Language-pair heterogeneity
   Do language pairs with more transferable difficulty obtain
   larger MAE gains from Stratified-Human?

3. Sentence-length diagnostic
   Does difficulty stratification materially change the source
   sentence-length distribution relative to the full test set?

Uses the FINAL multilingual strong-baseline experiment.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr


# ============================================================
# Configuration
# ============================================================

LANGUAGE_PAIRS = [
    "cs-de_DE",
    "cs-uk_UA",
    "en-ar_EG",
    "en-bho_IN",
    "en-cs_CZ",
    "en-et_EE",
    "en-is_IS",
    "en-it_IT",
    "en-ja_JP",
    "en-mas_KE",
    "en-ru_RU",
    "en-sr_Cyrl_RS",
    "en-uk_UA",
    "en-zh_CN",
]

BUDGETS = [10, 25, 50, 100]

N_BINS = 5
N_STOCHASTIC_TRIALS = 30
BASE_SEED = 20260925

METRIC = "MetricX-24-Hybrid-XL"


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "results"
    / "wmt25_subset2evaluate_all"
)

BENCHMARK_DIR = (
    ROOT
    / "results"
    / "zouhar_multilanguage_benchmark"
)

OUT_DIR = (
    ROOT
    / "results"
    / "feedback_analysis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


LANGUAGE_SUMMARY_FILE = (
    BENCHMARK_DIR
    / "multilanguage_language_summary.csv"
)

SYSTEM_SPLITS_FILE = (
    BENCHMARK_DIR
    / "multilanguage_system_splits.json"
)


# ============================================================
# Load benchmark metadata
# ============================================================

language_results = pd.read_csv(
    LANGUAGE_SUMMARY_FILE
)

with open(
    SYSTEM_SPLITS_FILE,
    "r",
    encoding="utf-8",
) as f:

    system_splits = json.load(f)


# ============================================================
# Helpers
# ============================================================

def load_language(language_pair):

    path = (
        DATA_DIR
        / f"{language_pair}_subset2evaluate.jsonl"
    )

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            rows.append(
                json.loads(line)
            )

    return rows


def difficulty_vector(
    data,
    systems,
    signal="human",
):

    values = []

    for item in data:

        if signal == "human":

            quality = np.mean(
                [
                    item["scores"][system]["human"]
                    for system in systems
                ]
            )

        elif signal == "metricx":

            quality = np.mean(
                [
                    item["scores"][system][METRIC]
                    for system in systems
                ]
            )

        else:

            raise ValueError(signal)

        values.append(
            -float(quality)
        )

    return np.asarray(
        values,
        dtype=float,
    )


def make_bins(difficulty):

    order = np.argsort(
        difficulty,
        kind="stable",
    )

    return [
        np.asarray(part, dtype=int)
        for part in np.array_split(
            order,
            N_BINS,
        )
    ]


def stratified_sample(
    bins,
    budget,
    seed,
):

    rng = np.random.default_rng(seed)

    base = budget // len(bins)
    remainder = budget % len(bins)

    allocation = [
        base for _ in bins
    ]

    for i in range(remainder):
        allocation[i] += 1

    selected = []

    for bin_indices, n_take in zip(
        bins,
        allocation,
    ):

        chosen = rng.choice(
            bin_indices,
            size=n_take,
            replace=False,
        )

        selected.extend(
            chosen.tolist()
        )

    return selected


def safe_corr(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if (
        len(x) < 3
        or np.std(x) == 0
        or np.std(y) == 0
    ):

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    pearson = pearsonr(
        x,
        y,
    )

    spear = spearmanr(
        x,
        y,
    )

    return (
        float(pearson.statistic),
        float(pearson.pvalue),
        float(spear.statistic),
        float(spear.pvalue),
    )


# ============================================================
# 1. Difficulty transfer
# ============================================================

transfer_rows = []


for language_index, language_pair in enumerate(
    LANGUAGE_PAIRS
):

    data = load_language(
        language_pair
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"Difficulty transfer: "
        f"{language_pair}"
    )

    print(
        "=" * 100
    )


    for split_info in system_splits[
        language_pair
    ]:

        split_id = int(
            split_info["split_id"]
        )

        dev_systems = list(
            split_info["dev_systems"]
        )

        test_systems = list(
            split_info["test_systems"]
        )


        # ----------------------------------------
        # Human difficulty estimated from dev
        # versus actual difficulty of held-out
        # systems.
        # ----------------------------------------

        dev_difficulty = (
            difficulty_vector(
                data,
                dev_systems,
                signal="human",
            )
        )

        heldout_difficulty = (
            difficulty_vector(
                data,
                test_systems,
                signal="human",
            )
        )


        (
            pearson_r,
            pearson_p,
            spearman_rho,
            spearman_p,
        ) = safe_corr(
            dev_difficulty,
            heldout_difficulty,
        )


        # ----------------------------------------
        # MetricX difficulty vs held-out human
        # difficulty.
        # ----------------------------------------

        metricx_difficulty = (
            difficulty_vector(
                data,
                dev_systems,
                signal="metricx",
            )
        )


        (
            metricx_pearson_r,
            metricx_pearson_p,
            metricx_spearman_rho,
            metricx_spearman_p,
        ) = safe_corr(
            metricx_difficulty,
            heldout_difficulty,
        )


        transfer_rows.append(
            {
                "language_pair":
                    language_pair,

                "split_id":
                    split_id,

                "num_segments":
                    len(data),

                "n_dev":
                    len(dev_systems),

                "n_test":
                    len(test_systems),

                "human_transfer_pearson":
                    pearson_r,

                "human_transfer_pearson_p":
                    pearson_p,

                "human_transfer_spearman":
                    spearman_rho,

                "human_transfer_spearman_p":
                    spearman_p,

                "metricx_transfer_pearson":
                    metricx_pearson_r,

                "metricx_transfer_pearson_p":
                    metricx_pearson_p,

                "metricx_transfer_spearman":
                    metricx_spearman_rho,

                "metricx_transfer_spearman_p":
                    metricx_spearman_p,
            }
        )


transfer = pd.DataFrame(
    transfer_rows
)


transfer.to_csv(
    OUT_DIR
    / "difficulty_transfer_by_split.csv",
    index=False,
)


# ============================================================
# Average difficulty transfer within language
# ============================================================

language_transfer = (
    transfer
    .groupby(
        "language_pair",
        as_index=False,
    )
    .agg(
        num_splits=(
            "split_id",
            "nunique",
        ),

        human_transfer_pearson=(
            "human_transfer_pearson",
            "mean",
        ),

        human_transfer_spearman=(
            "human_transfer_spearman",
            "mean",
        ),

        human_transfer_spearman_std=(
            "human_transfer_spearman",
            "std",
        ),

        metricx_transfer_pearson=(
            "metricx_transfer_pearson",
            "mean",
        ),

        metricx_transfer_spearman=(
            "metricx_transfer_spearman",
            "mean",
        ),

        metricx_transfer_spearman_std=(
            "metricx_transfer_spearman",
            "std",
        ),
    )
)


# ============================================================
# Attach our MAE gain to each language
# ============================================================

human_gain = (
    language_results[
        language_results["method"]
        == "Stratified-Human"
    ][
        [
            "language_pair",
            "budget",
            "mae_reduction_vs_random_pct",
        ]
    ]
)


metricx_gain = (
    language_results[
        language_results["method"]
        == "Stratified-MetricX"
    ][
        [
            "language_pair",
            "budget",
            "mae_reduction_vs_random_pct",
        ]
    ]
)


for budget in BUDGETS:

    temp = (
        human_gain[
            human_gain["budget"]
            == budget
        ][
            [
                "language_pair",
                "mae_reduction_vs_random_pct",
            ]
        ]
        .rename(
            columns={
                "mae_reduction_vs_random_pct":
                    f"human_gain_m{budget}_pct"
            }
        )
    )

    language_transfer = (
        language_transfer
        .merge(
            temp,
            on="language_pair",
            how="left",
        )
    )


    temp_metricx = (
        metricx_gain[
            metricx_gain["budget"]
            == budget
        ][
            [
                "language_pair",
                "mae_reduction_vs_random_pct",
            ]
        ]
        .rename(
            columns={
                "mae_reduction_vs_random_pct":
                    f"metricx_gain_m{budget}_pct"
            }
        )
    )


    language_transfer = (
        language_transfer
        .merge(
            temp_metricx,
            on="language_pair",
            how="left",
        )
    )


language_transfer.to_csv(
    OUT_DIR
    / "language_mechanism_summary.csv",
    index=False,
)


# ============================================================
# Relationship:
# difficulty transfer vs MAE improvement
# ============================================================

mechanism_rows = []


for budget in BUDGETS:

    # ----------------------------------------
    # Human difficulty
    # ----------------------------------------

    x = (
        language_transfer[
            "human_transfer_spearman"
        ]
        .to_numpy(
            dtype=float
        )
    )

    y = (
        language_transfer[
            f"human_gain_m{budget}_pct"
        ]
        .to_numpy(
            dtype=float
        )
    )


    (
        pearson_r,
        pearson_p,
        spearman_rho,
        spearman_p,
    ) = safe_corr(
        x,
        y,
    )


    mechanism_rows.append(
        {
            "difficulty_signal":
                "Human",

            "budget":
                budget,

            "pearson_r":
                pearson_r,

            "pearson_p":
                pearson_p,

            "spearman_rho":
                spearman_rho,

            "spearman_p":
                spearman_p,
        }
    )


    # ----------------------------------------
    # MetricX difficulty
    # ----------------------------------------

    x = (
        language_transfer[
            "metricx_transfer_spearman"
        ]
        .to_numpy(
            dtype=float
        )
    )

    y = (
        language_transfer[
            f"metricx_gain_m{budget}_pct"
        ]
        .to_numpy(
            dtype=float
        )
    )


    (
        pearson_r,
        pearson_p,
        spearman_rho,
        spearman_p,
    ) = safe_corr(
        x,
        y,
    )


    mechanism_rows.append(
        {
            "difficulty_signal":
                "MetricX",

            "budget":
                budget,

            "pearson_r":
                pearson_r,

            "pearson_p":
                pearson_p,

            "spearman_rho":
                spearman_rho,

            "spearman_p":
                spearman_p,
        }
    )


mechanism = pd.DataFrame(
    mechanism_rows
)


mechanism.to_csv(
    OUT_DIR
    / "difficulty_transfer_vs_gain.csv",
    index=False,
)


# ============================================================
# 2. Sentence-length diagnostic
#
# Primary measure = source CHARACTER length.
#
# Character length is preferable for a multilingual analysis
# because whitespace word counts are misleading for languages
# such as Japanese and Chinese.
# ============================================================

length_rows = []


for language_index, language_pair in enumerate(
    LANGUAGE_PAIRS
):

    data = load_language(
        language_pair
    )

    source_char_lengths = np.asarray(
        [
            len(
                item["src"]
            )
            for item in data
        ],
        dtype=float,
    )


    full_mean_chars = float(
        np.mean(
            source_char_lengths
        )
    )


    for split_info in system_splits[
        language_pair
    ]:

        split_id = int(
            split_info["split_id"]
        )

        dev_systems = list(
            split_info["dev_systems"]
        )


        human_bins = (
            make_bins(
                difficulty_vector(
                    data,
                    dev_systems,
                    signal="human",
                )
            )
        )


        metricx_bins = (
            make_bins(
                difficulty_vector(
                    data,
                    dev_systems,
                    signal="metricx",
                )
            )
        )


        for trial in range(
            N_STOCHASTIC_TRIALS
        ):

            for budget in BUDGETS:

                seed = (
                    BASE_SEED
                    + language_index
                    * 100_000_000
                    + split_id
                    * 1_000_000
                    + trial
                    * 1000
                    + budget
                )


                # --------------------------------
                # Random
                # --------------------------------

                rng = (
                    np.random
                    .default_rng(
                        seed
                    )
                )


                random_ids = (
                    rng.choice(
                        len(data),
                        size=budget,
                        replace=False,
                    )
                    .tolist()
                )


                # --------------------------------
                # Stratified-Human
                # --------------------------------

                human_ids = (
                    stratified_sample(
                        human_bins,
                        budget,
                        seed,
                    )
                )


                # --------------------------------
                # Stratified-MetricX
                # --------------------------------

                metricx_ids = (
                    stratified_sample(
                        metricx_bins,
                        budget,
                        seed,
                    )
                )


                selections = {
                    "Random":
                        random_ids,

                    "Stratified-Human":
                        human_ids,

                    "Stratified-MetricX":
                        metricx_ids,
                }


                for method, ids in selections.items():

                    subset_mean_chars = float(
                        np.mean(
                            source_char_lengths[
                                ids
                            ]
                        )
                    )


                    signed_pct = (
                        100.0
                        * (
                            subset_mean_chars
                            - full_mean_chars
                        )
                        / full_mean_chars
                    )


                    length_rows.append(
                        {
                            "language_pair":
                                language_pair,

                            "split_id":
                                split_id,

                            "trial":
                                trial,

                            "budget":
                                budget,

                            "method":
                                method,

                            "full_mean_source_chars":
                                full_mean_chars,

                            "subset_mean_source_chars":
                                subset_mean_chars,

                            "signed_length_deviation_pct":
                                signed_pct,

                            "absolute_length_deviation_pct":
                                abs(
                                    signed_pct
                                ),
                        }
                    )


length_trials = pd.DataFrame(
    length_rows
)


length_trials.to_csv(
    OUT_DIR
    / "length_trials.csv",
    index=False,
)


# ============================================================
# Per-language length summary
# ============================================================

length_language = (
    length_trials
    .groupby(
        [
            "language_pair",
            "method",
            "budget",
        ],
        as_index=False,
    )
    .agg(
        full_mean_source_chars=(
            "full_mean_source_chars",
            "mean",
        ),

        subset_mean_source_chars=(
            "subset_mean_source_chars",
            "mean",
        ),

        mean_signed_length_deviation_pct=(
            "signed_length_deviation_pct",
            "mean",
        ),

        mean_absolute_length_deviation_pct=(
            "absolute_length_deviation_pct",
            "mean",
        ),
    )
)


length_language.to_csv(
    OUT_DIR
    / "length_language_summary.csv",
    index=False,
)


# ============================================================
# Macro length diagnostic
# ============================================================

length_macro = (
    length_language
    .groupby(
        [
            "method",
            "budget",
        ],
        as_index=False,
    )
    .agg(
        macro_signed_length_deviation_pct=(
            "mean_signed_length_deviation_pct",
            "mean",
        ),

        macro_absolute_length_deviation_pct=(
            "mean_absolute_length_deviation_pct",
            "mean",
        ),
    )
)


length_macro.to_csv(
    OUT_DIR
    / "length_macro_summary.csv",
    index=False,
)


# ============================================================
# Figure 1:
# human difficulty transfer vs gain at m=25
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)


x = (
    language_transfer[
        "human_transfer_spearman"
    ]
)

y = (
    language_transfer[
        "human_gain_m25_pct"
    ]
)


ax.scatter(
    x,
    y,
)


for _, row in language_transfer.iterrows():

    ax.annotate(
        row["language_pair"],
        (
            row[
                "human_transfer_spearman"
            ],
            row[
                "human_gain_m25_pct"
            ],
        ),
        fontsize=8,
        xytext=(3, 3),
        textcoords="offset points",
    )


ax.set_xlabel(
    "Dev-to-held-out difficulty correlation "
    "(Spearman)"
)

ax.set_ylabel(
    "Stratified-Human MAE reduction vs Random (%)"
)

ax.set_title(
    "Does transferable sentence difficulty explain "
    "stratification gains?"
)

ax.grid(
    True,
    alpha=0.25,
)


fig.tight_layout()


fig.savefig(
    OUT_DIR
    / "difficulty_transfer_vs_gain_m25.pdf",
    bbox_inches="tight",
)


fig.savefig(
    OUT_DIR
    / "difficulty_transfer_vs_gain_m25.png",
    dpi=220,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# Figure 2:
# sentence-length deviation
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)


for method in [
    "Random",
    "Stratified-MetricX",
    "Stratified-Human",
]:

    temp = (
        length_macro[
            length_macro["method"]
            == method
        ]
        .sort_values(
            "budget"
        )
    )


    ax.plot(
        temp["budget"],
        temp[
            "macro_absolute_length_deviation_pct"
        ],
        marker="o",
        label=method,
    )


ax.set_xlabel(
    "Sentence budget"
)

ax.set_ylabel(
    "Mean absolute deviation from full-set "
    "source length (%)"
)

ax.set_title(
    "Sentence-length composition diagnostic"
)

ax.set_xticks(
    BUDGETS
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend(
    frameon=False
)


fig.tight_layout()


fig.savefig(
    OUT_DIR
    / "length_diagnostic.pdf",
    bbox_inches="tight",
)


fig.savefig(
    OUT_DIR
    / "length_diagnostic.png",
    dpi=220,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# Print final summaries
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "LANGUAGE-LEVEL DIFFICULTY TRANSFER"
)

print(
    "=" * 120
)


print(
    language_transfer[
        [
            "language_pair",
            "human_transfer_spearman",
            "metricx_transfer_spearman",
            "human_gain_m25_pct",
            "metricx_gain_m25_pct",
        ]
    ]
    .sort_values(
        "human_transfer_spearman",
        ascending=False,
    )
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print(
    "\n"
    + "=" * 120
)

print(
    "DIFFICULTY TRANSFER VS MAE GAIN"
)

print(
    "=" * 120
)


print(
    mechanism.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


print(
    "\n"
    + "=" * 120
)

print(
    "SENTENCE-LENGTH DIAGNOSTIC"
)

print(
    "=" * 120
)


print(
    length_macro.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print(
    "\nSaved to:"
)

print(
    OUT_DIR
)

print(
    "\nDone."
)
