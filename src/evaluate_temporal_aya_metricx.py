"""
Final cross-year temporal-transfer evaluation using Aya-23 + MetricX.

Historical selector signal:
    Frozen 2024-era Aya-23 translations of WMT25 source sentences,
    scored with MetricX-24 against the WMT25 references.

Current evaluation population:
    All human-evaluated WMT25 MT systems for each language pair.

Important:
    NO WMT25 system output or WMT25 system score is used to construct
    the temporal difficulty strata.

For each WMT25 sentence:
    1. Read the previously generated Aya-23 translation.
    2. Read its previously computed MetricX-24 prediction.
    3. Use MetricX error directly as difficulty:
           higher MetricX error = harder sentence
    4. Divide sentences into 5 empirical quantile strata.
    5. Sample approximately proportionally across strata.

Compare:
    Random
    Temporal-Aya-MetricX-Stratified

Evaluation:
    MAE
    signed bias
    Spearman rho
    Kendall tau
    pairwise ranking accuracy

Budgets:
    10, 25, 50, 100

Statistical unit:
    language pair

NOTE:
    This script performs NO Aya inference and NO MetricX inference.
    It only reads previously saved outputs and runs inexpensive
    sampling/evaluation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    kendalltau,
    pearsonr,
    spearmanr,
    wilcoxon,
)


# ============================================================
# Configuration
# ============================================================

LANGUAGE_PAIRS = [
    "cs-uk_UA",
    "en-cs_CZ",
    "en-ja_JP",
    "en-ru_RU",
    "en-uk_UA",
    "en-zh_CN",
]

BUDGETS = [
    10,
    25,
    50,
    100,
]

N_BINS = 5

N_TRIALS = 1000

BOOTSTRAP_REPS = 10000

BASE_SEED = 20261001


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "results"
    / "wmt25_subset2evaluate_all"
)

METRICX_DIR = (
    ROOT
    / "results"
    / "temporal_aya_metricx"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "temporal_aya_metricx_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRIAL_FILE = (
    OUTPUT_DIR
    / "temporal_metricx_trials.csv"
)

LANGUAGE_FILE = (
    OUTPUT_DIR
    / "temporal_metricx_language_summary.csv"
)

MACRO_FILE = (
    OUTPUT_DIR
    / "temporal_metricx_macro_summary.csv"
)

SIGNIFICANCE_FILE = (
    OUTPUT_DIR
    / "temporal_metricx_significance.csv"
)

DIFFICULTY_FILE = (
    OUTPUT_DIR
    / "temporal_metricx_difficulty_transfer.csv"
)


# ============================================================
# Helpers
# ============================================================

def load_jsonl(path):

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


def full_reference_scores(
    data,
    systems,
):

    """
    Full-test human reference score for every WMT25 system.
    """

    return {
        system:
            float(
                np.mean(
                    [
                        item[
                            "scores"
                        ][
                            system
                        ][
                            "human"
                        ]

                        for item
                        in data
                    ]
                )
            )

        for system
        in systems
    }


def subset_scores(
    data,
    selected_ids,
    systems,
):

    """
    Human score estimated using only the selected sentence subset.
    """

    return {
        system:
            float(
                np.mean(
                    [
                        data[
                            item_index
                        ][
                            "scores"
                        ][
                            system
                        ][
                            "human"
                        ]

                        for item_index
                        in selected_ids
                    ]
                )
            )

        for system
        in systems
    }


def pairwise_accuracy(
    reference,
    estimate,
    systems,
):

    """
    Fraction of system pairs whose ordering agrees with the
    full-test ordering.
    """

    correct = 0
    total = 0

    for i in range(
        len(systems)
    ):

        for j in range(
            i + 1,
            len(systems),
        ):

            system_a = systems[i]
            system_b = systems[j]

            reference_difference = (
                reference[
                    system_a
                ]
                - reference[
                    system_b
                ]
            )

            # Ignore exact full-test ties.
            if reference_difference == 0:
                continue

            estimate_difference = (
                estimate[
                    system_a
                ]
                - estimate[
                    system_b
                ]
            )

            total += 1

            if (
                np.sign(
                    reference_difference
                )
                == np.sign(
                    estimate_difference
                )
            ):

                correct += 1

    if total == 0:
        return np.nan

    return (
        correct
        / total
    )


def evaluate_selection(
    data,
    selected_ids,
    systems,
    reference,
):

    """
    Evaluate one selected subset against the full-test
    human reference scores.
    """

    estimate = subset_scores(
        data,
        selected_ids,
        systems,
    )

    reference_vector = np.asarray(
        [
            reference[
                system
            ]

            for system
            in systems
        ],
        dtype=float,
    )

    estimate_vector = np.asarray(
        [
            estimate[
                system
            ]

            for system
            in systems
        ],
        dtype=float,
    )

    errors = (
        estimate_vector
        - reference_vector
    )

    mae = float(
        np.mean(
            np.abs(
                errors
            )
        )
    )

    bias = float(
        np.mean(
            errors
        )
    )

    spearman_value = float(
        spearmanr(
            reference_vector,
            estimate_vector,
        ).statistic
    )

    kendall_value = float(
        kendalltau(
            reference_vector,
            estimate_vector,
        ).statistic
    )

    pairwise_value = float(
        pairwise_accuracy(
            reference,
            estimate,
            systems,
        )
    )

    return {
        "mae":
            mae,

        "bias":
            bias,

        "spearman":
            spearman_value,

        "kendall":
            kendall_value,

        "pairwise":
            pairwise_value,
    }


# ============================================================
# Difficulty stratification
# ============================================================

def make_bins(
    difficulty,
):

    """
    Sort sentences by temporal difficulty and divide them
    into five approximately equal-size empirical quantile strata.
    """

    order = np.argsort(
        difficulty,
        kind="stable",
    )

    return [
        np.asarray(
            part,
            dtype=int,
        )

        for part
        in np.array_split(
            order,
            N_BINS,
        )
    ]


def stratified_sample(
    bins,
    budget,
    seed,
):

    """
    Approximately proportional sampling across equal-mass strata.

    Because the strata contain approximately N/B items each,
    drawing approximately m/B items from each stratum preserves
    the marginal difficulty mixture of the full test set.
    """

    rng = np.random.default_rng(
        seed
    )

    base = (
        budget
        // len(bins)
    )

    remainder = (
        budget
        % len(bins)
    )

    allocation = [
        base
        for _ in bins
    ]

    for bin_index in range(
        remainder
    ):

        allocation[
            bin_index
        ] += 1

    selected = []

    for (
        bin_indices,
        n_take,
    ) in zip(
        bins,
        allocation,
    ):

        if n_take > len(
            bin_indices
        ):

            raise ValueError(
                "Requested more samples from a stratum "
                "than the stratum contains."
            )

        chosen = rng.choice(
            bin_indices,
            size=n_take,
            replace=False,
        )

        selected.extend(
            chosen.tolist()
        )

    assert (
        len(selected)
        == budget
    )

    assert (
        len(
            set(
                selected
            )
        )
        == budget
    )

    return selected


# ============================================================
# Experiment storage
# ============================================================

trial_rows = []

difficulty_rows = []


# ============================================================
# Main experiment
# ============================================================

for (
    language_index,
    language_pair,
) in enumerate(
    LANGUAGE_PAIRS
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"TEMPORAL AYA+METRICX EVALUATION: "
        f"{language_pair}"
    )

    print(
        "=" * 100
    )


    # --------------------------------------------------------
    # Load WMT25 human-evaluation dataset
    # --------------------------------------------------------

    data_file = (
        DATA_DIR
        / (
            f"{language_pair}"
            "_subset2evaluate.jsonl"
        )
    )

    if not data_file.exists():

        raise FileNotFoundError(
            f"Missing WMT25 data:\n"
            f"{data_file}"
        )

    data = load_jsonl(
        data_file
    )


    # --------------------------------------------------------
    # Load PRECOMPUTED Aya+MetricX scores
    # --------------------------------------------------------

    metricx_file = (
        METRICX_DIR
        / (
            f"{language_pair}"
            "_output.jsonl"
        )
    )

    if not metricx_file.exists():

        raise FileNotFoundError(
            f"Missing MetricX output:\n"
            f"{metricx_file}"
        )

    metricx_rows = load_jsonl(
        metricx_file
    )


    # --------------------------------------------------------
    # Verify row counts
    # --------------------------------------------------------

    if (
        len(data)
        != len(
            metricx_rows
        )
    ):

        raise ValueError(
            f"{language_pair}: "
            f"WMT25 rows={len(data)}, "
            f"MetricX rows={len(metricx_rows)}"
        )


    # --------------------------------------------------------
    # Map MetricX scores by sentence ID
    # --------------------------------------------------------

    metricx_by_id = {}


    for row in metricx_rows:

        item_id = int(
            row["i"]
        )

        score = float(
            row[
                "prediction"
            ]
        )

        if not np.isfinite(
            score
        ):

            raise ValueError(
                f"{language_pair}: "
                f"non-finite MetricX score "
                f"for id={item_id}"
            )

        metricx_by_id[
            item_id
        ] = score


    data_ids = {
        int(
            item["i"]
        )
        for item
        in data
    }


    if (
        set(
            metricx_by_id.keys()
        )
        != data_ids
    ):

        missing = (
            data_ids
            - set(
                metricx_by_id.keys()
            )
        )

        extra = (
            set(
                metricx_by_id.keys()
            )
            - data_ids
        )

        raise ValueError(
            f"{language_pair}: "
            f"ID mismatch. "
            f"missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


    # --------------------------------------------------------
    # WMT25 systems
    # --------------------------------------------------------

    systems = sorted(
        data[0][
            "scores"
        ].keys()
    )


    print(
        f"Segments: "
        f"{len(data)}"
    )

    print(
        f"WMT25 systems evaluated: "
        f"{len(systems)}"
    )


    # ========================================================
    # Historical temporal difficulty
    #
    # IMPORTANT:
    #
    # MetricX-24 is an error score.
    #
    # Higher prediction = worse Aya translation
    #                   = greater sentence difficulty.
    #
    # Therefore we use the prediction DIRECTLY.
    # ========================================================

    temporal_difficulty = np.asarray(
        [
            metricx_by_id[
                int(
                    item["i"]
                )
            ]

            for item
            in data
        ],
        dtype=float,
    )


    print(
        f"Aya+MetricX mean error: "
        f"{np.mean(temporal_difficulty):.4f}"
    )

    print(
        f"Aya+MetricX error std: "
        f"{np.std(temporal_difficulty):.4f}"
    )


    # ========================================================
    # Construct difficulty strata
    # ========================================================

    bins = make_bins(
        temporal_difficulty
    )


    print(
        "Difficulty stratum sizes:",
        [
            len(
                x
            )
            for x
            in bins
        ],
    )


    # ========================================================
    # Measure temporal difficulty transfer
    #
    # Compare the historical Aya+MetricX difficulty signal
    # against average human difficulty of the WMT25 system
    # population.
    # ========================================================

    wmt25_human_difficulty = []


    for item in data:

        mean_wmt25_quality = np.mean(
            [
                item[
                    "scores"
                ][
                    system
                ][
                    "human"
                ]

                for system
                in systems
            ]
        )

        wmt25_human_difficulty.append(
            -float(
                mean_wmt25_quality
            )
        )


    wmt25_human_difficulty = np.asarray(
        wmt25_human_difficulty,
        dtype=float,
    )


    temporal_pearson_result = (
        pearsonr(
            temporal_difficulty,
            wmt25_human_difficulty,
        )
    )

    temporal_spearman_result = (
        spearmanr(
            temporal_difficulty,
            wmt25_human_difficulty,
        )
    )


    temporal_pearson = float(
        temporal_pearson_result.statistic
    )

    temporal_pearson_p = float(
        temporal_pearson_result.pvalue
    )

    temporal_spearman = float(
        temporal_spearman_result.statistic
    )

    temporal_spearman_p = float(
        temporal_spearman_result.pvalue
    )


    print(
        "Aya+MetricX → WMT25 human difficulty "
        f"Pearson: {temporal_pearson:.4f}"
    )

    print(
        "Aya+MetricX → WMT25 human difficulty "
        f"Spearman: {temporal_spearman:.4f}"
    )


    difficulty_rows.append(
        {
            "language_pair":
                language_pair,

            "num_segments":
                len(
                    data
                ),

            "num_wmt25_systems":
                len(
                    systems
                ),

            "aya_metricx_mean":
                float(
                    np.mean(
                        temporal_difficulty
                    )
                ),

            "aya_metricx_std":
                float(
                    np.std(
                        temporal_difficulty
                    )
                ),

            "temporal_difficulty_pearson":
                temporal_pearson,

            "temporal_difficulty_pearson_p":
                temporal_pearson_p,

            "temporal_difficulty_spearman":
                temporal_spearman,

            "temporal_difficulty_spearman_p":
                temporal_spearman_p,
        }
    )


    # ========================================================
    # Full-test human reference scores
    #
    # Every WMT25 system is unseen by the temporal selector,
    # because no WMT25 system was used to construct difficulty.
    # ========================================================

    reference_scores = (
        full_reference_scores(
            data,
            systems,
        )
    )


    # ========================================================
    # Random vs Temporal-Aya-MetricX-Stratified
    # ========================================================

    for trial in range(
        N_TRIALS
    ):

        for budget in BUDGETS:

            seed = (
                BASE_SEED
                + language_index
                * 10_000_000
                + trial
                * 1000
                + budget
            )


            # ------------------------------------------------
            # Random baseline
            # ------------------------------------------------

            rng = (
                np.random
                .default_rng(
                    seed
                )
            )

            random_ids = (
                rng.choice(
                    len(
                        data
                    ),
                    size=budget,
                    replace=False,
                )
                .tolist()
            )


            random_metrics = (
                evaluate_selection(
                    data,
                    random_ids,
                    systems,
                    reference_scores,
                )
            )


            trial_rows.append(
                {
                    "language_pair":
                        language_pair,

                    "method":
                        "Random",

                    "budget":
                        budget,

                    "trial":
                        trial,

                    **random_metrics,
                }
            )


            # ------------------------------------------------
            # Temporal Aya + MetricX stratification
            # ------------------------------------------------

            temporal_ids = (
                stratified_sample(
                    bins,
                    budget,
                    seed,
                )
            )


            temporal_metrics = (
                evaluate_selection(
                    data,
                    temporal_ids,
                    systems,
                    reference_scores,
                )
            )


            trial_rows.append(
                {
                    "language_pair":
                        language_pair,

                    "method":
                        "Temporal-Aya-MetricX-Stratified",

                    "budget":
                        budget,

                    "trial":
                        trial,

                    **temporal_metrics,
                }
            )


    print(
        f"{language_pair}: COMPLETE"
    )


# ============================================================
# Save trial-level results
# ============================================================

trials = pd.DataFrame(
    trial_rows
)


trials.to_csv(
    TRIAL_FILE,
    index=False,
)


# ============================================================
# Save temporal difficulty-transfer diagnostics
# ============================================================

difficulty_df = pd.DataFrame(
    difficulty_rows
)


difficulty_df.to_csv(
    DIFFICULTY_FILE,
    index=False,
)


# ============================================================
# Average sentence-sampling trials inside each language
# ============================================================

language_summary = (
    trials
    .groupby(
        [
            "language_pair",
            "method",
            "budget",
        ],
        as_index=False,
    )
    .agg(
        mae=(
            "mae",
            "mean",
        ),

        bias=(
            "bias",
            "mean",
        ),

        spearman=(
            "spearman",
            "mean",
        ),

        kendall=(
            "kendall",
            "mean",
        ),

        pairwise=(
            "pairwise",
            "mean",
        ),

        n_trials=(
            "mae",
            "size",
        ),
    )
)


# ============================================================
# Attach matched Random baseline
# ============================================================

random_language = (
    language_summary[
        language_summary[
            "method"
        ]
        == "Random"
    ][
        [
            "language_pair",
            "budget",
            "mae",
        ]
    ]
    .rename(
        columns={
            "mae":
                "random_mae"
        }
    )
)


language_summary = (
    language_summary
    .merge(
        random_language,
        on=[
            "language_pair",
            "budget",
        ],
        how="left",
    )
)


language_summary[
    "mae_improvement"
] = (
    language_summary[
        "random_mae"
    ]
    - language_summary[
        "mae"
    ]
)


language_summary[
    "mae_reduction_vs_random_pct"
] = (
    100.0
    * language_summary[
        "mae_improvement"
    ]
    / language_summary[
        "random_mae"
    ]
)


language_summary.to_csv(
    LANGUAGE_FILE,
    index=False,
)


# ============================================================
# Macro average across the six temporal-transfer languages
# ============================================================

macro = (
    language_summary
    .groupby(
        [
            "method",
            "budget",
        ],
        as_index=False,
    )
    .agg(
        num_languages=(
            "language_pair",
            "nunique",
        ),

        mae_mean=(
            "mae",
            "mean",
        ),

        bias_mean=(
            "bias",
            "mean",
        ),

        spearman_mean=(
            "spearman",
            "mean",
        ),

        kendall_mean=(
            "kendall",
            "mean",
        ),

        pairwise_mean=(
            "pairwise",
            "mean",
        ),

        random_mae_mean=(
            "random_mae",
            "mean",
        ),

        language_win_rate=(
            "mae_improvement",
            lambda values:
                float(
                    np.mean(
                        np.asarray(
                            values,
                            dtype=float,
                        )
                        > 0
                    )
                ),
        ),
    )
)


macro[
    "macro_absolute_mae_improvement"
] = (
    macro[
        "random_mae_mean"
    ]
    - macro[
        "mae_mean"
    ]
)


macro[
    "macro_mae_reduction_vs_random_pct"
] = (
    100.0
    * macro[
        "macro_absolute_mae_improvement"
    ]
    / macro[
        "random_mae_mean"
    ]
)


macro.to_csv(
    MACRO_FILE,
    index=False,
)


# ============================================================
# Paired language-level significance
#
# Only six languages are available in the temporal experiment,
# so effect sizes and consistency should be emphasized.
# ============================================================

significance_rows = []


bootstrap_rng = (
    np.random
    .default_rng(
        BASE_SEED
        + 999999
    )
)


for budget in BUDGETS:

    local = (
        language_summary[
            (
                language_summary[
                    "method"
                ]
                == "Temporal-Aya-MetricX-Stratified"
            )
            &
            (
                language_summary[
                    "budget"
                ]
                == budget
            )
        ]
        .sort_values(
            "language_pair"
        )
    )


    deltas = (
        local[
            "mae_improvement"
        ]
        .to_numpy(
            dtype=float
        )
    )


    if (
        len(
            deltas
        )
        == 0
    ):

        continue


    bootstrap_means = np.empty(
        BOOTSTRAP_REPS,
        dtype=float,
    )


    for bootstrap_index in range(
        BOOTSTRAP_REPS
    ):

        sampled = (
            bootstrap_rng.choice(
                deltas,
                size=len(
                    deltas
                ),
                replace=True,
            )
        )

        bootstrap_means[
            bootstrap_index
        ] = float(
            np.mean(
                sampled
            )
        )


    ci_low = float(
        np.percentile(
            bootstrap_means,
            2.5,
        )
    )


    ci_high = float(
        np.percentile(
            bootstrap_means,
            97.5,
        )
    )


    try:

        wilcoxon_p = float(
            wilcoxon(
                deltas,
                alternative="greater",
                zero_method="wilcox",
            ).pvalue
        )


    except ValueError:

        wilcoxon_p = np.nan


    significance_rows.append(
        {
            "budget":
                budget,

            "num_languages":
                len(
                    deltas
                ),

            "mean_absolute_mae_improvement":
                float(
                    np.mean(
                        deltas
                    )
                ),

            "median_absolute_mae_improvement":
                float(
                    np.median(
                        deltas
                    )
                ),

            "bootstrap_ci_low":
                ci_low,

            "bootstrap_ci_high":
                ci_high,

            "bootstrap_nonpositive_rate":
                float(
                    np.mean(
                        bootstrap_means
                        <= 0
                    )
                ),

            "wilcoxon_one_sided_p":
                wilcoxon_p,

            "language_win_rate":
                float(
                    np.mean(
                        deltas
                        > 0
                    )
                ),
        }
    )


significance = pd.DataFrame(
    significance_rows
)


significance.to_csv(
    SIGNIFICANCE_FILE,
    index=False,
)


# ============================================================
# Final printed output
# ============================================================

print(
    "\n"
    + "=" * 125
)

print(
    "TEMPORAL AYA+METRICX DIFFICULTY TRANSFER"
)

print(
    "=" * 125
)


print(
    difficulty_df[
        [
            "language_pair",
            "num_segments",
            "num_wmt25_systems",
            "aya_metricx_mean",
            "aya_metricx_std",
            "temporal_difficulty_pearson",
            "temporal_difficulty_spearman",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print(
    "\n"
    + "=" * 125
)

print(
    "TEMPORAL AYA+METRICX MACRO SUMMARY"
)

print(
    "=" * 125
)


print(
    macro[
        [
            "method",
            "budget",
            "num_languages",
            "mae_mean",
            "bias_mean",
            "spearman_mean",
            "kendall_mean",
            "pairwise_mean",
            "macro_mae_reduction_vs_random_pct",
            "language_win_rate",
        ]
    ]
    .sort_values(
        [
            "budget",
            "method",
        ]
    )
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print(
    "\n"
    + "=" * 125
)

print(
    "TEMPORAL AYA+METRICX LANGUAGE-LEVEL SIGNIFICANCE"
)

print(
    "=" * 125
)


print(
    significance[
        [
            "budget",
            "num_languages",
            "mean_absolute_mae_improvement",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "wilcoxon_one_sided_p",
            "language_win_rate",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


print(
    "\nSaved:"
)

print(
    TRIAL_FILE
)

print(
    LANGUAGE_FILE
)

print(
    MACRO_FILE
)

print(
    SIGNIFICANCE_FILE
)

print(
    DIFFICULTY_FILE
)


print(
    "\nDone."
)