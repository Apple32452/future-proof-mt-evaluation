"""
Cross-year temporal-transfer evaluation.

Historical selector:
    Frozen 2024-era Aya-23-8B model.

Current evaluation population:
    WMT25 human-evaluated MT systems.

For every WMT25 sentence:
    1. Score Aya-23 translation against WMT25 reference using chrF.
    2. Define historical difficulty = 100 - chrF.
    3. Divide sentences into five equal-mass difficulty strata.
    4. Sample proportionally across strata.

Compare:
    Random
    Temporal-Aya-Stratified

Target:
    Reproduce full WMT25 human scores of ALL WMT25 systems.

This differs from the same-year experiment:
    No WMT25 system contributes to subset construction.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    kendalltau,
    spearmanr,
    wilcoxon,
)

from sacrebleu.metrics import CHRF


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

BASE_SEED = 20260930


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "results"
    / "wmt25_subset2evaluate_all"
)

AYA_DIR = (
    ROOT
    / "results"
    / "temporal_aya_wmt25"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "temporal_aya_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRIAL_FILE = (
    OUTPUT_DIR
    / "temporal_trials.csv"
)

LANGUAGE_FILE = (
    OUTPUT_DIR
    / "temporal_language_summary.csv"
)

MACRO_FILE = (
    OUTPUT_DIR
    / "temporal_macro_summary.csv"
)

SIGNIFICANCE_FILE = (
    OUTPUT_DIR
    / "temporal_significance.csv"
)

DIFFICULTY_FILE = (
    OUTPUT_DIR
    / "temporal_difficulty_transfer.csv"
)


# ============================================================
# chrF
# ============================================================

chrf = CHRF()


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

    return {
        system:
            float(
                np.mean(
                    [
                        item["scores"][
                            system
                        ]["human"]

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

    return {
        system:
            float(
                np.mean(
                    [
                        data[i]["scores"][
                            system
                        ]["human"]

                        for i
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

    correct = 0
    total = 0

    for i in range(
        len(systems)
    ):

        for j in range(
            i + 1,
            len(systems),
        ):

            a = systems[i]
            b = systems[j]

            ref_diff = (
                reference[a]
                - reference[b]
            )

            if ref_diff == 0:
                continue

            est_diff = (
                estimate[a]
                - estimate[b]
            )

            total += 1

            if (
                np.sign(ref_diff)
                == np.sign(est_diff)
            ):

                correct += 1

    if total == 0:
        return np.nan

    return (
        correct
        / total
    )


def evaluate(
    data,
    selected_ids,
    systems,
    reference,
):

    estimate = subset_scores(
        data,
        selected_ids,
        systems,
    )

    ref = np.asarray(
        [
            reference[s]
            for s in systems
        ],
        dtype=float,
    )

    est = np.asarray(
        [
            estimate[s]
            for s in systems
        ],
        dtype=float,
    )

    errors = est - ref

    rho = spearmanr(
        ref,
        est,
    ).statistic

    tau = kendalltau(
        ref,
        est,
    ).statistic

    return {
        "mae":
            float(
                np.mean(
                    np.abs(errors)
                )
            ),

        "bias":
            float(
                np.mean(errors)
            ),

        "spearman":
            float(rho),

        "kendall":
            float(tau),

        "pairwise":
            float(
                pairwise_accuracy(
                    reference,
                    estimate,
                    systems,
                )
            ),
    }


def make_bins(
    difficulty,
):

    order = np.argsort(
        difficulty,
        kind="stable",
    )

    return [
        np.asarray(
            x,
            dtype=int,
        )
        for x
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

    for i in range(
        remainder
    ):

        allocation[i] += 1

    selected = []

    for (
        indices,
        n_take,
    ) in zip(
        bins,
        allocation,
    ):

        chosen = rng.choice(
            indices,
            size=n_take,
            replace=False,
        )

        selected.extend(
            chosen.tolist()
        )

    assert len(selected) == budget
    assert len(set(selected)) == budget

    return selected


# ============================================================
# Main experiment
# ============================================================

trial_rows = []

difficulty_rows = []


for language_index, language_pair in enumerate(
    LANGUAGE_PAIRS
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"TEMPORAL EVALUATION: "
        f"{language_pair}"
    )

    print(
        "=" * 100
    )


    data_file = (
        DATA_DIR
        / (
            f"{language_pair}"
            "_subset2evaluate.jsonl"
        )
    )

    aya_file = (
        AYA_DIR
        / (
            f"{language_pair}"
            "_aya23_translations.jsonl"
        )
    )


    data = load_jsonl(
        data_file
    )

    aya_rows = load_jsonl(
        aya_file
    )


    assert (
        len(data)
        == len(aya_rows)
    )


    aya_by_id = {
        int(row["i"]):
            row
        for row
        in aya_rows
    }


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
    # Historical Aya difficulty
    # ========================================================

    aya_chrf = []


    for item in data:

        item_id = int(
            item["i"]
        )

        aya_item = (
            aya_by_id[
                item_id
            ]
        )

        hypothesis = (
            aya_item[
                "aya23_translation"
            ]
        )

        reference_text = (
            item["ref"]
        )


        if not reference_text:

            raise ValueError(
                f"Missing reference for "
                f"{language_pair}, "
                f"id={item_id}"
            )


        score = float(
            chrf.sentence_score(
                hypothesis,
                [
                    reference_text
                ],
            ).score
        )


        aya_chrf.append(
            score
        )


    aya_chrf = np.asarray(
        aya_chrf,
        dtype=float,
    )


    temporal_difficulty = (
        100.0
        - aya_chrf
    )


    bins = make_bins(
        temporal_difficulty
    )


    # ========================================================
    # How transferable is historical Aya difficulty?
    #
    # Compare Aya-2024 difficulty with average WMT25 human
    # difficulty across ALL WMT25 systems.
    # ========================================================

    wmt25_human_difficulty = []


    for item in data:

        mean_quality = np.mean(
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
                mean_quality
            )
        )


    wmt25_human_difficulty = np.asarray(
        wmt25_human_difficulty,
        dtype=float,
    )


    temporal_pearson = np.corrcoef(
        temporal_difficulty,
        wmt25_human_difficulty,
    )[0, 1]


    temporal_spearman = (
        spearmanr(
            temporal_difficulty,
            wmt25_human_difficulty,
        ).statistic
    )


    difficulty_rows.append(
        {
            "language_pair":
                language_pair,

            "num_segments":
                len(data),

            "num_wmt25_systems":
                len(systems),

            "aya_mean_chrf":
                float(
                    np.mean(
                        aya_chrf
                    )
                ),

            "aya_chrf_std":
                float(
                    np.std(
                        aya_chrf
                    )
                ),

            "temporal_difficulty_pearson":
                float(
                    temporal_pearson
                ),

            "temporal_difficulty_spearman":
                float(
                    temporal_spearman
                ),
        }
    )


    print(
        f"Aya mean chrF: "
        f"{np.mean(aya_chrf):.4f}"
    )

    print(
        f"Aya→WMT25 difficulty "
        f"Spearman: "
        f"{temporal_spearman:.4f}"
    )


    # ========================================================
    # WMT25 full-test reference scores
    # ========================================================

    reference_scores = (
        full_reference_scores(
            data,
            systems,
        )
    )


    # ========================================================
    # Random vs historical-Aya stratification
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


            # --------------------------------------------
            # Random
            # --------------------------------------------

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


            metrics = evaluate(
                data,
                random_ids,
                systems,
                reference_scores,
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

                    **metrics,
                }
            )


            # --------------------------------------------
            # Temporal Aya stratification
            # --------------------------------------------

            temporal_ids = (
                stratified_sample(
                    bins,
                    budget,
                    seed,
                )
            )


            metrics = evaluate(
                data,
                temporal_ids,
                systems,
                reference_scores,
            )


            trial_rows.append(
                {
                    "language_pair":
                        language_pair,

                    "method":
                        "Temporal-Aya-Stratified",

                    "budget":
                        budget,

                    "trial":
                        trial,

                    **metrics,
                }
            )


    print(
        f"{language_pair}: COMPLETE"
    )


# ============================================================
# Save trial data
# ============================================================

trials = pd.DataFrame(
    trial_rows
)


trials.to_csv(
    TRIAL_FILE,
    index=False,
)


difficulty_df = pd.DataFrame(
    difficulty_rows
)


difficulty_df.to_csv(
    DIFFICULTY_FILE,
    index=False,
)


# ============================================================
# Average trials within language
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
    )
)


# ============================================================
# Attach Random baseline
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
# Macro across six temporal-transfer languages
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
            lambda x:
                float(
                    np.mean(
                        np.asarray(
                            x
                        )
                        > 0
                    )
                ),
        ),
    )
)


macro[
    "macro_mae_reduction_vs_random_pct"
] = (
    100.0
    * (
        macro[
            "random_mae_mean"
        ]
        - macro[
            "mae_mean"
        ]
    )
    / macro[
        "random_mae_mean"
    ]
)


macro.to_csv(
    MACRO_FILE,
    index=False,
)


# ============================================================
# Language-level inference
#
# Only six languages, so treat significance conservatively.
# ============================================================

sig_rows = []


rng_bootstrap = (
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
                == "Temporal-Aya-Stratified"
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


    bootstrap_means = []


    for _ in range(
        BOOTSTRAP_REPS
    ):

        sample = (
            rng_bootstrap.choice(
                deltas,
                size=len(deltas),
                replace=True,
            )
        )

        bootstrap_means.append(
            float(
                np.mean(
                    sample
                )
            )
        )


    bootstrap_means = np.asarray(
        bootstrap_means
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

        p_value = float(
            wilcoxon(
                deltas,
                alternative="greater",
            ).pvalue
        )


    except ValueError:

        p_value = np.nan


    sig_rows.append(
        {
            "budget":
                budget,

            "num_languages":
                len(deltas),

            "mean_absolute_mae_improvement":
                float(
                    np.mean(
                        deltas
                    )
                ),

            "bootstrap_ci_low":
                ci_low,

            "bootstrap_ci_high":
                ci_high,

            "wilcoxon_one_sided_p":
                p_value,

            "language_win_rate":
                float(
                    np.mean(
                        deltas > 0
                    )
                ),
        }
    )


significance = pd.DataFrame(
    sig_rows
)


significance.to_csv(
    SIGNIFICANCE_FILE,
    index=False,
)


# ============================================================
# Print
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "TEMPORAL DIFFICULTY TRANSFER"
)

print(
    "=" * 120
)


print(
    difficulty_df.to_string(
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
    "TEMPORAL TRANSFER MACRO SUMMARY"
)

print(
    "=" * 120
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
    + "=" * 120
)

print(
    "TEMPORAL LANGUAGE-LEVEL SIGNIFICANCE"
)

print(
    "=" * 120
)


print(
    significance.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nDone."
)
