"""
Strong-baseline model-family holdout experiment.

Dataset:
    WMT25 cs-de_DE
    189 common human + MetricX-covered segments
    20 MT systems

Family holdouts:
    Gemma
    Qwen
    CommandA
    EuroLLM

For each family:
    - every system belonging to that family is completely hidden
      during sentence selection
    - all remaining systems are development systems

Methods:
    Random

    Zouhar et al.:
        MetricAvg
        MetricVar
        MetricCons
        Diversity
        DiffDisc

    Ours:
        Stratified-MetricX
        Stratified-Human

Evaluation:
    MAE
    signed bias
    Spearman rho
    Kendall tau
    pairwise ranking accuracy

Important:
    Ranking metrics are coarse for held-out families containing
    only 2 or 3 systems. MAE is therefore especially important
    in this experiment.
"""

import copy
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    kendalltau,
    spearmanr,
)

import subset2evaluate


# ============================================================
# Configuration
# ============================================================

LANGUAGE_PAIR = "cs-de_DE"

METRIC = "MetricX-24-Hybrid-XL"

BUDGETS = [
    10,
    25,
    50,
    100,
]

N_BINS = 5

# Stable estimates for Random and our stochastic methods.
N_TRIALS = 1000

BASE_SEED = 20260925

IRT_EPOCHS = 1000

RUN_DIFFDISC = True


# ============================================================
# Manually curated model-family mapping
#
# Based on the WMT25 system metadata used in our previous
# family-holdout experiment.
#
# SRPOL is intentionally NOT assigned to EuroLLM because its
# metadata indicates a hybrid EuroLLM/NLLB setup.
# ============================================================

FAMILY_MAP = {
    "CommandA": [
        "CommandA",
        "CommandA-MT",
    ],

    "EuroLLM": [
        "CUNI-MH-v2",
        "Laniqo",
    ],

    "Gemma": [
        "Gemma-3-12B",
        "Gemma-3-27B",
        "GemTrans",
        "IRB-MT",
        "UvA-MT",
    ],

    "Qwen": [
        "Algharb",
        "Wenyiil",
        "Yolu",
    ],
}


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "results"
    / "wmt25_subset2evaluate"
    / f"{LANGUAGE_PAIR}_subset2evaluate.jsonl"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "zouhar_family_holdout"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRIAL_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_family_trials.csv"
)

FAMILY_SUMMARY_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_family_summary.csv"
)

MACRO_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_family_macro_summary.csv"
)

REFERENCE_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_family_reference_scores.csv"
)

FAILURE_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_diffdisc_failures.csv"
)


# ============================================================
# CPU compatibility for DiffDisc
# ============================================================

def enable_cpu_irt_compatibility():

    try:
        import torch
        import py_irt.training

    except ImportError as exc:

        print(
            "WARNING: py-irt unavailable:"
        )

        print(exc)

        return False


    if torch.cuda.is_available():

        print(
            "DiffDisc device: CUDA"
        )

        return True


    original_train = (
        py_irt
        .training
        .IrtModelTrainer
        .train
    )


    if getattr(
        original_train,
        "_futureproof_cpu_wrapper",
        False,
    ):

        print(
            "DiffDisc device: CPU"
        )

        return True


    def train_cpu(
        self,
        *args,
        **kwargs,
    ):

        if (
            kwargs.get("device")
            == "cuda"
        ):

            kwargs["device"] = "cpu"


        return original_train(
            self,
            *args,
            **kwargs,
        )


    train_cpu._futureproof_cpu_wrapper = True


    (
        py_irt
        .training
        .IrtModelTrainer
        .train
    ) = train_cpu


    print(
        "DiffDisc device: CPU"
    )


    return True


# ============================================================
# Load common WMT25 dataset
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"Missing dataset:\n{DATA_FILE}"
    )


data = []

with open(
    DATA_FILE,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        data.append(
            json.loads(line)
        )


for i, item in enumerate(
    data
):

    item["i"] = i


ALL_SYSTEMS = sorted(
    data[0]["scores"].keys()
)


print("=" * 100)

print(
    "WMT25 STRONG-BASELINE MODEL-FAMILY HOLDOUT"
)

print("=" * 100)

print(
    f"\nLanguage pair: "
    f"{LANGUAGE_PAIR}"
)

print(
    f"Segments: "
    f"{len(data)}"
)

print(
    f"Systems: "
    f"{len(ALL_SYSTEMS)}"
)


assert (
    len(data)
    == 189
)

assert (
    len(ALL_SYSTEMS)
    == 20
)


for item in data:

    assert (
        set(item["scores"].keys())
        == set(ALL_SYSTEMS)
    )

    assert (
        set(item["tgt"].keys())
        == set(ALL_SYSTEMS)
    )


print(
    "Dataset validation: PASS"
)


# ============================================================
# Validate families
# ============================================================

eligible_families = {}


for family, members in FAMILY_MAP.items():

    members_present = [
        system
        for system
        in members
        if system
        in ALL_SYSTEMS
    ]


    if (
        len(members_present)
        >= 2
    ):

        eligible_families[
            family
        ] = members_present


print(
    "\nEligible family holdouts:"
)


for family, members in eligible_families.items():

    print(
        f"  {family:10s}: "
        f"{len(members)} systems "
        f"({', '.join(members)})"
    )


assert (
    len(eligible_families)
    == 4
)


# ============================================================
# Evaluation helpers
# ============================================================

def full_reference_scores(
    test_systems,
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
        in test_systems
    }


def subset_scores(
    selected_ids,
    test_systems,
):

    return {
        system:
            float(
                np.mean(
                    [
                        data[i][
                            "scores"
                        ][
                            system
                        ][
                            "human"
                        ]

                        for i
                        in selected_ids
                    ]
                )
            )

        for system
        in test_systems
    }


def pairwise_accuracy(
    reference,
    estimate,
    test_systems,
):

    correct = 0

    total = 0


    for (
        system_a,
        system_b,
    ) in combinations(
        test_systems,
        2,
    ):

        ref_diff = (
            reference[
                system_a
            ]
            - reference[
                system_b
            ]
        )


        # Ignore exact reference ties.
        if ref_diff == 0:
            continue


        pred_diff = (
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
                ref_diff
            )
            == np.sign(
                pred_diff
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
    selected_ids,
    test_systems,
    reference,
):

    estimate = subset_scores(
        selected_ids,
        test_systems,
    )


    ref_vector = np.asarray(
        [
            reference[
                system
            ]

            for system
            in test_systems
        ],
        dtype=float,
    )


    est_vector = np.asarray(
        [
            estimate[
                system
            ]

            for system
            in test_systems
        ],
        dtype=float,
    )


    errors = (
        est_vector
        - ref_vector
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


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )


        rho = (
            spearmanr(
                ref_vector,
                est_vector,
            )
            .statistic
        )


        tau = (
            kendalltau(
                ref_vector,
                est_vector,
            )
            .statistic
        )


    pairwise = (
        pairwise_accuracy(
            reference,
            estimate,
            test_systems,
        )
    )


    return {
        "mae":
            mae,

        "bias":
            bias,

        "spearman":
            float(rho),

        "kendall":
            float(tau),

        "pairwise":
            float(pairwise),
    }


# ============================================================
# Leakage-safe selector data
# ============================================================

def make_selector_data(
    dev_systems,
    test_systems,
):

    selector_data = []


    for original in data:

        item = copy.deepcopy(
            original
        )


        item["scores"] = {
            system:
                item["scores"][
                    system
                ]

            for system
            in dev_systems
        }


        item["tgt"] = {
            system:
                item["tgt"][
                    system
                ]

            for system
            in dev_systems
        }


        assert (
            set(item["scores"].keys())
            == set(dev_systems)
        )


        assert (
            set(item["tgt"].keys())
            == set(dev_systems)
        )


        assert (
            set(test_systems)
            .isdisjoint(
                item["scores"].keys()
            )
        )


        assert (
            set(test_systems)
            .isdisjoint(
                item["tgt"].keys()
            )
        )


        selector_data.append(
            item
        )


    return selector_data


# ============================================================
# Difficulty
# ============================================================

def make_difficulty(
    dev_systems,
    signal,
):

    values = []


    for item in data:

        if signal == "human":

            mean_quality = np.mean(
                [
                    item["scores"][
                        system
                    ]["human"]

                    for system
                    in dev_systems
                ]
            )


        elif signal == "metricx":

            mean_quality = np.mean(
                [
                    item["scores"][
                        system
                    ][METRIC]

                    for system
                    in dev_systems
                ]
            )


        else:

            raise ValueError(
                signal
            )


        # Higher = better quality.
        # Negating converts quality to difficulty.
        values.append(
            -float(
                mean_quality
            )
        )


    return np.asarray(
        values,
        dtype=float,
    )


def make_bins(
    difficulty,
):

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


# ============================================================
# Stratified random sampling
# ============================================================

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


    for j in range(
        remainder
    ):

        allocation[j] += 1


    selected = []


    for (
        bin_indices,
        n_take,
    ) in zip(
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


    assert (
        len(selected)
        == budget
    )


    assert (
        len(set(selected))
        == budget
    )


    return selected


# ============================================================
# Result storage
# ============================================================

rows = []

reference_rows = []

failures = []


def add_result(
    family,
    family_size,
    method,
    budget,
    trial,
    selected_ids,
    dev_systems,
    test_systems,
    reference,
):

    metrics = (
        evaluate_selection(
            selected_ids,
            test_systems,
            reference,
        )
    )


    rows.append(
        {
            "family":
                family,

            "family_size":
                family_size,

            "method":
                method,

            "budget":
                budget,

            "trial":
                trial,

            "n_dev":
                len(
                    dev_systems
                ),

            "n_test":
                len(
                    test_systems
                ),

            **metrics,
        }
    )


# ============================================================
# DiffDisc
# ============================================================

IRT_AVAILABLE = False


if RUN_DIFFDISC:

    IRT_AVAILABLE = (
        enable_cpu_irt_compatibility()
    )


# ============================================================
# Main family loop
# ============================================================

for family_index, (
    family,
    test_systems,
) in enumerate(
    eligible_families.items()
):

    test_systems = list(
        test_systems
    )


    test_set = set(
        test_systems
    )


    dev_systems = [
        system
        for system
        in ALL_SYSTEMS
        if system
        not in test_set
    ]


    family_size = len(
        test_systems
    )


    reference = (
        full_reference_scores(
            test_systems
        )
    )


    print(
        "\n"
        + "=" * 100
    )


    print(
        f"FAMILY HOLDOUT: "
        f"{family}"
    )


    print(
        "=" * 100
    )


    print(
        f"Held-out systems "
        f"({family_size}): "
        + ", ".join(
            test_systems
        )
    )


    print(
        f"Development systems: "
        f"{len(dev_systems)}"
    )


    for system in test_systems:

        reference_rows.append(
            {
                "family":
                    family,

                "system":
                    system,

                "full_common_score":
                    reference[
                        system
                    ],
            }
        )


    # --------------------------------------------------------
    # Leakage-safe data
    # --------------------------------------------------------

    selector_data = (
        make_selector_data(
            dev_systems,
            test_systems,
        )
    )


    print(
        "Leakage check: PASS"
    )


    # --------------------------------------------------------
    # Difficulty bins
    # --------------------------------------------------------

    human_bins = (
        make_bins(
            make_difficulty(
                dev_systems,
                "human",
            )
        )
    )


    metricx_bins = (
        make_bins(
            make_difficulty(
                dev_systems,
                "metricx",
            )
        )
    )


    # --------------------------------------------------------
    # Random + stratified
    # --------------------------------------------------------

    print(
        "  Random + stratified..."
    )


    for trial in range(
        N_TRIALS
    ):

        for budget in BUDGETS:

            seed = (
                BASE_SEED
                + family_index
                * 1_000_000
                + trial
                * 1000
                + budget
            )


            # Random
            rng = (
                np.random
                .default_rng(
                    seed
                )
            )


            selected_random = (
                rng.choice(
                    len(data),
                    size=budget,
                    replace=False,
                )
                .tolist()
            )


            add_result(
                family,
                family_size,
                "Random",
                budget,
                trial,
                selected_random,
                dev_systems,
                test_systems,
                reference,
            )


            # Stratified-Human
            selected_human = (
                stratified_sample(
                    human_bins,
                    budget,
                    seed,
                )
            )


            add_result(
                family,
                family_size,
                "Stratified-Human",
                budget,
                trial,
                selected_human,
                dev_systems,
                test_systems,
                reference,
            )


            # Stratified-MetricX
            selected_metricx = (
                stratified_sample(
                    metricx_bins,
                    budget,
                    seed,
                )
            )


            add_result(
                family,
                family_size,
                "Stratified-MetricX",
                budget,
                trial,
                selected_metricx,
                dev_systems,
                test_systems,
                reference,
            )


    # --------------------------------------------------------
    # Deterministic selector helper
    # --------------------------------------------------------

    def run_ranked_selector(
        name,
        **kwargs,
    ):

        ranked = (
            subset2evaluate
            .select_subset
            .basic(
                copy.deepcopy(
                    selector_data
                ),
                **kwargs,
            )
        )


        ranked_ids = [
            int(
                item["i"]
            )

            for item
            in ranked
        ]


        assert (
            len(ranked_ids)
            == len(data)
        )


        assert (
            len(set(ranked_ids))
            == len(data)
        )


        for budget in BUDGETS:

            selected_ids = (
                ranked_ids[
                    :budget
                ]
            )


            add_result(
                family,
                family_size,
                name,
                budget,
                0,
                selected_ids,
                dev_systems,
                test_systems,
                reference,
            )


            metrics = (
                evaluate_selection(
                    selected_ids,
                    test_systems,
                    reference,
                )
            )


            print(
                f"    m={budget:3d} | "
                f"MAE={metrics['mae']:.4f} | "
                f"bias={metrics['bias']:.4f} | "
                f"rho={metrics['spearman']:.4f} | "
                f"pair={metrics['pairwise']:.4f}"
            )


    # --------------------------------------------------------
    # MetricAvg
    # --------------------------------------------------------

    print(
        "  MetricAvg..."
    )


    run_ranked_selector(
        "MetricAvg",
        method="metric_avg",
        metric=METRIC,
    )


    # --------------------------------------------------------
    # MetricVar
    # --------------------------------------------------------

    print(
        "  MetricVar..."
    )


    run_ranked_selector(
        "MetricVar",
        method="metric_var",
        metric=METRIC,
    )


    # --------------------------------------------------------
    # MetricCons
    # --------------------------------------------------------

    print(
        "  MetricCons..."
    )


    run_ranked_selector(
        "MetricCons",
        method="metric_cons",
        metric=METRIC,
    )


    # --------------------------------------------------------
    # Diversity
    # --------------------------------------------------------

    print(
        "  Diversity..."
    )


    run_ranked_selector(
        "Diversity",
        method="diversity",
        metric="lm",
    )


    # --------------------------------------------------------
    # DiffDisc
    # --------------------------------------------------------

    if (
        RUN_DIFFDISC
        and IRT_AVAILABLE
    ):

        print(
            "  DiffDisc..."
        )


        try:

            run_ranked_selector(
                "DiffDisc",
                method="pyirt_diffdisc",
                model="4pl_score",
                metric=METRIC,
                epochs=IRT_EPOCHS,
                retry_on_error=False,
            )


        except Exception as exc:

            print(
                "  WARNING: "
                f"DiffDisc failed for "
                f"{family}:"
            )


            print(
                f"  {type(exc).__name__}: "
                f"{exc}"
            )


            failures.append(
                {
                    "family":
                        family,

                    "error_type":
                        type(exc).__name__,

                    "error":
                        str(exc),
                }
            )


    # --------------------------------------------------------
    # Incremental checkpoint
    # --------------------------------------------------------

    pd.DataFrame(
        rows
    ).to_csv(
        TRIAL_FILE,
        index=False,
    )


    print(
        f"{family} complete."
    )


# ============================================================
# Save references
# ============================================================

pd.DataFrame(
    reference_rows
).to_csv(
    REFERENCE_FILE,
    index=False,
)


# ============================================================
# Trial-level results
# ============================================================

trials = pd.DataFrame(
    rows
)


trials.to_csv(
    TRIAL_FILE,
    index=False,
)


# ============================================================
# Average stochastic trials within family
# ============================================================

family_summary = (
    trials
    .groupby(
        [
            "family",
            "family_size",
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
# Paired Random comparison within each family
# ============================================================

random_family = (
    family_summary[
        family_summary[
            "method"
        ]
        == "Random"
    ][
        [
            "family",
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


family_summary = (
    family_summary
    .merge(
        random_family,
        on=[
            "family",
            "budget",
        ],
        how="left",
    )
)


family_summary[
    "mae_improvement"
] = (
    family_summary[
        "random_mae"
    ]
    - family_summary[
        "mae"
    ]
)


family_summary[
    "mae_reduction_vs_random_pct"
] = (
    100.0
    * family_summary[
        "mae_improvement"
    ]
    / family_summary[
        "random_mae"
    ]
)


family_summary.to_csv(
    FAMILY_SUMMARY_FILE,
    index=False,
)


# ============================================================
# Family-balanced macro summary
#
# Each family receives equal weight regardless of how many
# systems it contains.
# ============================================================

macro = (
    family_summary
    .groupby(
        [
            "method",
            "budget",
        ],
        as_index=False,
    )
    .agg(
        num_families=(
            "family",
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

        mean_family_relative_mae_reduction_pct=(
            "mae_reduction_vs_random_pct",
            "mean",
        ),

        family_win_rate=(
            "mae_improvement",
            lambda x:
                float(
                    np.mean(
                        np.asarray(
                            x,
                            dtype=float,
                        )
                        > 0
                    )
                ),
        ),
    )
)


# ============================================================
# Macro Random MAE
# ============================================================

macro_random = (
    macro[
        macro["method"]
        == "Random"
    ][
        [
            "budget",
            "mae_mean",
        ]
    ]
    .rename(
        columns={
            "mae_mean":
                "random_macro_mae",
        }
    )
)


macro = (
    macro
    .merge(
        macro_random,
        on="budget",
        how="left",
    )
)


macro[
    "macro_mae_reduction_vs_random_pct"
] = (
    100.0
    * (
        macro[
            "random_macro_mae"
        ]
        - macro[
            "mae_mean"
        ]
    )
    / macro[
        "random_macro_mae"
    ]
)


# ============================================================
# Sort methods
# ============================================================

METHOD_ORDER = [
    "Random",
    "MetricAvg",
    "MetricVar",
    "MetricCons",
    "Diversity",
    "DiffDisc",
    "Stratified-MetricX",
    "Stratified-Human",
]


macro["method"] = (
    pd.Categorical(
        macro["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
)


macro = (
    macro
    .sort_values(
        [
            "budget",
            "method",
        ]
    )
    .reset_index(
        drop=True,
    )
)


macro.to_csv(
    MACRO_FILE,
    index=False,
)


# ============================================================
# Failures
# ============================================================

if failures:

    pd.DataFrame(
        failures
    ).to_csv(
        FAILURE_FILE,
        index=False,
    )


# ============================================================
# Print per-family results
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "PER-FAMILY SUMMARY"
)

print(
    "=" * 120
)


family_display = [
    "family",
    "family_size",
    "method",
    "budget",
    "mae",
    "bias",
    "spearman",
    "pairwise",
    "mae_reduction_vs_random_pct",
]


print(
    family_summary[
        family_display
    ]
    .sort_values(
        [
            "family",
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


# ============================================================
# Print family-balanced macro results
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "FAMILY-BALANCED MACRO SUMMARY"
)

print(
    "=" * 120
)


macro_display = [
    "method",
    "budget",
    "num_families",
    "mae_mean",
    "bias_mean",
    "spearman_mean",
    "kendall_mean",
    "pairwise_mean",
    "macro_mae_reduction_vs_random_pct",
    "family_win_rate",
]


print(
    macro[
        macro_display
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# Failure summary
# ============================================================

if failures:

    print(
        "\nDiffDisc failures:"
    )

    print(
        pd.DataFrame(
            failures
        )[
            [
                "family",
                "error_type",
            ]
        ]
        .to_string(
            index=False,
        )
    )


else:

    print(
        "\nDiffDisc failures: 0"
    )


# ============================================================
# Output paths
# ============================================================

print(
    "\nSaved:"
)

print(
    TRIAL_FILE
)

print(
    FAMILY_SUMMARY_FILE
)

print(
    MACRO_FILE
)

print(
    REFERENCE_FILE
)

if failures:

    print(
        FAILURE_FILE
    )


print(
    "\nDone."
)
