"""
Strong-baseline comparison for future-proof MT evaluation.

Dataset:
    WMT25 cs-de_DE
    189 segments with complete human + MetricX coverage
    20 MT systems

Experiment:
    14 development systems are visible during subset construction.
    6 held-out systems are completely invisible during selection.

Compared methods:
    Random

    Zouhar et al.:
        MetricAvg
        MetricVar
        MetricCons
        Diversity
        DiffDisc

    Ours:
        Difficulty-Stratified (MetricX-dev)
        Difficulty-Stratified (Human-dev)

Evaluation:
    MAE
    Spearman rho
    Kendall tau
    Pairwise ranking accuracy

IMPORTANT:
    Full-test reference scores are computed over the SAME 189-segment
    common evaluation universe for every method.
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

# Random and our stratified methods are stochastic.
N_TRIALS = 1000

BASE_SEED = 20260925

# DiffDisc / IRT
IRT_EPOCHS = 1000


# ============================================================
# Same 14/6 pilot split used in our earlier unseen-system study
# ============================================================

DEV_SYSTEMS = [
    "Algharb",
    "Claude-4",
    "CommandA",
    "CommandA-MT",
    "DeepSeek-V3",
    "GPT-4.1",
    "Gemma-3-12B",
    "Gemma-3-27B",
    "Mistral-Medium",
    "SRPOL",
    "Shy",
    "TowerPlus-9B",
    "Wenyiil",
    "Yolu",
]

TEST_SYSTEMS = [
    "CUNI-MH-v2",
    "GemTrans",
    "Gemini-2.5-Pro",
    "IRB-MT",
    "Laniqo",
    "UvA-MT",
]


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
    / "zouhar_unseen_system_benchmark"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRIAL_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_trials.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_summary.csv"
)

REFERENCE_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_heldout_reference_scores.csv"
)

SELECTION_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_deterministic_selections.json"
)


# ============================================================
# CPU patch for released DiffDisc implementation
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

            kwargs["device"] = (
                "cpu"
            )

        return original_train(
            self,
            *args,
            **kwargs,
        )


    train_cpu._futureproof_cpu_wrapper = (
        True
    )

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


print("=" * 80)
print("WMT25 UNSEEN-SYSTEM STRONG-BASELINE BENCHMARK")
print("=" * 80)

print(
    f"\nLanguage pair: "
    f"{LANGUAGE_PAIR}"
)

print(
    f"Segments: "
    f"{len(data)}"
)


# ============================================================
# Validate item IDs
# ============================================================

for i, item in enumerate(
    data
):

    item["i"] = i


# ============================================================
# Validate system universe
# ============================================================

ALL_SYSTEMS = sorted(
    data[0]["scores"].keys()
)

print(
    f"Systems: "
    f"{len(ALL_SYSTEMS)}"
)

print(
    f"Development systems: "
    f"{len(DEV_SYSTEMS)}"
)

print(
    f"Held-out systems: "
    f"{len(TEST_SYSTEMS)}"
)


assert (
    set(DEV_SYSTEMS)
    .isdisjoint(
        TEST_SYSTEMS
    )
)

assert (
    set(DEV_SYSTEMS)
    | set(TEST_SYSTEMS)
    == set(ALL_SYSTEMS)
), (
    "Dev + test systems do not match "
    "the 20 systems in the adapter."
)


for item in data:

    assert (
        set(item["scores"])
        == set(ALL_SYSTEMS)
    )

    assert (
        set(item["tgt"])
        == set(ALL_SYSTEMS)
    )


print(
    "System split validation: PASS"
)


# ============================================================
# Full-test held-out reference scores
#
# IMPORTANT:
#
# These are reference scores over the SAME 189 sentences
# available to every method in this comparison.
# ============================================================

FULL_REFERENCE = {}

for system in TEST_SYSTEMS:

    FULL_REFERENCE[
        system
    ] = float(
        np.mean(
            [
                item["scores"][
                    system
                ]["human"]

                for item in data
            ]
        )
    )


reference_df = pd.DataFrame(
    {
        "system":
            TEST_SYSTEMS,

        "full_common_score":
            [
                FULL_REFERENCE[
                    system
                ]
                for system
                in TEST_SYSTEMS
            ],
    }
)

reference_df = (
    reference_df
    .sort_values(
        "full_common_score",
        ascending=False,
    )
)

reference_df.to_csv(
    REFERENCE_FILE,
    index=False,
)


print(
    "\nHeld-out reference ranking "
    "(189 common segments):"
)

print(
    reference_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# Evaluation functions
# ============================================================

def estimate_scores(
    selected_ids,
):

    selected_ids = list(
        selected_ids
    )

    estimates = {}

    for system in TEST_SYSTEMS:

        estimates[
            system
        ] = float(
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

    return estimates


def pairwise_accuracy(
    reference,
    estimate,
):

    correct = 0
    total = 0

    for system_a, system_b in combinations(
        TEST_SYSTEMS,
        2,
    ):

        ref_diff = (
            reference[system_a]
            - reference[system_b]
        )

        # Ignore true full-test ties.
        if ref_diff == 0:
            continue

        pred_diff = (
            estimate[system_a]
            - estimate[system_b]
        )

        total += 1

        if (
            np.sign(ref_diff)
            == np.sign(pred_diff)
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
):

    estimate = estimate_scores(
        selected_ids
    )


    reference_vector = np.asarray(
        [
            FULL_REFERENCE[
                system
            ]
            for system
            in TEST_SYSTEMS
        ],
        dtype=float,
    )


    estimate_vector = np.asarray(
        [
            estimate[
                system
            ]
            for system
            in TEST_SYSTEMS
        ],
        dtype=float,
    )


    mae = float(
        np.mean(
            np.abs(
                estimate_vector
                - reference_vector
            )
        )
    )


    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        rho = (
            spearmanr(
                reference_vector,
                estimate_vector,
            )
            .statistic
        )

        tau = (
            kendalltau(
                reference_vector,
                estimate_vector,
            )
            .statistic
        )


    pairwise = (
        pairwise_accuracy(
            FULL_REFERENCE,
            estimate,
        )
    )


    return {
        "mae":
            float(mae),

        "spearman":
            float(rho),

        "kendall":
            float(tau),

        "pairwise":
            float(pairwise),
    }


# ============================================================
# Build leakage-safe selector dataset
#
# This is the critical step.
#
# Zouhar selectors see ONLY the 14 development systems.
#
# Held-out system scores and translations are physically
# removed before sentence selection.
# ============================================================

def make_dev_selector_data():

    selector_data = []

    for original_item in data:

        item = copy.deepcopy(
            original_item
        )

        item["scores"] = {
            system:
                item["scores"][
                    system
                ]

            for system
            in DEV_SYSTEMS
        }

        item["tgt"] = {
            system:
                item["tgt"][
                    system
                ]

            for system
            in DEV_SYSTEMS
        }

        selector_data.append(
            item
        )


    # Explicit leakage checks
    for item in selector_data:

        assert (
            set(item["scores"])
            == set(DEV_SYSTEMS)
        )

        assert (
            set(item["tgt"])
            == set(DEV_SYSTEMS)
        )

        assert (
            set(TEST_SYSTEMS)
            .isdisjoint(
                item["scores"]
            )
        )

        assert (
            set(TEST_SYSTEMS)
            .isdisjoint(
                item["tgt"]
            )
        )


    return selector_data


selector_data = (
    make_dev_selector_data()
)

print(
    "\nHeld-out leakage check: PASS"
)


# ============================================================
# Difficulty values for our stratified selectors
# ============================================================

human_difficulty = np.asarray(
    [
        -np.mean(
            [
                item["scores"][
                    system
                ]["human"]

                for system
                in DEV_SYSTEMS
            ]
        )

        for item in data
    ],
    dtype=float,
)


metricx_difficulty = np.asarray(
    [
        -np.mean(
            [
                item["scores"][
                    system
                ][METRIC]

                for system
                in DEV_SYSTEMS
            ]
        )

        for item in data
    ],
    dtype=float,
)


# ============================================================
# Empirical quantile strata
#
# Sorting and splitting into equally-sized blocks gives
# empirical quantile strata.
#
# Ties are broken stably by original sentence order.
# ============================================================

def make_quantile_bins(
    difficulty,
    n_bins=N_BINS,
):

    order = np.argsort(
        difficulty,
        kind="stable",
    )

    bins = [
        np.asarray(
            chunk,
            dtype=int,
        )

        for chunk
        in np.array_split(
            order,
            n_bins,
        )
    ]

    return bins


HUMAN_BINS = (
    make_quantile_bins(
        human_difficulty
    )
)

METRICX_BINS = (
    make_quantile_bins(
        metricx_difficulty
    )
)


print(
    "\nHuman-difficulty bin sizes:",
    [
        len(x)
        for x in HUMAN_BINS
    ],
)

print(
    "MetricX-difficulty bin sizes:",
    [
        len(x)
        for x in METRICX_BINS
    ],
)


# ============================================================
# Stratified random sampler
# ============================================================

def stratified_sample(
    bins,
    budget,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    n_bins = len(
        bins
    )

    base = (
        budget
        // n_bins
    )

    remainder = (
        budget
        % n_bins
    )

    allocation = [
        base
        for _ in bins
    ]


    # Our current budgets are all divisible by 5,
    # but keep this general.
    for j in range(
        remainder
    ):

        allocation[j] += 1


    selected = []

    for bin_indices, n_take in zip(
        bins,
        allocation,
    ):

        if (
            n_take
            > len(bin_indices)
        ):

            raise ValueError(
                "Requested more samples "
                "than available in a stratum."
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
        len(set(selected))
        == budget
    )


    return selected


# ============================================================
# Store all evaluation rows
# ============================================================

rows = []

deterministic_selections = {}


def add_result(
    method,
    budget,
    trial,
    selected_ids,
):

    metrics = (
        evaluate_selection(
            selected_ids
        )
    )

    rows.append(
        {
            "method":
                method,

            "budget":
                budget,

            "trial":
                trial,

            **metrics,
        }
    )


# ============================================================
# 1. Random
# ============================================================

print("\n" + "=" * 80)
print("RANDOM")
print("=" * 80)


for trial in range(
    N_TRIALS
):

    for budget in BUDGETS:

        seed = (
            BASE_SEED
            + trial * 1000
            + budget
        )

        rng = (
            np.random
            .default_rng(
                seed
            )
        )

        selected_ids = (
            rng.choice(
                len(data),
                size=budget,
                replace=False,
            )
            .tolist()
        )

        add_result(
            method="Random",
            budget=budget,
            trial=trial,
            selected_ids=selected_ids,
        )


print(
    f"Completed "
    f"{N_TRIALS} Random trials."
)


# ============================================================
# 2. Difficulty-Stratified Human
# ============================================================

print("\n" + "=" * 80)
print("DIFFICULTY-STRATIFIED HUMAN")
print("=" * 80)


for trial in range(
    N_TRIALS
):

    for budget in BUDGETS:

        seed = (
            BASE_SEED
            + trial * 1000
            + budget
        )

        selected_ids = (
            stratified_sample(
                HUMAN_BINS,
                budget,
                seed,
            )
        )

        add_result(
            method="Stratified-Human",
            budget=budget,
            trial=trial,
            selected_ids=selected_ids,
        )


print(
    f"Completed "
    f"{N_TRIALS} Human-stratified trials."
)


# ============================================================
# 3. Difficulty-Stratified MetricX
# ============================================================

print("\n" + "=" * 80)
print("DIFFICULTY-STRATIFIED METRICX")
print("=" * 80)


for trial in range(
    N_TRIALS
):

    for budget in BUDGETS:

        seed = (
            BASE_SEED
            + trial * 1000
            + budget
        )

        selected_ids = (
            stratified_sample(
                METRICX_BINS,
                budget,
                seed,
            )
        )

        add_result(
            method="Stratified-MetricX",
            budget=budget,
            trial=trial,
            selected_ids=selected_ids,
        )


print(
    f"Completed "
    f"{N_TRIALS} MetricX-stratified trials."
)


# ============================================================
# Helper for deterministic Zouhar selectors
# ============================================================

def run_zouhar_selector(
    name,
    **kwargs,
):

    print("\n" + "=" * 80)

    print(
        f"ZOUHAR SELECTOR: {name}"
    )

    print("=" * 80)


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
        int(item["i"])
        for item in ranked
    ]


    assert (
        len(ranked_ids)
        == len(data)
    )

    assert (
        len(set(ranked_ids))
        == len(data)
    )


    deterministic_selections[
        name
    ] = {}


    for budget in BUDGETS:

        selected_ids = (
            ranked_ids[
                :budget
            ]
        )

        deterministic_selections[
            name
        ][str(budget)] = (
            selected_ids
        )

        add_result(
            method=name,
            budget=budget,
            trial=0,
            selected_ids=selected_ids,
        )


        metrics = (
            evaluate_selection(
                selected_ids
            )
        )

        print(
            f"m={budget:3d} | "
            f"MAE={metrics['mae']:.4f} | "
            f"rho={metrics['spearman']:.4f} | "
            f"tau={metrics['kendall']:.4f} | "
            f"pair={metrics['pairwise']:.4f}"
        )


# ============================================================
# 4. MetricAvg
# ============================================================

run_zouhar_selector(
    "MetricAvg",
    method="metric_avg",
    metric=METRIC,
)


# ============================================================
# 5. MetricVar
# ============================================================

run_zouhar_selector(
    "MetricVar",
    method="metric_var",
    metric=METRIC,
)


# ============================================================
# 6. MetricCons
# ============================================================

run_zouhar_selector(
    "MetricCons",
    method="metric_cons",
    metric=METRIC,
)


# ============================================================
# 7. Diversity
# ============================================================

run_zouhar_selector(
    "Diversity",
    method="diversity",
    metric="lm",
)


# ============================================================
# 8. DiffDisc
# ============================================================

IRT_AVAILABLE = (
    enable_cpu_irt_compatibility()
)


if IRT_AVAILABLE:

    try:

        run_zouhar_selector(
            "DiffDisc",
            method="pyirt_diffdisc",
            model="4pl_score",
            metric=METRIC,
            epochs=IRT_EPOCHS,
            retry_on_error=False,
        )

    except Exception as exc:

        print(
            "\nDiffDisc failed:"
        )

        print(
            type(exc).__name__,
            exc,
        )

else:

    print(
        "\nDiffDisc skipped."
    )


# ============================================================
# Save trial-level results
# ============================================================

trials_df = pd.DataFrame(
    rows
)

trials_df.to_csv(
    TRIAL_FILE,
    index=False,
)


# ============================================================
# Summarize
# ============================================================

summary = (
    trials_df
    .groupby(
        [
            "method",
            "budget",
        ],
        as_index=False,
    )
    .agg(
        n=(
            "mae",
            "size",
        ),

        mae_mean=(
            "mae",
            "mean",
        ),

        mae_std=(
            "mae",
            "std",
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
    )
)


summary["mae_std"] = (
    summary["mae_std"]
    .fillna(0.0)
)


# ============================================================
# Compare MAE against mean Random baseline
# ============================================================

random_reference = (
    summary[
        summary["method"]
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
                "random_mae",
        }
    )
)


summary = summary.merge(
    random_reference,
    on="budget",
    how="left",
)


summary[
    "mae_reduction_vs_random_pct"
] = (
    100.0
    * (
        summary["random_mae"]
        - summary["mae_mean"]
    )
    / summary["random_mae"]
)


# ============================================================
# Method ordering
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


summary["method"] = (
    pd.Categorical(
        summary["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
)


summary = (
    summary
    .sort_values(
        [
            "budget",
            "method",
        ]
    )
    .reset_index(
        drop=True
    )
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# Save deterministic selections
# ============================================================

with open(
    SELECTION_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "language_pair":
                LANGUAGE_PAIR,

            "num_segments":
                len(data),

            "dev_systems":
                DEV_SYSTEMS,

            "test_systems":
                TEST_SYSTEMS,

            "selections":
                deterministic_selections,
        },
        f,
        indent=2,
    )


# ============================================================
# Print final summary
# ============================================================

print("\n" + "=" * 100)
print("FINAL UNSEEN-SYSTEM SUMMARY")
print("=" * 100)


display_columns = [
    "method",
    "budget",
    "mae_mean",
    "spearman_mean",
    "kendall_mean",
    "pairwise_mean",
    "mae_reduction_vs_random_pct",
]


print(
    summary[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print(
    "\nSaved trial results:"
)

print(
    TRIAL_FILE
)

print(
    "\nSaved summary:"
)

print(
    SUMMARY_FILE
)

print(
    "\nSaved held-out reference scores:"
)

print(
    REFERENCE_FILE
)

print(
    "\nSaved deterministic selections:"
)

print(
    SELECTION_FILE
)

print(
    "\nDone."
)
