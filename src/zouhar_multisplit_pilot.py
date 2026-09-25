"""
Multi-split robustness pilot for the WMT25 strong-baseline experiment.

Dataset:
    WMT25 cs-de_DE
    189 common segments
    20 MT systems

Experiment:
    20 unique random system splits
    14 development systems
    6 completely held-out systems

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

This is a robustness pilot before scaling to 100 system splits.
"""

import copy
import json
import random
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import kendalltau, spearmanr

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

# First robustness pilot.
N_SPLITS = 20

# Same general scale as our previous multisplit experiment.
N_TRIALS = 50

N_TEST_SYSTEMS = 6

BASE_SEED = 20260925

IRT_EPOCHS = 1000

RUN_DIFFDISC = True


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
    / "zouhar_multisplit_pilot"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRIAL_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_trials.csv"
)

SPLIT_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_split_summary.csv"
)

MACRO_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_macro_summary.csv"
)

SYSTEM_SPLITS_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_system_splits.json"
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
# Load common data
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


for i, item in enumerate(data):
    item["i"] = i


ALL_SYSTEMS = sorted(
    data[0]["scores"].keys()
)


print("=" * 90)
print("WMT25 ZOUHAR MULTI-SPLIT ROBUSTNESS PILOT")
print("=" * 90)

print(
    f"\nLanguage pair: {LANGUAGE_PAIR}"
)

print(
    f"Segments:      {len(data)}"
)

print(
    f"Systems:       {len(ALL_SYSTEMS)}"
)

print(
    f"System splits: {N_SPLITS}"
)

print(
    f"Trials/split:  {N_TRIALS}"
)


assert len(data) == 189

assert len(ALL_SYSTEMS) == 20


for item in data:

    assert (
        set(item["scores"])
        == set(ALL_SYSTEMS)
    )

    assert (
        set(item["tgt"])
        == set(ALL_SYSTEMS)
    )


# ============================================================
# Generate unique 14/6 system splits
# ============================================================

split_rng = random.Random(
    BASE_SEED
)

system_splits = []

seen_test_sets = set()


while (
    len(system_splits)
    < N_SPLITS
):

    test_systems = tuple(
        sorted(
            split_rng.sample(
                ALL_SYSTEMS,
                N_TEST_SYSTEMS,
            )
        )
    )

    if test_systems in seen_test_sets:
        continue

    seen_test_sets.add(
        test_systems
    )

    test_set = set(
        test_systems
    )

    dev_systems = tuple(
        system
        for system
        in ALL_SYSTEMS
        if system not in test_set
    )

    assert len(
        dev_systems
    ) == 14

    assert len(
        test_systems
    ) == 6

    system_splits.append(
        {
            "split_id":
                len(system_splits),

            "dev_systems":
                list(dev_systems),

            "test_systems":
                list(test_systems),
        }
    )


with open(
    SYSTEM_SPLITS_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        system_splits,
        f,
        indent=2,
    )


# ============================================================
# Evaluation
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

    for a, b in combinations(
        test_systems,
        2,
    ):

        ref_diff = (
            reference[a]
            - reference[b]
        )

        if ref_diff == 0:
            continue

        pred_diff = (
            estimate[a]
            - estimate[b]
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
    test_systems,
    reference,
):

    estimate = subset_scores(
        selected_ids,
        test_systems,
    )

    ref_vector = np.asarray(
        [
            reference[s]
            for s
            in test_systems
        ],
        dtype=float,
    )

    est_vector = np.asarray(
        [
            estimate[s]
            for s
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
            np.abs(errors)
        )
    )

    # Diagnostic:
    # negative = subset underestimates system quality.
    bias = float(
        np.mean(errors)
    )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        rho = spearmanr(
            ref_vector,
            est_vector,
        ).statistic

        tau = kendalltau(
            ref_vector,
            est_vector,
        ).statistic

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
            float(
                pairwise_accuracy(
                    reference,
                    estimate,
                    test_systems,
                )
            ),
    }


# ============================================================
# Leakage-safe selector data
# ============================================================

def make_selector_data(
    dev_systems,
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

        selector_data.append(
            item
        )

    return selector_data


# ============================================================
# Difficulty bins
# ============================================================

def make_difficulty(
    dev_systems,
    signal,
):

    values = []

    for item in data:

        if signal == "human":

            quality = np.mean(
                [
                    item["scores"][
                        system
                    ]["human"]

                    for system
                    in dev_systems
                ]
            )

        elif signal == "metricx":

            quality = np.mean(
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

        # Both human scores and MetricX are oriented
        # such that larger values correspond to better
        # quality in this representation.
        #
        # Therefore negative mean quality = difficulty.
        values.append(
            -quality
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

    assert len(
        selected
    ) == budget

    assert len(
        set(selected)
    ) == budget

    return selected


# ============================================================
# Results
# ============================================================

rows = []


def add_row(
    split_id,
    method,
    budget,
    trial,
    selected_ids,
    dev_systems,
    test_systems,
    reference,
):

    metrics = evaluate_selection(
        selected_ids,
        test_systems,
        reference,
    )

    rows.append(
        {
            "split_id":
                split_id,

            "method":
                method,

            "budget":
                budget,

            "trial":
                trial,

            "n_dev":
                len(dev_systems),

            "n_test":
                len(test_systems),

            **metrics,
        }
    )


# ============================================================
# DiffDisc preparation
# ============================================================

IRT_AVAILABLE = False

if RUN_DIFFDISC:

    IRT_AVAILABLE = (
        enable_cpu_irt_compatibility()
    )


# ============================================================
# Main split loop
# ============================================================

for split_info in system_splits:

    split_id = (
        split_info[
            "split_id"
        ]
    )

    dev_systems = (
        split_info[
            "dev_systems"
        ]
    )

    test_systems = (
        split_info[
            "test_systems"
        ]
    )

    reference = (
        full_reference_scores(
            test_systems
        )
    )


    print("\n" + "=" * 90)

    print(
        f"SPLIT {split_id + 1}/{N_SPLITS}"
    )

    print("=" * 90)

    print(
        "Held out:",
        ", ".join(
            test_systems
        ),
    )


    # --------------------------------------------------------
    # Leakage-safe data
    # --------------------------------------------------------

    selector_data = (
        make_selector_data(
            dev_systems
        )
    )


    for item in selector_data:

        assert (
            set(item["scores"])
            == set(dev_systems)
        )

        assert (
            set(item["tgt"])
            == set(dev_systems)
        )


    # --------------------------------------------------------
    # Difficulty bins
    # --------------------------------------------------------

    human_bins = make_bins(
        make_difficulty(
            dev_systems,
            "human",
        )
    )

    metricx_bins = make_bins(
        make_difficulty(
            dev_systems,
            "metricx",
        )
    )


    # --------------------------------------------------------
    # Random + stratified methods
    # --------------------------------------------------------

    for trial in range(
        N_TRIALS
    ):

        for budget in BUDGETS:

            seed = (
                BASE_SEED
                + split_id * 1_000_000
                + trial * 1000
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

            add_row(
                split_id,
                "Random",
                budget,
                trial,
                selected_random,
                dev_systems,
                test_systems,
                reference,
            )


            # Human stratification
            selected_human = (
                stratified_sample(
                    human_bins,
                    budget,
                    seed,
                )
            )

            add_row(
                split_id,
                "Stratified-Human",
                budget,
                trial,
                selected_human,
                dev_systems,
                test_systems,
                reference,
            )


            # MetricX stratification
            selected_metricx = (
                stratified_sample(
                    metricx_bins,
                    budget,
                    seed,
                )
            )

            add_row(
                split_id,
                "Stratified-MetricX",
                budget,
                trial,
                selected_metricx,
                dev_systems,
                test_systems,
                reference,
            )


    # --------------------------------------------------------
    # Helper for deterministic ranked selectors
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
            int(item["i"])
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

            add_row(
                split_id,
                name,
                budget,
                0,
                selected_ids,
                dev_systems,
                test_systems,
                reference,
            )


    # --------------------------------------------------------
    # Zouhar methods
    # --------------------------------------------------------

    print(
        "  MetricAvg..."
    )

    run_ranked_selector(
        "MetricAvg",
        method="metric_avg",
        metric=METRIC,
    )


    print(
        "  MetricVar..."
    )

    run_ranked_selector(
        "MetricVar",
        method="metric_var",
        metric=METRIC,
    )


    print(
        "  MetricCons..."
    )

    run_ranked_selector(
        "MetricCons",
        method="metric_cons",
        metric=METRIC,
    )


    print(
        "  Diversity..."
    )

    run_ranked_selector(
        "Diversity",
        method="diversity",
        metric="lm",
    )


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
                f"  WARNING: DiffDisc failed "
                f"on split {split_id}:"
            )

            print(
                f"  {type(exc).__name__}: "
                f"{exc}"
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
        f"Split {split_id + 1} complete."
    )


# ============================================================
# Trial-level data
# ============================================================

trials = pd.DataFrame(
    rows
)

trials.to_csv(
    TRIAL_FILE,
    index=False,
)


# ============================================================
# First average stochastic sentence trials within each split
#
# This gives every system split equal weight.
# ============================================================

split_summary = (
    trials
    .groupby(
        [
            "split_id",
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
# Paired comparison against Random within each split
# ============================================================

random_mae = (
    split_summary[
        split_summary["method"]
        == "Random"
    ][
        [
            "split_id",
            "budget",
            "mae",
        ]
    ]
    .rename(
        columns={
            "mae":
                "random_mae",
        }
    )
)


split_summary = (
    split_summary
    .merge(
        random_mae,
        on=[
            "split_id",
            "budget",
        ],
        how="left",
    )
)


split_summary[
    "mae_improvement"
] = (
    split_summary[
        "random_mae"
    ]
    - split_summary[
        "mae"
    ]
)


split_summary[
    "mae_reduction_vs_random_pct"
] = (
    100
    * split_summary[
        "mae_improvement"
    ]
    / split_summary[
        "random_mae"
    ]
)


split_summary.to_csv(
    SPLIT_FILE,
    index=False,
)


# ============================================================
# Macro average across system splits
# ============================================================

macro = (
    split_summary
    .groupby(
        [
            "method",
            "budget",
        ],
        as_index=False,
    )
    .agg(
        num_splits=(
            "split_id",
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

        mae_reduction_vs_random_pct=(
            "mae_reduction_vs_random_pct",
            "mean",
        ),

        split_win_rate=(
            "mae_improvement",
            lambda x:
                float(
                    np.mean(
                        np.asarray(x)
                        > 0
                    )
                ),
        ),
    )
)


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


macro["method"] = pd.Categorical(
    macro["method"],
    categories=METHOD_ORDER,
    ordered=True,
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
        drop=True
    )
)


macro.to_csv(
    MACRO_FILE,
    index=False,
)


# ============================================================
# Print final results
# ============================================================

print("\n" + "=" * 120)
print("MULTI-SPLIT PILOT SUMMARY")
print("=" * 120)


display_columns = [
    "method",
    "budget",
    "num_splits",
    "mae_mean",
    "bias_mean",
    "spearman_mean",
    "kendall_mean",
    "pairwise_mean",
    "mae_reduction_vs_random_pct",
    "split_win_rate",
]


print(
    macro[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print(
    "\nSaved:"
)

print(
    TRIAL_FILE
)

print(
    SPLIT_FILE
)

print(
    MACRO_FILE
)

print(
    SYSTEM_SPLITS_FILE
)

print(
    "\nDone."
)
