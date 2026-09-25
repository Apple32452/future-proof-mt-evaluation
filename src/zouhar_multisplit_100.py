"""
Final 100-split strong-baseline experiment.

Dataset:
    WMT25 cs-de_DE
    189 common human + MetricX-covered segments
    20 MT systems

Per split:
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

Final inference:
    paired MAE improvement vs Random across system splits
    bootstrap 95% CI
    one-sided paired Wilcoxon test
    split win rate

The script checkpoints after every completed system split and can resume.
"""

import copy
import json
import random
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    kendalltau,
    spearmanr,
    wilcoxon,
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

N_SPLITS = 100

# Sentence-sampling trials for stochastic methods:
#   Random
#   Stratified-MetricX
#   Stratified-Human
N_TRIALS = 50

N_TEST_SYSTEMS = 6

BASE_SEED = 20260925

IRT_EPOCHS = 1000

RUN_DIFFDISC = True

# DiffDisc is deterministic in the released implementation.
# If it numerically diverges, record that split as a failure
# rather than retrying indefinitely.
DIFFDISC_MAX_ATTEMPTS = 1

BOOTSTRAP_REPS = 10000


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
    / "zouhar_multisplit_100"
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

SIGNIFICANCE_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_paired_significance.csv"
)

SYSTEM_SPLITS_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_system_splits.json"
)

FAILURE_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_diffdisc_failures.csv"
)

PROGRESS_FILE = (
    OUTPUT_DIR
    / "progress.json"
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
        f"Missing adapter output:\n{DATA_FILE}"
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
    "WMT25 ZOUHAR 100-SPLIT "
    "UNSEEN-SYSTEM BENCHMARK"
)

print("=" * 100)

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


assert (
    len(data)
    == 189
), (
    f"Expected 189 common segments, "
    f"found {len(data)}."
)

assert (
    len(ALL_SYSTEMS)
    == 20
), (
    f"Expected 20 systems, "
    f"found {len(ALL_SYSTEMS)}."
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
# Generate / load fixed system splits
# ============================================================

def generate_system_splits():

    rng = random.Random(
        BASE_SEED
    )

    splits = []

    seen = set()


    while len(splits) < N_SPLITS:

        test_systems = tuple(
            sorted(
                rng.sample(
                    ALL_SYSTEMS,
                    N_TEST_SYSTEMS,
                )
            )
        )

        if test_systems in seen:
            continue

        seen.add(
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

        assert (
            len(dev_systems)
            == 14
        )

        assert (
            len(test_systems)
            == 6
        )

        splits.append(
            {
                "split_id":
                    len(splits),

                "dev_systems":
                    dev_systems,

                "test_systems":
                    list(
                        test_systems
                    ),
            }
        )


    return splits


if SYSTEM_SPLITS_FILE.exists():

    with open(
        SYSTEM_SPLITS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        system_splits = json.load(
            f
        )

    if (
        len(system_splits)
        != N_SPLITS
    ):

        raise RuntimeError(
            "Existing system split file does not "
            f"contain {N_SPLITS} splits.\n"
            f"Delete {SYSTEM_SPLITS_FILE} and rerun."
        )

else:

    system_splits = (
        generate_system_splits()
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


print(
    "System splits: READY"
)


# ============================================================
# Resume support
# ============================================================

if (
    TRIAL_FILE.exists()
    and PROGRESS_FILE.exists()
):

    previous_trials = (
        pd.read_csv(
            TRIAL_FILE
        )
    )

    rows = (
        previous_trials
        .to_dict(
            orient="records"
        )
    )


    with open(
        PROGRESS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        progress = json.load(
            f
        )


    completed_splits = set(
        progress.get(
            "completed_splits",
            [],
        )
    )


    print(
        f"Resume enabled: "
        f"{len(completed_splits)} "
        f"splits already complete."
    )

else:

    rows = []

    completed_splits = set()


if FAILURE_FILE.exists():

    failures_df = pd.read_csv(
        FAILURE_FILE
    )

    failures = (
        failures_df
        .to_dict(
            orient="records"
        )
    )

else:

    failures = []


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
            reference[system_a]
            - reference[system_b]
        )


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
    test_systems,
    reference,
):

    estimate = subset_scores(
        selected_ids,
        test_systems,
    )


    reference_vector = np.asarray(
        [
            reference[
                system
            ]

            for system
            in test_systems
        ],
        dtype=float,
    )


    estimate_vector = np.asarray(
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


        # Explicit anti-leakage checks.
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
# Difficulty and quantile bins
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
                f"Unknown signal: {signal}"
            )


        # Higher human score / less-negative MetricX
        # corresponds to higher translation quality.
        # Negative mean quality is therefore used as
        # a difficulty ordering.
        values.append(
            -float(
                quality
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


    bins = [
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


    return bins


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

        if (
            n_take
            > len(bin_indices)
        ):

            raise ValueError(
                "Budget exceeds one of "
                "the available stratum sizes."
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
# Add result row
# ============================================================

def add_result(
    split_id,
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
# DiffDisc setup
# ============================================================

IRT_AVAILABLE = False


if RUN_DIFFDISC:

    IRT_AVAILABLE = (
        enable_cpu_irt_compatibility()
    )


# ============================================================
# Main 100-split experiment
# ============================================================

for split_info in system_splits:

    split_id = int(
        split_info[
            "split_id"
        ]
    )


    if (
        split_id
        in completed_splits
    ):

        print(
            f"Skipping completed split "
            f"{split_id + 1}/{N_SPLITS}"
        )

        continue


    dev_systems = list(
        split_info[
            "dev_systems"
        ]
    )


    test_systems = list(
        split_info[
            "test_systems"
        ]
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
        f"SPLIT "
        f"{split_id + 1}/{N_SPLITS}"
    )


    print(
        "=" * 100
    )


    print(
        "Held out:",
        ", ".join(
            test_systems
        ),
    )


    # --------------------------------------------------------
    # Leakage-safe selector data
    # --------------------------------------------------------

    selector_data = (
        make_selector_data(
            dev_systems,
            test_systems,
        )
    )


    # --------------------------------------------------------
    # Difficulty strata
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
    # Random + stratified methods
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
                + split_id
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
                split_id,
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
                split_id,
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
    # Ranked selector helper
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
    # Zouhar et al. selectors
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


        diffdisc_success = False


        for attempt in range(
            DIFFDISC_MAX_ATTEMPTS
        ):

            try:

                run_ranked_selector(
                    "DiffDisc",
                    method="pyirt_diffdisc",
                    model="4pl_score",
                    metric=METRIC,
                    epochs=IRT_EPOCHS,
                    retry_on_error=False,
                )


                diffdisc_success = True

                break


            except Exception as exc:

                error_text = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )


                print(
                    f"  DiffDisc attempt "
                    f"{attempt + 1}/"
                    f"{DIFFDISC_MAX_ATTEMPTS} "
                    f"failed."
                )


                print(
                    f"  {error_text}"
                )


                failures.append(
                    {
                        "split_id":
                            split_id,

                        "attempt":
                            attempt + 1,

                        "error_type":
                            type(exc).__name__,

                        "error":
                            str(exc),
                    }
                )


        if not diffdisc_success:

            print(
                f"  WARNING: DiffDisc unavailable "
                f"for split {split_id}."
            )


    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    trial_df = (
        pd.DataFrame(
            rows
        )
        .drop_duplicates(
            subset=[
                "split_id",
                "method",
                "budget",
                "trial",
            ],
            keep="last",
        )
        .sort_values(
            [
                "split_id",
                "budget",
                "method",
                "trial",
            ]
        )
    )


    trial_df.to_csv(
        TRIAL_FILE,
        index=False,
    )


    if failures:

        pd.DataFrame(
            failures
        ).to_csv(
            FAILURE_FILE,
            index=False,
        )


    completed_splits.add(
        split_id
    )


    with open(
        PROGRESS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "completed_splits":
                    sorted(
                        completed_splits
                    ),

                "n_splits":
                    N_SPLITS,

                "n_trials":
                    N_TRIALS,
            },
            f,
            indent=2,
        )


    print(
        f"Split {split_id + 1} complete. "
        f"Checkpoint saved."
    )


# ============================================================
# Final trial-level table
# ============================================================

trials = (
    pd.DataFrame(
        rows
    )
    .drop_duplicates(
        subset=[
            "split_id",
            "method",
            "budget",
            "trial",
        ],
        keep="last",
    )
)


trials.to_csv(
    TRIAL_FILE,
    index=False,
)


# ============================================================
# Average sentence trials within each system split
#
# Each held-out system split gets equal weight.
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
# Paired Random MAE for each system split
# ============================================================

random_split = (
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
                "random_mae"
        }
    )
)


split_summary = (
    split_summary
    .merge(
        random_split,
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
    100.0
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
# Macro averages across system splits
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

        mae_std=(
            "mae",
            "std",
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

        mae_improvement_mean=(
            "mae_improvement",
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
# Method order
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
# Paired bootstrap + Wilcoxon significance
# ============================================================

significance_rows = []


bootstrap_rng = (
    np.random
    .default_rng(
        BASE_SEED
        + 999999
    )
)


for method in METHOD_ORDER:

    if method == "Random":
        continue


    for budget in BUDGETS:

        local = (
            split_summary[
                (
                    split_summary[
                        "method"
                    ]
                    == method
                )
                &
                (
                    split_summary[
                        "budget"
                    ]
                    == budget
                )
            ]
            .sort_values(
                "split_id"
            )
        )


        deltas = (
            local[
                "mae_improvement"
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )


        if len(deltas) == 0:
            continue


        bootstrap_means = np.empty(
            BOOTSTRAP_REPS,
            dtype=float,
        )


        n = len(
            deltas
        )


        for b in range(
            BOOTSTRAP_REPS
        ):

            sample = (
                bootstrap_rng.choice(
                    deltas,
                    size=n,
                    replace=True,
                )
            )


            bootstrap_means[b] = (
                np.mean(
                    sample
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


        # Positive delta:
        # method MAE < Random MAE.
        if np.allclose(
            deltas,
            0.0,
        ):

            wilcoxon_p = 1.0

        else:

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
                "method":
                    method,

                "budget":
                    budget,

                "num_splits":
                    n,

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

                "split_win_rate":
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


significance["method"] = (
    pd.Categorical(
        significance["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
)


significance = (
    significance
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


significance.to_csv(
    SIGNIFICANCE_FILE,
    index=False,
)


# ============================================================
# Final output
# ============================================================

print(
    "\n"
    + "=" * 125
)

print(
    "FINAL 100-SPLIT SUMMARY"
)

print(
    "=" * 125
)


DISPLAY_COLUMNS = [
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
        DISPLAY_COLUMNS
    ].to_string(
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
    "PAIRED MAE SIGNIFICANCE VS RANDOM"
)

print(
    "=" * 125
)


SIG_COLUMNS = [
    "method",
    "budget",
    "num_splits",
    "mean_absolute_mae_improvement",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "wilcoxon_one_sided_p",
    "split_win_rate",
]


print(
    significance[
        SIG_COLUMNS
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
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
                "split_id",
                "attempt",
                "error_type",
            ]
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "\nDiffDisc failures: 0"
    )


# ============================================================
# Output files
# ============================================================

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
    SIGNIFICANCE_FILE
)

print(
    SYSTEM_SPLITS_FILE
)

if failures:

    print(
        FAILURE_FILE
    )

print(
    "\nDone."
)
