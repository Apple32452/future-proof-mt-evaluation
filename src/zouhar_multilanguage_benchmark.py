"""
Final multilingual strong-baseline benchmark.

WMT25:
    14 language pairs
    common human + MetricX-covered subsets
    speech excluded because MetricX coverage is unavailable there

Per language:
    20 unique unseen-system splits
    6 held-out systems per split
    remaining systems used for subset construction

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

Stochastic methods:
    30 sentence-sampling trials per system split

Evaluation:
    MAE
    signed bias
    Spearman rho
    Kendall tau
    pairwise ranking accuracy

Final statistical unit:
    language pair

Final inference:
    paired MAE improvement vs Random across 14 languages
    95% language-level bootstrap CI
    one-sided paired Wilcoxon test

The script checkpoints after every language/system split and can resume.
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

METRIC = "MetricX-24-Hybrid-XL"

BUDGETS = [
    10,
    25,
    50,
    100,
]

N_BINS = 5

N_SYSTEM_SPLITS = 20

N_TEST_SYSTEMS = 6

N_STOCHASTIC_TRIALS = 30

BASE_SEED = 20260925

IRT_EPOCHS = 1000

RUN_DIFFDISC = True

BOOTSTRAP_REPS = 10000


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "results"
    / "wmt25_subset2evaluate_all"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "zouhar_multilanguage_benchmark"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRIAL_FILE = (
    OUTPUT_DIR
    / "multilanguage_trials.csv"
)

SPLIT_SUMMARY_FILE = (
    OUTPUT_DIR
    / "multilanguage_split_summary.csv"
)

LANGUAGE_SUMMARY_FILE = (
    OUTPUT_DIR
    / "multilanguage_language_summary.csv"
)

MACRO_FILE = (
    OUTPUT_DIR
    / "multilanguage_macro_summary.csv"
)

SIGNIFICANCE_FILE = (
    OUTPUT_DIR
    / "multilanguage_paired_significance.csv"
)

SYSTEM_SPLITS_FILE = (
    OUTPUT_DIR
    / "multilanguage_system_splits.json"
)

FAILURE_FILE = (
    OUTPUT_DIR
    / "diffdisc_failures.csv"
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
# Load all language datasets
# ============================================================

data_by_language = {}

systems_by_language = {}


print(
    "=" * 110
)

print(
    "WMT25 MULTILINGUAL STRONG-BASELINE BENCHMARK"
)

print(
    "=" * 110
)


for language_pair in LANGUAGE_PAIRS:

    path = (
        DATA_DIR
        / (
            f"{language_pair}"
            "_subset2evaluate.jsonl"
        )
    )


    if not path.exists():

        raise FileNotFoundError(
            f"Missing dataset:\n{path}"
        )


    data = []


    with open(
        path,
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


    systems = sorted(
        data[0][
            "scores"
        ].keys()
    )


    assert (
        len(data)
        >= max(BUDGETS)
    )


    assert (
        len(systems)
        > N_TEST_SYSTEMS
    )


    expected_systems = set(
        systems
    )


    for item in data:

        assert (
            set(
                item["scores"].keys()
            )
            == expected_systems
        )

        assert (
            set(
                item["tgt"].keys()
            )
            == expected_systems
        )


    data_by_language[
        language_pair
    ] = data


    systems_by_language[
        language_pair
    ] = systems


    print(
        f"{language_pair:15s} | "
        f"segments={len(data):3d} | "
        f"systems={len(systems):2d}"
    )


print(
    "\nDataset validation: PASS"
)


# ============================================================
# Generate fixed system splits
# ============================================================

def generate_system_splits():

    all_splits = {}


    for language_index, language_pair in enumerate(
        LANGUAGE_PAIRS
    ):

        systems = (
            systems_by_language[
                language_pair
            ]
        )


        rng = random.Random(
            BASE_SEED
            + language_index
            * 100_000
        )


        splits = []

        seen = set()


        while (
            len(splits)
            < N_SYSTEM_SPLITS
        ):

            test_systems = tuple(
                sorted(
                    rng.sample(
                        systems,
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
                in systems
                if system
                not in test_set
            ]


            assert (
                len(test_systems)
                == N_TEST_SYSTEMS
            )


            assert (
                set(dev_systems)
                .isdisjoint(
                    test_systems
                )
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


        all_splits[
            language_pair
        ] = splits


    return all_splits


if SYSTEM_SPLITS_FILE.exists():

    with open(
        SYSTEM_SPLITS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        system_splits = json.load(
            f
        )


    for language_pair in LANGUAGE_PAIRS:

        assert (
            language_pair
            in system_splits
        )

        assert (
            len(
                system_splits[
                    language_pair
                ]
            )
            == N_SYSTEM_SPLITS
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


    completed_keys = set(
        progress.get(
            "completed_language_splits",
            [],
        )
    )


    print(
        f"Resume enabled: "
        f"{len(completed_keys)} / "
        f"{len(LANGUAGE_PAIRS) * N_SYSTEM_SPLITS} "
        f"language/splits complete."
    )


else:

    rows = []

    completed_keys = set()


if FAILURE_FILE.exists():

    failures = (
        pd.read_csv(
            FAILURE_FILE
        )
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
    data,
    test_systems,
):

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
        in test_systems
    }


def subset_scores(
    data,
    selected_ids,
    test_systems,
):

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
    data,
    selected_ids,
    test_systems,
    reference,
):

    estimate = (
        subset_scores(
            data,
            selected_ids,
            test_systems,
        )
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
    data,
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
                item[
                    "scores"
                ][
                    system
                ]

            for system
            in dev_systems
        }


        item["tgt"] = {
            system:
                item[
                    "tgt"
                ][
                    system
                ]

            for system
            in dev_systems
        }


        assert (
            set(
                item[
                    "scores"
                ].keys()
            )
            == set(
                dev_systems
            )
        )


        assert (
            set(
                test_systems
            )
            .isdisjoint(
                item[
                    "scores"
                ].keys()
            )
        )


        assert (
            set(
                test_systems
            )
            .isdisjoint(
                item[
                    "tgt"
                ].keys()
            )
        )


        selector_data.append(
            item
        )


    return selector_data


# ============================================================
# Difficulty / stratification
# ============================================================

def make_difficulty(
    data,
    dev_systems,
    signal,
):

    difficulty = []


    for item in data:

        if signal == "human":

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
                    in dev_systems
                ]
            )


        elif signal == "metricx":

            mean_quality = np.mean(
                [
                    item[
                        "scores"
                    ][
                        system
                    ][
                        METRIC
                    ]

                    for system
                    in dev_systems
                ]
            )


        else:

            raise ValueError(
                signal
            )


        # Larger human scores and less-negative MetricX
        # correspond to higher translation quality.
        # Negation converts quality to difficulty.
        difficulty.append(
            -float(
                mean_quality
            )
        )


    return np.asarray(
        difficulty,
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
# Result helper
# ============================================================

def add_result(
    language_pair,
    split_id,
    method,
    budget,
    trial,
    selected_ids,
    data,
    dev_systems,
    test_systems,
    reference,
):

    metrics = (
        evaluate_selection(
            data,
            selected_ids,
            test_systems,
            reference,
        )
    )


    rows.append(
        {
            "language_pair":
                language_pair,

            "split_id":
                split_id,

            "method":
                method,

            "budget":
                budget,

            "trial":
                trial,

            "num_segments":
                len(
                    data
                ),

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
# DiffDisc setup
# ============================================================

IRT_AVAILABLE = False


if RUN_DIFFDISC:

    IRT_AVAILABLE = (
        enable_cpu_irt_compatibility()
    )


# ============================================================
# Main multilingual experiment
# ============================================================

total_settings = (
    len(
        LANGUAGE_PAIRS
    )
    * N_SYSTEM_SPLITS
)


for language_index, language_pair in enumerate(
    LANGUAGE_PAIRS
):

    data = (
        data_by_language[
            language_pair
        ]
    )


    all_systems = (
        systems_by_language[
            language_pair
        ]
    )


    print(
        "\n"
        + "#" * 110
    )


    print(
        f"LANGUAGE "
        f"{language_index + 1}/"
        f"{len(LANGUAGE_PAIRS)}: "
        f"{language_pair}"
    )


    print(
        "#" * 110
    )


    for split_info in system_splits[
        language_pair
    ]:

        split_id = int(
            split_info[
                "split_id"
            ]
        )


        progress_key = (
            f"{language_pair}"
            f"|{split_id}"
        )


        if (
            progress_key
            in completed_keys
        ):

            print(
                f"Skipping completed "
                f"{language_pair} "
                f"split {split_id + 1}/"
                f"{N_SYSTEM_SPLITS}"
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


        assert (
            set(
                dev_systems
            )
            .isdisjoint(
                test_systems
            )
        )


        assert (
            set(
                dev_systems
            )
            | set(
                test_systems
            )
            == set(
                all_systems
            )
        )


        reference = (
            full_reference_scores(
                data,
                test_systems,
            )
        )


        print(
            "\n"
            + "=" * 100
        )


        print(
            f"{language_pair} | "
            f"SPLIT "
            f"{split_id + 1}/"
            f"{N_SYSTEM_SPLITS}"
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


        # ----------------------------------------------------
        # Leakage-safe data
        # ----------------------------------------------------

        selector_data = (
            make_selector_data(
                data,
                dev_systems,
                test_systems,
            )
        )


        # ----------------------------------------------------
        # Difficulty strata
        # ----------------------------------------------------

        human_bins = (
            make_bins(
                make_difficulty(
                    data,
                    dev_systems,
                    "human",
                )
            )
        )


        metricx_bins = (
            make_bins(
                make_difficulty(
                    data,
                    dev_systems,
                    "metricx",
                )
            )
        )


        # ----------------------------------------------------
        # Random + our stochastic selectors
        # ----------------------------------------------------

        print(
            "  Random + stratified..."
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


                # Random
                rng = (
                    np.random
                    .default_rng(
                        seed
                    )
                )


                selected_random = (
                    rng.choice(
                        len(
                            data
                        ),
                        size=budget,
                        replace=False,
                    )
                    .tolist()
                )


                add_result(
                    language_pair,
                    split_id,
                    "Random",
                    budget,
                    trial,
                    selected_random,
                    data,
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


                add_result(
                    language_pair,
                    split_id,
                    "Stratified-Human",
                    budget,
                    trial,
                    selected_human,
                    data,
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


                add_result(
                    language_pair,
                    split_id,
                    "Stratified-MetricX",
                    budget,
                    trial,
                    selected_metricx,
                    data,
                    dev_systems,
                    test_systems,
                    reference,
                )


        # ----------------------------------------------------
        # Ranked selector helper
        # ----------------------------------------------------

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
                    item[
                        "i"
                    ]
                )

                for item
                in ranked
            ]


            assert (
                len(
                    ranked_ids
                )
                == len(
                    data
                )
            )


            assert (
                len(
                    set(
                        ranked_ids
                    )
                )
                == len(
                    data
                )
            )


            for budget in BUDGETS:

                selected_ids = (
                    ranked_ids[
                        :budget
                    ]
                )


                add_result(
                    language_pair,
                    split_id,
                    name,
                    budget,
                    0,
                    selected_ids,
                    data,
                    dev_systems,
                    test_systems,
                    reference,
                )


        # ----------------------------------------------------
        # Zouhar selectors
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # DiffDisc
        # ----------------------------------------------------

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
                    f"{language_pair}, "
                    f"split {split_id}."
                )


                print(
                    f"  {type(exc).__name__}: "
                    f"{exc}"
                )


                failures.append(
                    {
                        "language_pair":
                            language_pair,

                        "split_id":
                            split_id,

                        "error_type":
                            type(
                                exc
                            ).__name__,

                        "error":
                            str(
                                exc
                            ),
                    }
                )


        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        checkpoint = (
            pd.DataFrame(
                rows
            )
            .drop_duplicates(
                subset=[
                    "language_pair",
                    "split_id",
                    "method",
                    "budget",
                    "trial",
                ],
                keep="last",
            )
            .sort_values(
                [
                    "language_pair",
                    "split_id",
                    "budget",
                    "method",
                    "trial",
                ]
            )
        )


        checkpoint.to_csv(
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


        completed_keys.add(
            progress_key
        )


        with open(
            PROGRESS_FILE,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "completed_language_splits":
                        sorted(
                            completed_keys
                        ),

                    "completed_count":
                        len(
                            completed_keys
                        ),

                    "total_count":
                        total_settings,

                    "n_languages":
                        len(
                            LANGUAGE_PAIRS
                        ),

                    "n_system_splits":
                        N_SYSTEM_SPLITS,

                    "n_stochastic_trials":
                        N_STOCHASTIC_TRIALS,
                },
                f,
                indent=2,
            )


        print(
            f"Completed "
            f"{language_pair} "
            f"split "
            f"{split_id + 1}/"
            f"{N_SYSTEM_SPLITS}. "
            f"Overall: "
            f"{len(completed_keys)}/"
            f"{total_settings}"
        )


# ============================================================
# Final trial table
# ============================================================

trials = (
    pd.DataFrame(
        rows
    )
    .drop_duplicates(
        subset=[
            "language_pair",
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
# Average stochastic trials inside each system split
#
# One system split = one unit at this stage.
# ============================================================

split_summary = (
    trials
    .groupby(
        [
            "language_pair",
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
# Attach matched Random result to each method/split
#
# Important for DiffDisc:
# if DiffDisc fails on a split, that split is not used in the
# DiffDisc-vs-Random comparison.
# ============================================================

random_split = (
    split_summary[
        split_summary[
            "method"
        ]
        == "Random"
    ][
        [
            "language_pair",
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
            "language_pair",
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
    SPLIT_SUMMARY_FILE,
    index=False,
)


# ============================================================
# Average system splits inside each language
#
# LANGUAGE is the final statistical unit.
# ============================================================

language_summary = (
    split_summary
    .groupby(
        [
            "language_pair",
            "method",
            "budget",
        ],
        as_index=False,
    )
    .agg(
        num_system_splits=(
            "split_id",
            "nunique",
        ),

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

        # Matched Random average.
        random_mae_matched=(
            "random_mae",
            "mean",
        ),
    )
)


language_summary[
    "mae_improvement"
] = (
    language_summary[
        "random_mae_matched"
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
        "random_mae_matched"
    ]
)


language_summary.to_csv(
    LANGUAGE_SUMMARY_FILE,
    index=False,
)


# ============================================================
# Macro average across languages
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

        matched_random_mae_mean=(
            "random_mae_matched",
            "mean",
        ),

        mean_language_relative_mae_reduction_pct=(
            "mae_reduction_vs_random_pct",
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
        "matched_random_mae_mean"
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
        "matched_random_mae_mean"
    ]
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


macro[
    "method"
] = pd.Categorical(
    macro[
        "method"
    ],
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
# Paired significance across LANGUAGES
# ============================================================

significance_rows = []


bootstrap_rng = (
    np.random
    .default_rng(
        BASE_SEED
        + 987654321
    )
)


for method in METHOD_ORDER:

    if method == "Random":
        continue


    for budget in BUDGETS:

        local = (
            language_summary[
                (
                    language_summary[
                        "method"
                    ]
                    == method
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
            .dropna()
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


        n_languages = len(
            deltas
        )


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
                    size=n_languages,
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

                "num_languages":
                    n_languages,

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


significance[
    "method"
] = pd.Categorical(
    significance[
        "method"
    ],
    categories=METHOD_ORDER,
    ordered=True,
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
    + "=" * 135
)

print(
    "FINAL MULTILINGUAL MACRO SUMMARY"
)

print(
    "=" * 135
)


MACRO_DISPLAY = [
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


print(
    macro[
        MACRO_DISPLAY
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print(
    "\n"
    + "=" * 135
)

print(
    "PAIRED LANGUAGE-LEVEL MAE SIGNIFICANCE VS RANDOM"
)

print(
    "=" * 135
)


SIGNIFICANCE_DISPLAY = [
    "method",
    "budget",
    "num_languages",
    "mean_absolute_mae_improvement",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "wilcoxon_one_sided_p",
    "language_win_rate",
]


print(
    significance[
        SIGNIFICANCE_DISPLAY
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# DiffDisc failures
# ============================================================

if failures:

    failure_dataframe = pd.DataFrame(
        failures
    )


    print(
        "\nDiffDisc failures:"
    )


    print(
        failure_dataframe[
            [
                "language_pair",
                "split_id",
                "error_type",
            ]
        ].to_string(
            index=False,
        )
    )


    print(
        f"\nTotal DiffDisc failures: "
        f"{len(failures)}"
    )


else:

    print(
        "\nDiffDisc failures: 0"
    )


# ============================================================
# Saved files
# ============================================================

print(
    "\nSaved:"
)

print(
    TRIAL_FILE
)

print(
    SPLIT_SUMMARY_FILE
)

print(
    LANGUAGE_SUMMARY_FILE
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
