#!/usr/bin/env python3

"""
Small reproduction of the output-based MT datapoint selectors from:

Zouhar, Cui, and Sachan (2025)
"How to Select Datapoints for Efficient Human Evaluation of NLG Models?"

Dataset:
    WMT23 English -> Czech

Purpose:
    This is a sanity-check reproduction using the authors' released
    subset2evaluate implementation before adapting the selectors to
    our WMT25 unseen-system / unseen-family experiments.

Important:
    The original subset2evaluate DiffDisc implementation hard-codes
    device="cuda". On Apple Silicon / CPU-only machines, this script
    transparently redirects that training call to CPU without changing
    the IRT model, utility function, or number of epochs.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import subset2evaluate


# ============================================================
# Configuration
# ============================================================

DATASET = "wmt23/en-cs"

# Zouhar et al. evaluate subsets from 5% through 50%.
PROPS = np.linspace(
    0.05,
    0.50,
    10,
)

# Random is stochastic, so average over many runs.
RANDOM_REPETITIONS = 100

# Small reproduction:
# run one DiffDisc fit rather than five.
#
# The original model/training objective remains unchanged.
IRT_REPETITIONS = 1

# Keep the standard IRT training length.
IRT_EPOCHS = 1000

BASE_SEED = 42


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "results"
    / "zouhar_wmt23_reproduction"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CURVE_FILE = (
    OUTPUT_DIR
    / "wmt23_en-cs_output_based_curve.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "wmt23_en-cs_output_based_summary.csv"
)


# ============================================================
# CPU compatibility for DiffDisc
# ============================================================

def enable_cpu_irt_compatibility():
    """
    subset2evaluate currently calls:

        trainer.train(..., device="cuda")

    inside its py_irt selector.

    On machines without CUDA, intercept the py-irt trainer call and
    replace device="cuda" with device="cpu".

    This does NOT modify the installed source code and does NOT alter
    the IRT model or selection objective.
    """

    try:
        import torch
        import py_irt.training

    except ImportError as exc:
        print("\nWARNING:")
        print("  py-irt is not installed.")
        print("  DiffDisc will be skipped.")
        print(f"  Reason: {exc}")
        return False

    # If CUDA actually exists, nothing needs to change.
    if torch.cuda.is_available():
        print("\nDiffDisc device: CUDA")
        return True

    original_train = (
        py_irt.training
        .IrtModelTrainer
        .train
    )

    # Avoid wrapping multiple times.
    if getattr(
        original_train,
        "_futureproof_cpu_wrapper",
        False,
    ):
        print("\nDiffDisc device: CPU")
        return True

    def train_cpu_compatible(
        self,
        *args,
        **kwargs,
    ):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"

        return original_train(
            self,
            *args,
            **kwargs,
        )

    train_cpu_compatible._futureproof_cpu_wrapper = True

    (
        py_irt.training
        .IrtModelTrainer
        .train
    ) = train_cpu_compatible

    print("\nDiffDisc device: CPU")
    print(
        "  Redirecting the released CUDA trainer call "
        "to CPU for this reproduction."
    )

    return True


# ============================================================
# Load WMT23
# ============================================================

print("=" * 80)
print("ZOUHAR ET AL. WMT23 SMALL REPRODUCTION")
print("=" * 80)

data = subset2evaluate.utils.load_data(
    DATASET
)

print(f"\nDataset: {DATASET}")
print(f"Items:   {len(data):,}")


# ============================================================
# Determine available MetricX version
# ============================================================

first_system = next(
    iter(
        data[0]["scores"]
    )
)

available_metrics = set(
    data[0]["scores"][first_system].keys()
)

if "MetricX-23-c" in available_metrics:
    METRIC = "MetricX-23-c"

elif "MetricX-23" in available_metrics:
    METRIC = "MetricX-23"

else:
    raise RuntimeError(
        "Could not find MetricX-23-c or MetricX-23.\n"
        "Available metrics include:\n"
        + "\n".join(
            sorted(available_metrics)
        )
    )

print(f"Selection metric: {METRIC}")


# ============================================================
# Verify human evaluation exists
# ============================================================

if "human" not in available_metrics:
    raise RuntimeError(
        "Human scores were not found in the WMT23 data."
    )

print("Evaluation target: human")


# ============================================================
# Setup DiffDisc compatibility
# ============================================================

IRT_AVAILABLE = (
    enable_cpu_irt_compatibility()
)


# ============================================================
# Selector definitions
# ============================================================

# Random is deliberately first.
#
# Therefore, even if the more expensive DiffDisc fit encounters
# an issue, we will already have the main random reference.

METHODS = [
    {
        "name": "Random",
        "repetitions": RANDOM_REPETITIONS,
        "kwargs": {
            "method": "random",
        },
    },

    {
        "name": "MetricAvg",
        "repetitions": 1,
        "kwargs": {
            "method": "metric_avg",
            "metric": METRIC,
        },
    },

    {
        "name": "MetricVar",
        "repetitions": 1,
        "kwargs": {
            "method": "metric_var",
            "metric": METRIC,
        },
    },

    {
        "name": "MetricCons",
        "repetitions": 1,
        "kwargs": {
            "method": "metric_cons",
            "metric": METRIC,
        },
    },

    {
        "name": "Diversity",
        "repetitions": 1,
        "kwargs": {
            "method": "diversity",
            "metric": "lm",
        },
    },
]

if IRT_AVAILABLE:
    METHODS.append(
        {
            "name": "DiffDisc",
            "repetitions": IRT_REPETITIONS,
            "kwargs": {
                "method": "pyirt_diffdisc",
                "model": "4pl_score",
                "metric": METRIC,

                # Keep the standard IRT training length.
                "epochs": IRT_EPOCHS,

                # Important for debugging.
                # Do not endlessly retry CPU/CUDA failures.
                "retry_on_error": False,
            },
        }
    )


# ============================================================
# Run selector
# ============================================================

curve_rows = []
method_status = {}

for method_config in METHODS:

    method_name = (
        method_config["name"]
    )

    repetitions = (
        method_config["repetitions"]
    )

    base_kwargs = dict(
        method_config["kwargs"]
    )

    print("\n" + "-" * 80)

    print(
        f"Running {method_name} "
        f"({repetitions} repetition(s))..."
    )

    curves = []

    try:

        for repetition in range(
            repetitions
        ):

            kwargs = dict(
                base_kwargs
            )

            # Make Random deterministic/reproducible.
            if kwargs["method"] == "random":
                kwargs["seed"] = (
                    BASE_SEED
                    + repetition
                )

            selected = (
                subset2evaluate
                .select_subset
                .basic(
                    data,
                    **kwargs,
                )
            )

            # ------------------------------------------------
            # Soft Pairwise Accuracy
            #
            # This is the ranking-oriented meta-evaluation used
            # by Zouhar et al. for the output-based selectors.
            # ------------------------------------------------

            spa_curve = (
                subset2evaluate
                .evaluate
                .eval_spa(
                    selected,
                    data,
                    metric="human",
                    props=PROPS,
                )
            )

            curves.append(
                np.asarray(
                    spa_curve,
                    dtype=float,
                )
            )

            if repetitions > 1:
                print(
                    f"  repetition "
                    f"{repetition + 1}/"
                    f"{repetitions}",
                    end="\r",
                )

        if repetitions > 1:
            print()

        curves = np.asarray(
            curves,
            dtype=float,
        )

        mean_curve = (
            curves.mean(axis=0)
        )

        if repetitions > 1:
            std_curve = (
                curves.std(
                    axis=0,
                    ddof=1,
                )
            )
        else:
            std_curve = np.zeros(
                len(PROPS),
                dtype=float,
            )

        method_status[
            method_name
        ] = "success"

        print(
            "Average SPA over budgets: "
            f"{mean_curve.mean():.4f}"
        )

        # Save every budget point.
        for (
            prop,
            spa_mean,
            spa_std,
        ) in zip(
            PROPS,
            mean_curve,
            std_curve,
        ):

            curve_rows.append(
                {
                    "dataset":
                        DATASET,

                    "method":
                        method_name,

                    "selection_metric":
                        (
                            METRIC
                            if method_name
                            not in {
                                "Random",
                                "Diversity",
                            }
                            else (
                                "LM diversity"
                                if method_name
                                == "Diversity"
                                else "none"
                            )
                        ),

                    "proportion":
                        prop,

                    "num_items":
                        int(
                            len(data)
                            * prop
                        ),

                    "spa_mean":
                        spa_mean,

                    "spa_std":
                        spa_std,

                    "repetitions":
                        repetitions,
                }
            )

    except Exception as exc:

        method_status[
            method_name
        ] = (
            f"FAILED: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print(
            f"\n{method_name} FAILED"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        # Continue so one optional selector does not destroy
        # the rest of the reproduction.


# ============================================================
# Save curve-level results
# ============================================================

results = pd.DataFrame(
    curve_rows
)

if len(results) == 0:
    raise RuntimeError(
        "No selector completed successfully."
    )

results.to_csv(
    CURVE_FILE,
    index=False,
)


# ============================================================
# Summary
# ============================================================

summary = (
    results
    .groupby(
        "method",
        as_index=False,
    )
    .agg(
        mean_spa=(
            "spa_mean",
            "mean",
        ),

        spa_at_5pct=(
            "spa_mean",
            "first",
        ),

        spa_at_50pct=(
            "spa_mean",
            "last",
        ),
    )
)

summary["status"] = (
    summary["method"]
    .map(method_status)
)

summary = (
    summary
    .sort_values(
        "mean_spa",
        ascending=False,
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
# Print final result
# ============================================================

print("\n" + "=" * 80)
print("WMT23 EN-CS REPRODUCTION SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# Print failed/skipped methods
# ============================================================

completed = set(
    summary["method"]
)

failed_or_skipped = [
    (
        method_name,
        status,
    )
    for method_name, status
    in method_status.items()
    if method_name not in completed
]

if failed_or_skipped:

    print("\n" + "=" * 80)
    print("FAILED / SKIPPED METHODS")
    print("=" * 80)

    for method_name, status in failed_or_skipped:
        print(
            f"{method_name}: {status}"
        )


# ============================================================
# Output paths
# ============================================================

print("\nSaved curve results:")
print(CURVE_FILE)

print("\nSaved summary:")
print(SUMMARY_FILE)

print("\nDone.")