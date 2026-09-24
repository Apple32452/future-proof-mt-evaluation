#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


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

# Explicit, transparent family mapping.
#
# Only multi-system families are needed for the first
# family-holdout experiment.

FAMILY_MAP = {
    # Gemma
    "Gemma-3-12B": "Gemma",
    "Gemma-3-27B": "Gemma",
    "GemTrans": "Gemma",
    "IRB-MT": "Gemma",
    "UvA-MT": "Gemma",

    # Qwen
    "Algharb": "Qwen",
    "Wenyiil": "Qwen",
    "Yolu": "Qwen",

    # Command A
    "CommandA": "CommandA",
    "CommandA-MT": "CommandA",

    # EuroLLM
    "CUNI-MH-v2": "EuroLLM",
    "Laniqo": "EuroLLM",
}


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

OUTPUT_FILE = (
    ROOT
    / "results"
    / "family_holdout"
    / "family_holdout_audit.csv"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Collect scored systems for each language pair
# ============================================================

systems_by_pair = defaultdict(set)
segments_by_pair = defaultdict(int)

with open(WMT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        row = json.loads(line)

        pair = (
            row["doc_id"]
            .split("_#_", 1)[0]
        )

        if pair not in LANGUAGE_PAIRS:
            continue

        segments_by_pair[pair] += 1

        for system, annotations in row["scores"].items():

            # Human reference is not an MT system.
            if system == "refA":
                continue

            valid = any(
                annotation.get("score") is not None
                for annotation in annotations
            )

            if valid:
                systems_by_pair[pair].add(system)


# ============================================================
# Audit family availability
# ============================================================

rows = []

for pair in LANGUAGE_PAIRS:

    all_systems = systems_by_pair[pair]

    for family in sorted(set(FAMILY_MAP.values())):

        family_systems = sorted(
            system
            for system, mapped_family
            in FAMILY_MAP.items()
            if (
                mapped_family == family
                and system in all_systems
            )
        )

        n_family = len(family_systems)

        n_dev = (
            len(all_systems)
            - n_family
        )

        # At least two family members gives a meaningful
        # family-level holdout rather than a singleton system.
        eligible = (
            n_family >= 2
            and n_dev >= 5
        )

        rows.append(
            {
                "language_pair": pair,
                "segments":
                    segments_by_pair[pair],

                "total_mt_systems":
                    len(all_systems),

                "family":
                    family,

                "heldout_family_systems":
                    n_family,

                "remaining_dev_systems":
                    n_dev,

                "eligible":
                    eligible,

                "systems":
                    ", ".join(family_systems),
            }
        )


df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# Print eligible holdouts
# ============================================================

eligible = (
    df[df["eligible"]]
    .sort_values(
        [
            "language_pair",
            "family",
        ]
    )
)

print("=" * 110)
print("ELIGIBLE MODEL-FAMILY HOLDOUTS")
print("=" * 110)

print(
    eligible.to_string(
        index=False
    )
)

print("\n" + "=" * 110)
print("COUNTS BY FAMILY")
print("=" * 110)

print(
    eligible
    .groupby("family")
    .agg(
        language_pairs=(
            "language_pair",
            "nunique",
        ),
        total_holdout_cases=(
            "language_pair",
            "size",
        ),
        mean_family_size=(
            "heldout_family_systems",
            "mean",
        ),
    )
    .to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\nTotal eligible family-language cases:")
print(len(eligible))

print("\nSaved:")
print(OUTPUT_FILE)
