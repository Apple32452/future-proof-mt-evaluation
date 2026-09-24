#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd


LANGUAGE_PAIR = "cs-de_DE"
SEED = 42
N_TEST_SYSTEMS = 6

ROOT = Path(__file__).resolve().parents[1]

MATRIX_FILE = (
    ROOT
    / "results"
    / "score_matrices"
    / f"{LANGUAGE_PAIR}_human_scores.csv"
)

OUTPUT_FILE = (
    ROOT
    / "results"
    / "splits"
    / f"{LANGUAGE_PAIR}_pilot_system_split.json"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Load systems
# ------------------------------------------------------------

df = pd.read_csv(MATRIX_FILE)

metadata_columns = [
    "sentence_id",
    "source_text",
]

systems = sorted(
    [
        c
        for c in df.columns
        if c not in metadata_columns
    ]
)

rng = np.random.default_rng(SEED)

test_systems = sorted(
    rng.choice(
        systems,
        size=N_TEST_SYSTEMS,
        replace=False,
    ).tolist()
)

dev_systems = sorted(
    [
        s
        for s in systems
        if s not in test_systems
    ]
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

split = {
    "language_pair": LANGUAGE_PAIR,
    "seed": SEED,
    "dev_systems": dev_systems,
    "test_systems": test_systems,
}

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        split,
        f,
        indent=2,
    )


# ------------------------------------------------------------
# Print
# ------------------------------------------------------------

print("=" * 70)
print("PILOT SYSTEM SPLIT")
print("=" * 70)

print(f"\nDevelopment systems ({len(dev_systems)}):")

for system in dev_systems:
    print(" ", system)

print(f"\nUnseen test systems ({len(test_systems)}):")

for system in test_systems:
    print(" ", system)

print("\nSaved:")
print(OUTPUT_FILE)
