#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

LANGUAGE_PAIR = "cs-de_DE"

ROOT = Path(__file__).resolve().parents[1]

WMT_FILE = (
    ROOT.parent
    / "wmt25-general-mt"
    / "data"
    / "wmt25-genmt-humeval.jsonl"
)

OUTPUT_DIR = ROOT / "results" / "score_matrices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATRIX_FILE = OUTPUT_DIR / f"{LANGUAGE_PAIR}_human_scores.csv"
RANKING_FILE = OUTPUT_DIR / f"{LANGUAGE_PAIR}_full_ranking.csv"


# ============================================================
# Read human scores
# ============================================================

records = []

with open(WMT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        pair = row["doc_id"].split("_#_", 1)[0]

        if pair != LANGUAGE_PAIR:
            continue

        sentence_id = row["doc_id"]
        source_text = row["src_text"]

        record = {
            "sentence_id": sentence_id,
            "source_text": source_text,
        }

        for system, annotations in row["scores"].items():

            valid_scores = [
                annotation["score"]
                for annotation in annotations
                if annotation.get("score") is not None
            ]

            if valid_scores:
                # Average multiple human judgments
                record[system] = sum(valid_scores) / len(valid_scores)

        records.append(record)


df = pd.DataFrame(records)

metadata_columns = ["sentence_id", "source_text"]

system_columns = [
    col for col in df.columns
    if col not in metadata_columns
]
# Exclude the human reference translation.
system_columns = [
    col for col in system_columns
    if col != "refA"
]
# Stable ordering
system_columns = sorted(system_columns)

df = df[
    metadata_columns + system_columns
]


# ============================================================
# Validate matrix
# ============================================================

score_matrix = df[system_columns]

n_sentences = len(df)
n_systems = len(system_columns)
missing = int(score_matrix.isna().sum().sum())

print("=" * 80)
print(f"BUILDING HUMAN SCORE MATRIX: {LANGUAGE_PAIR}")
print("=" * 80)

print(f"\nSentences: {n_sentences}")
print(f"Systems:   {n_systems}")
print(f"Shape:     {score_matrix.shape}")
print(f"Missing score cells: {missing}")

if missing != 0:
    print("\nWARNING: Matrix contains missing values.")
else:
    print("\nMatrix is complete.")


# ============================================================
# Full-test-set system scores
# ============================================================

system_scores = score_matrix.mean(axis=0)

ranking = (
    system_scores
    .sort_values(ascending=False)
    .rename("full_mean_score")
    .reset_index()
    .rename(columns={"index": "system"})
)

ranking.insert(
    0,
    "rank",
    range(1, len(ranking) + 1),
)


# ============================================================
# Save
# ============================================================

df.to_csv(
    MATRIX_FILE,
    index=False,
)

ranking.to_csv(
    RANKING_FILE,
    index=False,
)


# ============================================================
# Print summary
# ============================================================

print("\nFULL-DATASET SYSTEM RANKING")
print("-" * 60)

print(
    ranking.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)

print("\nSaved score matrix:")
print(MATRIX_FILE)

print("\nSaved full ranking:")
print(RANKING_FILE)
