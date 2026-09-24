#!/usr/bin/env python3

import json
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd


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

OUTPUT_FILE = ROOT / "results" / "wmt25_language_pair_audit.csv"


# ============================================================
# Containers
# ============================================================

num_segments = defaultdict(int)

translation_systems = defaultdict(set)
human_scored_systems = defaultdict(set)

scored_systems_per_segment = defaultdict(list)

# Number of sentence-system cells that actually contain
# at least one human score.
observed_scored_cells = defaultdict(int)

# Total number of individual human annotations.
annotation_counts = defaultdict(int)


# ============================================================
# Read dataset
# ============================================================

print("=" * 80)
print("AUDITING WMT25 HUMAN-EVALUATION DATA")
print("=" * 80)

print(f"\nInput: {WMT_FILE}")

with open(WMT_FILE, "r", encoding="utf-8") as f:

    for line_number, line in enumerate(f, start=1):

        row = json.loads(line)

        doc_id = row["doc_id"]

        # Example:
        # cs-de_DE_#_news_#_blesk.cz.112043_#_0
        #
        # Keep the complete pair label "cs-de_DE".
        pair = doc_id.split("_#_", 1)[0]

        num_segments[pair] += 1

        translations = row.get("tgt_text", {})
        scores = row.get("scores", {})

        translation_systems[pair].update(translations.keys())

        n_scored_this_segment = 0

        for system, annotations in scores.items():

            valid_scores = [
                annotation.get("score")
                for annotation in annotations
                if annotation.get("score") is not None
            ]

            if not valid_scores:
                continue

            human_scored_systems[pair].add(system)

            n_scored_this_segment += 1

            observed_scored_cells[pair] += 1

            annotation_counts[pair] += len(valid_scores)

        scored_systems_per_segment[pair].append(
            n_scored_this_segment
        )

        if line_number % 1000 == 0:
            print(
                f"Processed {line_number:,} rows...",
                end="\r"
            )


# ============================================================
# Build audit table
# ============================================================

rows = []

for pair in sorted(num_segments):

    n_segments = num_segments[pair]

    n_translation_systems = len(
        translation_systems[pair]
    )

    n_human_systems = len(
        human_scored_systems[pair]
    )

    counts = scored_systems_per_segment[pair]

    mean_scored = (
        statistics.mean(counts)
        if counts else 0.0
    )

    median_scored = (
        statistics.median(counts)
        if counts else 0.0
    )

    min_scored = min(counts) if counts else 0
    max_scored = max(counts) if counts else 0

    # Rough sentence x system coverage.
    #
    # Denominator uses every human-scored system observed
    # anywhere for this language pair.
    possible_cells = (
        n_segments * n_human_systems
    )

    coverage = (
        observed_scored_cells[pair]
        / possible_cells
        if possible_cells > 0
        else 0.0
    )

    avg_annotations_per_cell = (
        annotation_counts[pair]
        / observed_scored_cells[pair]
        if observed_scored_cells[pair] > 0
        else 0.0
    )

    rows.append(
        {
            "language_pair": pair,
            "segments": n_segments,
            "translation_systems": n_translation_systems,
            "human_scored_systems": n_human_systems,
            "mean_scored_systems_per_segment": mean_scored,
            "median_scored_systems_per_segment": median_scored,
            "min_scored_systems_per_segment": min_scored,
            "max_scored_systems_per_segment": max_scored,
            "observed_sentence_system_cells":
                observed_scored_cells[pair],
            "human_score_coverage": coverage,
            "avg_annotations_per_scored_cell":
                avg_annotations_per_cell,
        }
    )


df = pd.DataFrame(rows)

# Sort first by number of human-scored systems,
# then by number of segments.
df = df.sort_values(
    [
        "human_scored_systems",
        "segments",
    ],
    ascending=False,
).reset_index(drop=True)


# ============================================================
# Save + print
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
)

pd.set_option(
    "display.max_rows",
    None,
)

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    180,
)

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.3f}",
)

print("\n\n" + "=" * 80)
print("LANGUAGE-PAIR AUDIT")
print("=" * 80)

print(
    df[
        [
            "language_pair",
            "segments",
            "translation_systems",
            "human_scored_systems",
            "mean_scored_systems_per_segment",
            "human_score_coverage",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_FILE)
