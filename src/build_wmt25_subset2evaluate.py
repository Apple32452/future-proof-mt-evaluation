import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


# ============================================================
# Configuration
# ============================================================

LANGUAGE_PAIR = "cs-de_DE"
METRIC = "MetricX-24-Hybrid-XL"


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

WMT_ROOT = ROOT.parent / "wmt25-general-mt"

HUMAN_FILE = (
    WMT_ROOT
    / "data"
    / "wmt25-genmt-humeval.jsonl"
)

AUTO_DIR = (
    WMT_ROOT
    / "data"
    / "automatic_scores"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "wmt25_subset2evaluate"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_subset2evaluate.jsonl"
)

EXCLUDED_FILE = (
    OUTPUT_DIR
    / f"{LANGUAGE_PAIR}_excluded_segments.csv"
)


# ============================================================
# Helpers
# ============================================================

def parse_human_doc_id(doc_id):
    """
    Example:

    cs-de_DE_#_news_#_blesk.cz.112043_#_0
    """

    parts = doc_id.split("_#_")

    if len(parts) != 4:
        raise ValueError(
            f"Unexpected human doc_id: {doc_id}"
        )

    pair = parts[0]
    domain = parts[1]
    document_id = parts[2]
    segment_index = int(parts[3])

    return (
        pair,
        domain,
        document_id,
        segment_index,
    )


# ============================================================
# Load human-evaluation rows
# ============================================================

human_rows = []

with open(
    HUMAN_FILE,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        row = json.loads(line)

        (
            pair,
            domain,
            document_id,
            segment_index,
        ) = parse_human_doc_id(
            row["doc_id"]
        )

        if pair != LANGUAGE_PAIR:
            continue

        human_rows.append(
            {
                "row": row,
                "domain": domain,
                "document_id": document_id,
                "segment_index": segment_index,
            }
        )


print("=" * 80)
print("WMT25 -> subset2evaluate ADAPTER V2")
print("=" * 80)

print(
    f"\nLanguage pair: {LANGUAGE_PAIR}"
)

print(
    f"Human-evaluated segments: "
    f"{len(human_rows)}"
)


# ============================================================
# Determine MT systems
# ============================================================

human_systems = sorted(
    {
        system
        for item in human_rows
        for system in item["row"]["scores"]
        if system != "refA"
    }
)

print(
    f"Human-scored MT systems: "
    f"{len(human_systems)}"
)


# ============================================================
# Load automatic MetricX scores
#
# Mapping:
#
# (system, document_id, segment_index)
#       -> MetricX score
# ============================================================

auto_scores = {}

systems_with_metric = set()


for system in human_systems:

    path = (
        AUTO_DIR
        / f"{system}.jsonl"
    )

    if not path.exists():

        print(
            f"WARNING: missing automatic-score file: "
            f"{system}"
        )

        continue


    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            row = json.loads(line)

            if (
                row.get("language_pair")
                != LANGUAGE_PAIR
            ):
                continue

            metric_values = (
                row
                .get("metric_scores", {})
                .get(METRIC)
            )

            if metric_values is None:
                continue

            document_id = row[
                "document_id"
            ]

            for (
                segment_index,
                value,
            ) in enumerate(
                metric_values
            ):

                auto_scores[
                    (
                        system,
                        document_id,
                        segment_index,
                    )
                ] = float(value)

            systems_with_metric.add(
                system
            )


print(
    f"Systems with {METRIC}: "
    f"{len(systems_with_metric)}"
)


# ============================================================
# Build rows
#
# IMPORTANT:
#
# We keep a SENTENCE only if all 20 systems have:
#
#   - human score
#   - MetricX score
#   - translation
#
# We do NOT search for systems that cover all original
# 231 sentences.
# ============================================================

complete_items = []

excluded_items = []

coverage_counts = []


for item in human_rows:

    row = item["row"]

    document_id = (
        item["document_id"]
    )

    segment_index = (
        item["segment_index"]
    )

    scores = {}
    targets = {}
    missing_systems = []


    for system in human_systems:

        # ----------------------------------------------------
        # Human score
        # ----------------------------------------------------

        annotations = (
            row["scores"]
            .get(
                system,
                [],
            )
        )

        human_values = [
            ann["score"]
            for ann in annotations
            if ann.get("score") is not None
        ]


        # ----------------------------------------------------
        # MetricX score
        # ----------------------------------------------------

        key = (
            system,
            document_id,
            segment_index,
        )

        has_metric = (
            key in auto_scores
        )


        # ----------------------------------------------------
        # Translation
        # ----------------------------------------------------

        has_translation = (
            system
            in row["tgt_text"]
        )


        if (
            not human_values
            or not has_metric
            or not has_translation
        ):

            missing_systems.append(
                system
            )

            continue


        scores[system] = {
            "human": float(
                np.mean(
                    human_values
                )
            ),

            METRIC: float(
                auto_scores[key]
            ),
        }

        targets[system] = (
            row["tgt_text"][
                system
            ]
        )


    n_available = len(
        scores
    )

    coverage_counts.append(
        n_available
    )


    if (
        n_available
        == len(human_systems)
    ):

        complete_items.append(
            {
                "src":
                    row["src_text"],

                "ref":
                    row["tgt_text"]
                    .get(
                        "refA",
                        "",
                    ),

                "tgt":
                    targets,

                "scores":
                    scores,

                "domain":
                    item["domain"],

                "doc":
                    document_id,

                "segment_index":
                    segment_index,

                "original_doc_id":
                    row["doc_id"],
            }
        )

    else:

        excluded_items.append(
            {
                "doc_id":
                    row["doc_id"],

                "domain":
                    item["domain"],

                "document_id":
                    document_id,

                "segment_index":
                    segment_index,

                "systems_available":
                    n_available,

                "systems_expected":
                    len(
                        human_systems
                    ),

                "missing_system_count":
                    len(
                        missing_systems
                    ),

                "missing_systems":
                    ", ".join(
                        missing_systems
                    ),
            }
        )


# ============================================================
# Coverage diagnostics
# ============================================================

print("\n" + "=" * 80)
print("RAW ADAPTER COVERAGE")
print("=" * 80)

print(
    f"Original human segments: "
    f"{len(human_rows)}"
)

print(
    f"Expected systems per segment: "
    f"{len(human_systems)}"
)

print(
    f"Min systems available: "
    f"{min(coverage_counts)}"
)

print(
    f"Max systems available: "
    f"{max(coverage_counts)}"
)

print(
    f"Mean systems available: "
    f"{np.mean(coverage_counts):.2f}"
)


hist = Counter(
    coverage_counts
)

print(
    "\nSystems-per-segment histogram:"
)

for n_systems in sorted(hist):

    print(
        f"  {n_systems:2d} systems: "
        f"{hist[n_systems]} segments"
    )


print(
    f"\nComplete segments: "
    f"{len(complete_items)}"
)

print(
    f"Excluded segments: "
    f"{len(excluded_items)}"
)


# ============================================================
# Excluded domains
# ============================================================

domain_counts = Counter(
    item["domain"]
    for item in excluded_items
)

print(
    "\nExcluded segments by domain:"
)

if domain_counts:

    for domain, count in sorted(
        domain_counts.items()
    ):

        print(
            f"  {domain:20s} "
            f"{count}"
        )

else:

    print("  none")


# ============================================================
# Excluded documents
# ============================================================

doc_counts = Counter(
    (
        item["domain"],
        item["document_id"],
    )
    for item in excluded_items
)

print(
    "\nDocuments containing excluded segments:"
)

if doc_counts:

    for (
        domain,
        document_id,
    ), count in sorted(
        doc_counts.items(),
        key=lambda x: -x[1],
    ):

        print(
            f"  {domain:15s} "
            f"{document_id:40s} "
            f"{count}"
        )

else:

    print("  none")


# ============================================================
# Re-index for subset2evaluate
# ============================================================

final_data = []

for index, item in enumerate(
    complete_items
):

    new_item = dict(
        item
    )

    new_item["i"] = index

    final_data.append(
        new_item
    )


# ============================================================
# Integrity checks
# ============================================================

expected_systems = set(
    human_systems
)

for item in final_data:

    assert (
        set(
            item["scores"]
        )
        == expected_systems
    )

    assert (
        set(
            item["tgt"]
        )
        == expected_systems
    )


# ============================================================
# Final dataset
# ============================================================

print("\n" + "=" * 80)
print("FINAL RECTANGULAR DATASET")
print("=" * 80)

print(
    f"Shape: "
    f"{len(final_data)} "
    f"x "
    f"{len(human_systems)}"
)

print(
    f"Retained fraction: "
    f"{len(final_data) / len(human_rows):.3%}"
)


if len(final_data) == 0:

    raise RuntimeError(
        "No complete rows found."
    )


example = final_data[0]

first_system = human_systems[0]

print("\nExample item:")

print(
    "source:",
    example["src"][:120]
)

print(
    "systems:",
    len(
        example["scores"]
    )
)

print(
    "example system:",
    first_system
)

print(
    "human:",
    example["scores"][
        first_system
    ]["human"]
)

print(
    METRIC + ":",
    example["scores"][
        first_system
    ][METRIC]
)


# ============================================================
# Save JSONL
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    for item in final_data:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# Save exclusions
# ============================================================

with open(
    EXCLUDED_FILE,
    "w",
    encoding="utf-8",
    newline="",
) as f:

    fieldnames = [
        "doc_id",
        "domain",
        "document_id",
        "segment_index",
        "systems_available",
        "systems_expected",
        "missing_system_count",
        "missing_systems",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        excluded_items
    )


print(
    "\nSaved dataset:"
)

print(
    OUTPUT_FILE
)

print(
    "\nSaved exclusion audit:"
)

print(
    EXCLUDED_FILE
)

print(
    "\nDone."
)
