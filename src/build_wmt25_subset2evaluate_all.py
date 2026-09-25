import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


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


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

WMT_ROOT = (
    ROOT.parent
    / "wmt25-general-mt"
)

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
    / "wmt25_subset2evaluate_all"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "coverage_manifest.csv"
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
            f"Unexpected doc_id: {doc_id}"
        )

    return (
        parts[0],
        parts[1],
        parts[2],
        int(parts[3]),
    )


# ============================================================
# Load all human-evaluation data once
# ============================================================

human_by_language = {
    language_pair: []
    for language_pair in LANGUAGE_PAIRS
}


with open(
    HUMAN_FILE,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        row = json.loads(line)

        (
            language_pair,
            domain,
            document_id,
            segment_index,
        ) = parse_human_doc_id(
            row["doc_id"]
        )

        if (
            language_pair
            not in human_by_language
        ):
            continue

        human_by_language[
            language_pair
        ].append(
            {
                "row":
                    row,

                "domain":
                    domain,

                "document_id":
                    document_id,

                "segment_index":
                    segment_index,
            }
        )


# ============================================================
# Cache automatic-score files
#
# We read each system file at most once.
# ============================================================

auto_file_cache = {}


def load_system_auto_file(
    system,
):

    if system in auto_file_cache:
        return auto_file_cache[
            system
        ]

    path = (
        AUTO_DIR
        / f"{system}.jsonl"
    )

    records = []

    if path.exists():

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                records.append(
                    json.loads(line)
                )

    auto_file_cache[
        system
    ] = records

    return records


# ============================================================
# Process one language
# ============================================================

def process_language(
    language_pair,
):

    human_rows = (
        human_by_language[
            language_pair
        ]
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        f"LANGUAGE: "
        f"{language_pair}"
    )

    print(
        "=" * 100
    )


    if not human_rows:

        print(
            "No human rows found."
        )

        return None


    # --------------------------------------------------------
    # MT systems
    #
    # refA is human reference, not an MT system.
    # --------------------------------------------------------

    human_systems = sorted(
        {
            system
            for item in human_rows
            for system
            in item[
                "row"
            ][
                "scores"
            ]
            if system != "refA"
        }
    )


    print(
        f"Human segments: "
        f"{len(human_rows)}"
    )

    print(
        f"MT systems: "
        f"{len(human_systems)}"
    )


    # --------------------------------------------------------
    # Load MetricX scores
    #
    # key:
    #
    # (system, document_id, segment_index)
    # --------------------------------------------------------

    automatic_scores = {}

    systems_with_metric = set()


    for system in human_systems:

        records = (
            load_system_auto_file(
                system
            )
        )


        for record in records:

            if (
                record.get(
                    "language_pair"
                )
                != language_pair
            ):
                continue


            metric_values = (
                record
                .get(
                    "metric_scores",
                    {},
                )
                .get(
                    METRIC
                )
            )


            if metric_values is None:
                continue


            document_id = (
                record[
                    "document_id"
                ]
            )


            for (
                segment_index,
                value,
            ) in enumerate(
                metric_values
            ):

                automatic_scores[
                    (
                        system,
                        document_id,
                        segment_index,
                    )
                ] = float(
                    value
                )


            systems_with_metric.add(
                system
            )


    print(
        f"Systems with {METRIC}: "
        f"{len(systems_with_metric)}"
    )


    # --------------------------------------------------------
    # Construct candidate rows
    # --------------------------------------------------------

    complete_items = []

    excluded_items = []

    coverage_counts = []


    for human_item in human_rows:

        row = (
            human_item[
                "row"
            ]
        )

        document_id = (
            human_item[
                "document_id"
            ]
        )

        segment_index = (
            human_item[
                "segment_index"
            ]
        )


        scores = {}

        targets = {}

        missing_systems = []


        for system in human_systems:

            # Human annotations
            annotations = (
                row[
                    "scores"
                ]
                .get(
                    system,
                    [],
                )
            )


            human_values = [
                annotation[
                    "score"
                ]

                for annotation
                in annotations

                if (
                    annotation.get(
                        "score"
                    )
                    is not None
                )
            ]


            auto_key = (
                system,
                document_id,
                segment_index,
            )


            has_metric = (
                auto_key
                in automatic_scores
            )


            has_translation = (
                system
                in row[
                    "tgt_text"
                ]
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


            scores[
                system
            ] = {
                "human":
                    float(
                        np.mean(
                            human_values
                        )
                    ),

                METRIC:
                    float(
                        automatic_scores[
                            auto_key
                        ]
                    ),
            }


            targets[
                system
            ] = (
                row[
                    "tgt_text"
                ][
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
            == len(
                human_systems
            )
        ):

            complete_items.append(
                {
                    "src":
                        row[
                            "src_text"
                        ],

                    "ref":
                        row[
                            "tgt_text"
                        ]
                        .get(
                            "refA",
                            "",
                        ),

                    "tgt":
                        targets,

                    "scores":
                        scores,

                    "domain":
                        human_item[
                            "domain"
                        ],

                    "doc":
                        document_id,

                    "segment_index":
                        segment_index,

                    "original_doc_id":
                        row[
                            "doc_id"
                        ],
                }
            )


        else:

            excluded_items.append(
                {
                    "doc_id":
                        row[
                            "doc_id"
                        ],

                    "domain":
                        human_item[
                            "domain"
                        ],

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


    # --------------------------------------------------------
    # Reindex
    # --------------------------------------------------------

    final_data = []


    for index, item in enumerate(
        complete_items
    ):

        item = dict(
            item
        )

        item[
            "i"
        ] = index

        final_data.append(
            item
        )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    expected_systems = set(
        human_systems
    )


    for item in final_data:

        assert (
            set(
                item[
                    "scores"
                ]
            )
            == expected_systems
        )

        assert (
            set(
                item[
                    "tgt"
                ]
            )
            == expected_systems
        )


    # --------------------------------------------------------
    # Coverage diagnostics
    # --------------------------------------------------------

    histogram = Counter(
        coverage_counts
    )


    domain_exclusions = Counter(
        item[
            "domain"
        ]
        for item
        in excluded_items
    )


    print(
        "\nSystems-per-segment histogram:"
    )


    for (
        n_systems,
        count,
    ) in sorted(
        histogram.items()
    ):

        print(
            f"  {n_systems:2d} systems: "
            f"{count} segments"
        )


    print(
        f"\nComplete segments: "
        f"{len(final_data)}"
    )

    print(
        f"Excluded segments: "
        f"{len(excluded_items)}"
    )

    print(
        f"Retained fraction: "
        f"{len(final_data) / len(human_rows):.3%}"
    )


    print(
        "Excluded by domain:"
    )


    if domain_exclusions:

        for (
            domain,
            count,
        ) in sorted(
            domain_exclusions.items()
        ):

            print(
                f"  {domain:20s} "
                f"{count}"
            )

    else:

        print(
            "  none"
        )


    # --------------------------------------------------------
    # Save JSONL
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / (
            f"{language_pair}"
            "_subset2evaluate.jsonl"
        )
    )


    with open(
        output_file,
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


    # --------------------------------------------------------
    # Save excluded rows
    # --------------------------------------------------------

    excluded_file = (
        OUTPUT_DIR
        / (
            f"{language_pair}"
            "_excluded_segments.csv"
        )
    )


    with open(
        excluded_file,
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


    # --------------------------------------------------------
    # Manifest row
    # --------------------------------------------------------

    manifest = {
        "language_pair":
            language_pair,

        "original_segments":
            len(
                human_rows
            ),

        "complete_segments":
            len(
                final_data
            ),

        "excluded_segments":
            len(
                excluded_items
            ),

        "retained_fraction":
            (
                len(
                    final_data
                )
                / len(
                    human_rows
                )
            ),

        "num_mt_systems":
            len(
                human_systems
            ),

        "systems_with_metric":
            len(
                systems_with_metric
            ),

        "min_systems_per_segment":
            min(
                coverage_counts
            ),

        "max_systems_per_segment":
            max(
                coverage_counts
            ),

        "speech_excluded":
            domain_exclusions.get(
                "speech",
                0,
            ),

        "non_speech_excluded":
            (
                len(
                    excluded_items
                )
                - domain_exclusions.get(
                    "speech",
                    0,
                )
            ),

        "output_file":
            str(
                output_file
            ),
    }


    return manifest


# ============================================================
# Run all languages
# ============================================================

manifest_rows = []


for language_pair in LANGUAGE_PAIRS:

    result = process_language(
        language_pair
    )

    if result is not None:

        manifest_rows.append(
            result
        )


# ============================================================
# Save + print manifest
# ============================================================

manifest = pd.DataFrame(
    manifest_rows
)


manifest.to_csv(
    MANIFEST_FILE,
    index=False,
)


print(
    "\n"
    + "=" * 120
)

print(
    "FINAL MULTILINGUAL COVERAGE MANIFEST"
)

print(
    "=" * 120
)


display_columns = [
    "language_pair",
    "original_segments",
    "complete_segments",
    "excluded_segments",
    "retained_fraction",
    "num_mt_systems",
    "systems_with_metric",
    "speech_excluded",
    "non_speech_excluded",
]


print(
    manifest[
        display_columns
    ]
    .sort_values(
        "language_pair"
    )
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


print(
    "\nSaved manifest:"
)

print(
    MANIFEST_FILE
)


print(
    "\nDone."
)
