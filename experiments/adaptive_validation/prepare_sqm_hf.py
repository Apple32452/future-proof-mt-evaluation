"""Prepare public WMT22 SQM panels using quality-blind coverage selection.

Usage:
  python prepare_sqm_hf.py --year 2022 --lp en-zh --out inputs/wmt22_sqm_enzh.csv
Official data: https://huggingface.co/datasets/RicardoRei/wmt-sqm-human-evaluation

Human scores are normalized from the documented [0,100] range to [0,1].
This is historical annotation replay, not newly obtained human judgments.
"""
import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--lp", choices=["en-de", "en-zh"], required=True)
    p.add_argument("--systems", type=int, default=8)
    p.add_argument("--revision", default=None, help="Optional pinned Hugging Face dataset commit")
    p.add_argument("--min-items", type=int, default=60)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("Refusing to overwrite existing prepared matrix")
    from datasets import load_dataset
    ds = load_dataset("RicardoRei/wmt-sqm-human-evaluation",
                      split="train", revision=args.revision)
    year_counts = collections.Counter(int(row["year"]) for row in ds)
    print("Available years:", dict(sorted(year_counts.items())))
    if args.year not in year_counts:
        p.error(f"Year {args.year} unavailable; available: {sorted(year_counts)}")
    lp_counts = collections.Counter(row["lp"] for row in ds
                                    if int(row["year"]) == args.year)
    print(f"Available language pairs in {args.year}:", dict(sorted(lp_counts.items())))
    if args.lp not in lp_counts:
        p.error(f"Language pair {args.lp} absent for {args.year}")
    cells = collections.defaultdict(list)
    for row in ds:
        if int(row["year"]) != args.year or row["lp"] != args.lp:
            continue
        source = str(row["src"]).strip()
        if not source:
            continue
        key = (str(row["domain"]), source)
        system = str(row["system"])
        if re.fullmatch(r"ref(?:[a-z]|[0-9]+)?", system, flags=re.IGNORECASE):
            continue
        value = float(row["score"])
        if not 0 <= value <= 100:
            raise ValueError("Released SQM score outside predeclared range")
        cells[(key, system)].append(value)

    coverage = collections.Counter()
    for (key, system) in cells:
        coverage[system] += 1
    selected = [s for s, n in sorted(coverage.items(),
                                    key=lambda kv: (-kv[1], kv[0]))[:args.systems]]
    print(f"For {args.year} {args.lp}: {len(coverage)} systems; "
          f"top coverage: {coverage.most_common(12)}")
    if len(selected) < args.systems:
        raise ValueError(f"Insufficient systems: {len(selected)}; requested {args.systems}")
    grouped = collections.defaultdict(dict)
    for (key, system), scores in cells.items():
        if system in selected:
            grouped[key][system] = sum(scores) / len(scores)
    retained = [(key, values) for key, values in grouped.items()
                if len(values) == len(selected)]
    retained.sort(key=lambda item: (item[0][0], item[0][1]))
    print(f"Complete item overlap for selected {len(selected)} systems: "
          f"{len(retained)} (required {args.min_items})")
    if len(retained) < args.min_items:
        raise ValueError("Insufficient common items. Do not silently lower coverage "
                         "or the number of systems after seeing results.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "doc_id"] + selected)
        for (domain, source), scores in retained:
            item_key = hashlib.sha256((domain + "\0" + source).encode()).hexdigest()
            writer.writerow([item_key, "unknown"] +
                            [scores[s] / 100 for s in selected])
    csv_hash = hashlib.sha256(args.out.read_bytes()).hexdigest()
    metadata = {
        "cohort": f"WMT{args.year}_SQM",
        "source": "RicardoRei/wmt-sqm-human-evaluation",
        "language_pair": args.lp,
        "year": args.year,
        "dataset_revision": args.revision,
        "score_bounds_original": [0, 100],
        "systems_selected_on_coverage_only": selected,
        "covered_items": len(retained),
        "item_key": "sha256(domain + NUL + source string)",
        "duplicate_handling": "mean repeated ratings for same source-domain-system",
        "document_ids": "unavailable: doc_id field is the literal unknown (no doc-disjoint claim)",
        "score_cells": len(retained) * len(selected),
        "matrix_sha256": csv_hash,
        "reference_system_entries_excluded": True,
        "caveat": "Retrospective released human-score panel, not new annotation",
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
