#!/usr/bin/env python3
"""Build full-human-universe score matrices for the quality-shift experiment.

This is deliberately not the human+MetricX intersection used by some older
benchmarks. All source items and all observed, human-scored MT systems for a
requested language pair must form a complete matrix; otherwise this script
stops rather than silently dropping items, systems, or imputing scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LANGUAGE_PAIRS = (
    "cs-de_DE", "en-cs_CZ", "cs-uk_UA", "en-sr_Cyrl_RS", "en-is_IS",
    "en-ru_RU", "en-ar_EG", "en-uk_UA", "en-zh_CN", "en-et_EE",
    "en-bho_IN", "en-ja_JP", "en-mas_KE", "en-it_IT",
)
METADATA_COLUMNS = ("sentence_id", "source_text")
RAW_PATH = "data/wmt25-genmt-humeval.jsonl"
REPOSITORY = "wmt-conference/wmt25-general-mt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def read_matrices(data: Path, language_pairs: tuple[str, ...]) -> tuple[
        dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Stream the release in its original order, with explicit completeness checks."""
    if not language_pairs or len(set(language_pairs)) != len(language_pairs):
        raise ValueError("Provide at least one language pair, without duplicates.")
    records: dict[str, list[dict[str, Any]]] = {lp: [] for lp in language_pairs}
    systems: dict[str, set[str]] = {lp: set() for lp in language_pairs}
    documents: dict[str, set[str]] = {lp: set() for lp in language_pairs}
    seen_ids: set[str] = set()
    annotation_counts = {lp: [] for lp in language_pairs}
    skipped_null = {lp: 0 for lp in language_pairs}
    ref_cells = {lp: 0 for lp in language_pairs}

    with data.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            item_id = row.get("doc_id")
            if not isinstance(item_id, str):
                raise ValueError(f"Missing string doc_id on line {line_number}")
            lp = item_id.split("_#_", 1)[0]
            if lp not in records:
                continue
            components = item_id.split("_#_")
            if len(components) != 4 or not all(components):
                raise ValueError(f"Unexpected WMT item-ID schema: {item_id!r}")
            if item_id in seen_ids:
                raise ValueError(f"Duplicate item ID: {item_id}")
            seen_ids.add(item_id)
            source = row.get("src_text")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"Missing or empty source text: {item_id}")
            score_map = row.get("scores")
            if not isinstance(score_map, dict):
                raise ValueError(f"Missing score dictionary: {item_id}")
            record: dict[str, Any] = {"sentence_id": item_id, "source_text": source}
            for system, annotations in score_map.items():
                if system == "refA":
                    ref_cells[lp] += 1
                    continue
                if not isinstance(system, str) or system in METADATA_COLUMNS:
                    raise ValueError(f"Invalid system identifier in {item_id}")
                systems[lp].add(system)
                if not isinstance(annotations, list):
                    raise ValueError(f"Annotations must be a list: {item_id}/{system}")
                values: list[float] = []
                for annotation in annotations:
                    if not isinstance(annotation, dict):
                        raise ValueError(f"Invalid annotation: {item_id}/{system}")
                    value = annotation.get("score")
                    if value is None:
                        skipped_null[lp] += 1
                        continue
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(f"Non-numeric score: {item_id}/{system}")
                    if not math.isfinite(value):
                        raise ValueError(f"Non-finite score: {item_id}/{system}")
                    values.append(value)
                annotation_counts[lp].append(len(values))
                if values:
                    # Match src/build_score_matrix.py exactly, including arithmetic.
                    record[system] = sum(values) / len(values)
            records[lp].append(record)
            documents[lp].add(item_id.rsplit("_#_", 1)[0])

    frames: dict[str, pd.DataFrame] = {}
    audit: list[dict[str, Any]] = []
    errors: list[str] = []
    for lp in language_pairs:
        ordered_systems = sorted(systems[lp])
        frame = pd.DataFrame(records[lp]).reindex(
            columns=list(METADATA_COLUMNS) + ordered_systems)
        scores = frame[ordered_systems].to_numpy(dtype=float)
        missing = int(np.isnan(scores).sum())
        if not records[lp]:
            errors.append(f"{lp}: no human-evaluation items")
        elif len(ordered_systems) < 12:
            errors.append(f"{lp}: only {len(ordered_systems)} systems (need at least 12)")
        elif missing:
            errors.append(f"{lp}: {missing} missing score cells; no filtering or imputation applied")
        elif not np.isfinite(scores).all():
            errors.append(f"{lp}: non-finite matrix values")
        if len(documents[lp]) < 2:
            errors.append(f"{lp}: fewer than two document groups")
        counts = annotation_counts[lp]
        audit.append({
            "language_pair": lp,
            "n_items": len(frame),
            "n_documents": len(documents[lp]),
            "n_systems": len(ordered_systems),
            "n_missing_cells": missing,
            "n_annotations_used": sum(counts),
            "min_annotations_per_present_cell": min(counts) if counts else 0,
            "max_annotations_per_present_cell": max(counts) if counts else 0,
            "n_null_annotations_skipped": skipped_null[lp],
            "n_reference_cells_excluded": ref_cells[lp],
            "min_score": float(np.nanmin(scores)) if scores.size else None,
            "max_score": float(np.nanmax(scores)) if scores.size else None,
        })
        frames[lp] = frame
    if errors:
        raise ValueError("Cannot build strict complete matrices:\n" + "\n".join(errors))
    return frames, audit


def check_pilot_matrix(frame: pd.DataFrame, reference: Path) -> dict[str, Any]:
    """Verify source order, text, system names and exact score values against pilot."""
    prior = pd.read_csv(reference, keep_default_na=False)
    if list(frame.columns) != list(prior.columns):
        raise ValueError("Czech-German system columns/order differ from the committed pilot.")
    for column in METADATA_COLUMNS:
        if frame[column].tolist() != prior[column].tolist():
            raise ValueError(f"Czech-German {column} values/order differ from the committed pilot.")
    scores = [c for c in frame if c not in METADATA_COLUMNS]
    if not np.array_equal(frame[scores].to_numpy(dtype=float),
                          prior[scores].to_numpy(dtype=float)):
        raise ValueError("Czech-German scores differ from the committed pilot.")
    return {"status": "exact data match", "reference_path": str(reference.resolve()),
            "reference_sha256": sha256_file(reference)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Released WMT25 human-evaluation JSONL (resolved LFS data).")
    parser.add_argument("--out", type=Path, required=True,
                        help="New empty output directory; existing results are never overwritten.")
    parser.add_argument("--language-pairs", nargs="+", default=list(LANGUAGE_PAIRS))
    parser.add_argument("--source-revision", default=None,
                        help="Upstream Git commit from which the data were retrieved, if known.")
    parser.add_argument("--pilot-matrix", type=Path,
                        default=Path(__file__).resolve().parents[2] /
                        "results/score_matrices/cs-de_DE_human_scores.csv")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("--data must be a readable local file.")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error("--out must be absent or empty; choose a new path.")
    with args.data.open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            parser.error("--data is an unresolved Git LFS pointer, not human-evaluation records.")
    frames, audit = read_matrices(args.data, tuple(args.language_pairs))
    pilot_check = None
    if "cs-de_DE" in frames:
        if not args.pilot_matrix.is_file():
            parser.error("Czech-German pilot reference is required for the reproducibility check.")
        pilot_check = check_pilot_matrix(frames["cs-de_DE"], args.pilot_matrix)
    args.out.mkdir(parents=True, exist_ok=True)
    for row in audit:
        lp = row["language_pair"]
        destination = args.out / f"{lp}_human_scores.csv"
        frames[lp].to_csv(destination, index=False, lineterminator="\n")
        row["matrix_file"] = destination.name
        row["matrix_sha256"] = sha256_file(destination)
        row["systems"] = [c for c in frames[lp] if c not in METADATA_COLUMNS]
        print(f"{lp}: {row['n_items']} items, {row['n_documents']} documents, "
              f"{row['n_systems']} systems, complete", flush=True)
    pd.DataFrame([{k: v for k, v in row.items() if k != "systems"} for row in audit]) \
        .to_csv(args.out / "matrix_audit.csv", index=False)
    revision = args.source_revision or "main"
    manifest = {
        "status": "complete",
        "input_path": str(args.data.resolve()),
        "input_sha256": sha256_file(args.data),
        "input_bytes": args.data.stat().st_size,
        "source_repository": REPOSITORY,
        "source_revision": args.source_revision,
        "source_url": f"https://github.com/{REPOSITORY}/blob/{revision}/{RAW_PATH}",
        "universe": "full human evaluation; no MetricX coverage filtering",
        "annotation_rule": "mean of non-null finite annotations, sum(values)/len(values)",
        "system_exclusions": ["refA"],
        "missing_score_policy": "error; no item/system filtering or imputation",
        "item_order": "original order in the source JSONL",
        "system_order": "lexicographic",
        "document_rule": "doc_id.rsplit('_#_', 1)[0]",
        "item_id_schema": "language_pair_#_domain_#_document_#_segment",
        "pilot_reproduction": pilot_check,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "builder_sha256": sha256_file(Path(__file__)),
        "language_pairs": audit,
    }
    write_json(args.out / "matrix_manifest.json", manifest)
    print(f"Saved {len(frames)} matrices to {args.out}", flush=True)


if __name__ == "__main__":
    main()
