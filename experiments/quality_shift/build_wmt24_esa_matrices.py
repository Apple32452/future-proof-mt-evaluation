#!/usr/bin/env python3
"""Build frozen, coverage-only WMT24 ESA machine-system score panels.

No quality/ranking statistic is used to choose the ten included systems or
directions. Repeated annotation actions by one worker count as one rating.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_wmt26_transfer_matrices import sha256_file
from run_wmt26_external import save_json


SOURCE_COMMIT = "fabe1f81d65449b3e3f9f8d37da280fa6cf3650a"
WAVES = {
    "esa_generalMT2024_wave1.csv": "7e820667be121fa92b22dbdb1ce55ad30bb0ad9d505bdb79811f386583055cb6",
    "esa_generalMT2024_wave2.csv": "195b00fc3c7233fa4fa674b9d888ca4cc6fb02b2138a571cd7b75951aabaf9f2",
    "esa_generalMT2024_wave3.csv": "70c8ff2df883ddf32dea7b6cef931912879f53d08dfeebfb7ffb7a1f0d1d0482",
}
LANGUAGES = ("ces-ukr", "eng-ces", "eng-hin", "eng-isl", "eng-jpn",
             "eng-rus", "eng-spa", "eng-ukr", "eng-zho")
RAW_COLUMNS = ("annotator", "system", "item_id", "side", "source_lang",
               "target_lang", "score", "document", "is_reference", "spans",
               "start", "end")
SYSTEMS = 10


def build(source_dir: Path, output: Path) -> dict:
    waves = []
    for name, digest in WAVES.items():
        file = source_dir / name
        if not file.is_file() or sha256_file(file) != digest:
            raise ValueError(f"Unverified official WMT24 annotation wave: {file}")
        waves.append(pd.read_csv(file, header=None, names=RAW_COLUMNS,
                                 dtype={"item_id": str}, low_memory=False))
    raw = pd.concat(waves, ignore_index=True)
    if raw[list(RAW_COLUMNS)].isna().any().any():
        # Start/end timestamps may legitimately be missing; annotation keys may not.
        if raw[list(RAW_COLUMNS[:9])].isna().any().any():
            raise ValueError("Missing official ESA annotation keys or scores")
    x = raw[(raw.side == "TGT") & raw.document.astype(str).str.startswith("test-")
            & (raw.system != "refA")
            & (raw.is_reference.astype(str).str.lower() == "false")].copy()
    x["score"] = pd.to_numeric(x.score, errors="raise")
    if not np.isfinite(x.score).all() or not x.score.between(0, 100).all():
        raise ValueError("Invalid ESA score in official target-language test annotations")
    x["language_pair"] = x.source_lang.astype(str) + "-" + x.target_lang.astype(str)
    x["document"] = x.document.astype(str).str.replace(
        r"(#(?:dup|incomplete))+$", "", regex=True)
    x["item_id"] = x.item_id.astype(str)
    doc_counts = x.groupby(["language_pair", "item_id"]).document.nunique()
    if doc_counts.max() != 1:
        raise ValueError("Conflicting canonical document IDs for the same segment")
    repeat_group = ["language_pair", "item_id", "document", "system", "annotator"]
    n_repeat_actions = len(x) - len(x.groupby(repeat_group))
    # Average repeat actions WITHIN annotator, then average independent annotators.
    once = x.groupby(repeat_group, as_index=False).score.mean()
    pooled = once.groupby(["language_pair", "item_id", "document", "system"],
                          as_index=False).score.mean()
    prepared = []
    panels = []
    for language in LANGUAGES:
        section = pooled[pooled.language_pair == language]
        counts = section.groupby("system").item_id.nunique().to_dict()
        panel = sorted(counts, key=lambda s: (-counts[s], s))[:SYSTEMS]
        if len(panel) != SYSTEMS or "refA" in panel or len(set(panel)) != SYSTEMS:
            raise ValueError(f"Insufficient nonreference coverage: {language}")
        matching = section[section.system.isin(panel)]
        matrix = matching.pivot(index=["document", "item_id"],
                                columns="system", values="score")
        complete = matrix.dropna(subset=panel).reset_index()
        complete = complete.sort_values(["document", "item_id"], kind="stable")
        if len(complete) < 250 or complete.document.nunique() < 50:
            raise ValueError(f"Insufficient eligible complete-case items: {language}")
        if not np.isfinite(complete[panel].to_numpy(float)).all():
            raise ValueError("Nonfinite complete-case quality scores")
        doc_id = language + "_#_" + complete.document.astype(str)
        result = pd.DataFrame({
            "sentence_id": doc_id + "_#_" + complete.item_id.astype(str),
            "document_id": doc_id,
            "domain": "WMT24-news-ESA",
        })
        for name in panel:
            result[name] = complete[name].to_numpy(float)
        if result.sentence_id.duplicated().any():
            raise ValueError(f"Duplicate scored segments: {language}")
        prepared.append((language, result))
        panels.append({
            "language_pair": language,
            "model_names": panel,
            "n_scored_items": int(section.item_id.nunique()),
            "n_complete_panel_items": int(len(result)),
            "n_complete_panel_documents": int(result.document_id.nunique()),
            "n_machine_systems_available": len(counts),
            "coverage_by_selected_model": {p: int(counts[p]) for p in panel},
            "selection_rule": "ten highest annotation-coverage systems; alphabetic ties",
            "human_score_type": "WMT24 ESA (0-100)",
        })
    output.mkdir(parents=True, exist_ok=False)
    for (language, frame), panel in zip(prepared, panels):
        filename = f"{language}_human_scores.csv"
        frame.to_csv(output / filename, index=False)
        panel["matrix_file"] = filename
        panel["matrix_sha256"] = sha256_file(output / filename)
        print(f"{language}: {len(frame)} complete items, "
              f"{frame.document_id.nunique()} documents", flush=True)
    manifest = {
        "status": "built", "official_source": "wmt-conference/wmt24-news-systems",
        "official_source_commit": SOURCE_COMMIT, "wave_sha256": WAVES,
        "annotator_duplicate_actions_averaged": int(n_repeat_actions),
        "n_raw_annotation_actions": int(len(raw)),
        "n_valid_machine_test_target_actions": int(len(x)),
        "document_suffixes_removed": ["#dup", "#incomplete"],
        "averaging_rule": "average within annotator-item-system, then across annotators",
        "direction_selection": list(LANGUAGES), "language_panels": panels,
    }
    save_json(output / "matrix_manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    build(args.source_dir, args.out)


if __name__ == "__main__":
    main()
