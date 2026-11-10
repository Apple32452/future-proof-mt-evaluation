#!/usr/bin/env python3
"""Prepare WMT26 cESA complete-case panels without looking at target outcomes.

The system panel is fixed before running any sampling outcomes: six named 2026
target systems, and twelve development systems chosen ONLY by the availability
of valid human judgments on the same items.  The official WMT26 reference
builder supplies the RESET, screen-validity, reliability and averaging rules.

WARNING: WMT26's dynamic human-evaluation design makes coverage non-random.
The resulting estimand is performance on a jointly annotated item core, not on
all of WMT26; the manifest documents the retained fraction explicitly.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path


ANNOTATIONS_SHA256 = "6023795cc592864a9d67ed6daaaada283041d0a8c87c4bfb4c3e5ae276c73b28"
WMT26_REPO_COMMIT = "63aa5d24311a7e7e3e8173528f3771c7e8b4be78"

# Model identities are fixed a priori.  Their realized cESA scores are NOT used
# to choose the target cohort, the development panel, or language eligibility.
TARGETS = (
    "GPT 5.5",
    "Gemini 3.1 Pro",
    "DeepSeek V4 Pro",
    "Gemma 4 - 31B",
    "Mistral Medium 3.5",
    "Qwen 3.6 - 27b",
)

EXCLUDED_DEVELOPMENT = frozenset(("PseudoRef", "Human (from scratch)", "Human (postediting)"))
DEVELOPMENT_COUNT = 12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def official_module(repo: Path):
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if revision != WMT26_REPO_COMMIT:
        raise ValueError(f"Expected frozen WMT26 commit {WMT26_REPO_COMMIT}; got {revision}")
    filename = repo / "humeval" / "07-build_synthetic_references.py"
    spec = importlib.util.spec_from_file_location("wmt26_reference_filter", filename)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load WMT26 human-judgment processing rules: {filename}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_panel(coverage: dict[str, set], targets: tuple[str, ...] = TARGETS,
                 n_dev: int = DEVELOPMENT_COUNT):
    """Select development models solely from observed-score MASKS, not values."""
    absent = [s for s in targets if s not in coverage]
    if absent:
        return [], set(), set(), absent
    target_core = set.intersection(*(coverage[t] for t in targets))
    available = [s for s in coverage if s not in targets and s not in EXCLUDED_DEVELOPMENT]
    available.sort(key=lambda s: (-len(coverage[s] & target_core), s))
    selected = available[:n_dev]
    if len(selected) != n_dev:
        return [], target_core, set(), [f"only {len(selected)} development systems"]
    core = set(target_core)
    for s in selected:
        core.intersection_update(coverage[s])
    return selected, target_core, core, []


def observed_panels(official, filtered: dict) -> tuple[dict, dict, dict]:
    evidence, _, unexpected = official.extract_item_system_evidence(filtered)
    if unexpected:
        raise ValueError(f"Unexpected off-screen judgments after official filtering: {unexpected}")
    source_by_item = {}
    domain_by_item = {}
    for raw_direction, rows in filtered.items():
        direction = official.normalise_direction(raw_direction)
        for row in rows:
            for item in row["item"]:
                domain, doc, segment = official.split_item_id(item["item_id"])
                key = (direction, f"{domain}_###_{doc}", segment)
                source = item.get("src")
                if not isinstance(source, str) or not source:
                    raise ValueError(f"Missing source text: {direction} / {item['item_id']}")
                if key in source_by_item and source_by_item[key] != source:
                    raise ValueError(f"Inconsistent source on repeated cESA screens: {key}")
                source_by_item[key] = source
                domain_by_item[key] = domain

    by_direction: dict[str, dict] = collections.defaultdict(lambda: collections.defaultdict(dict))
    for (direction, doc, segment, system), record in evidence.items():
        item_key = (doc, segment)
        key = (direction, doc, segment)
        if key not in source_by_item:
            raise ValueError(f"Missing source for scored item: {key}")
        by_direction[direction][item_key][system] = float(record["score"])
    return by_direction, source_by_item, domain_by_item


def create_matrices(by_direction: dict, source_by_item: dict, domain_by_item: dict,
                    destination: Path, target_models: tuple[str, ...] = TARGETS) -> list[dict]:
    destination.mkdir(parents=True, exist_ok=False)
    summaries = []
    for direction in sorted(by_direction):
        items = by_direction[direction]
        coverage: dict[str, set] = collections.defaultdict(set)
        for key, observed in items.items():
            for system in observed:
                coverage[system].add(key)
        dev, target_core, core, blockers = select_panel(coverage, target_models)
        ordered_core = sorted(core, key=lambda key: (key[0], key[1]))
        n_docs = len({doc for doc, _ in core})
        row = {
            "language_pair": direction,
            "n_scored_items": len(items),
            "n_scored_systems": len(coverage),
            "n_target_overlap_items": len(target_core),
            "n_complete_panel_items": len(core),
            "n_complete_panel_documents": n_docs,
            "retained_fraction_of_scored_items": len(core) / len(items) if items else 0,
            "target_models": list(target_models),
            "development_models_by_coverage": dev,
            "excluded_due_to_missing_model": blockers,
            "human_score_type": "WMT26 contrastive ESA (cESA; 0-100)",
            "full_wmt26_estimand": False,
        }
        if core and not blockers:
            filename = f"{direction}_human_scores.csv"
            with (destination / filename).open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sentence_id", "source_text", "document_id", "domain", *target_models, *dev])
                for doc, segment in ordered_core:
                    key = (direction, doc, segment)
                    scores = items[(doc, segment)]
                    writer.writerow([f"{direction}_#_{doc}_#_{segment}", source_by_item[key],
                                     f"{direction}_#_{doc}", domain_by_item[key],
                                     *[scores[s] for s in target_models], *[scores[s] for s in dev]])
            row["matrix_file"] = filename
            row["matrix_sha256"] = sha256_file(destination / filename)
        else:
            row["matrix_file"] = None
        summaries.append(row)
        print(f"{direction}: {len(core)} jointly scored segments, {n_docs} documents", flush=True)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--wmt26-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new output directory; existing files will not be overwritten")
    actual_hash = sha256_file(args.annotations)
    if actual_hash != ANNOTATIONS_SHA256:
        parser.error(f"Expected final WMT26 annotation SHA-256 {ANNOTATIONS_SHA256}; got {actual_hash}")
    official = official_module(args.wmt26_repo)
    raw = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or len(raw) != 23:
        parser.error("Expected 23 WMT26 language directions in finalized annotations")
    reset = official.apply_resets_and_remove_noncontent(raw)
    screen_valid, excluded_screen = official.remove_non_displayed_annotations(reset)
    filtered, removed_users, _ = official.remove_low_reliability_users(screen_valid)
    by_direction, sources, domains = observed_panels(official, filtered)
    summaries = create_matrices(by_direction, sources, domains, args.out)
    manifest = {
        "status": "built",
        "wmt26_repo_commit": WMT26_REPO_COMMIT,
        "official_filter_source": "humeval/07-build_synthetic_references.py",
        "final_annotation_sha256": actual_hash,
        "target_models_are_fixed_before_outcomes": True,
        "development_selection_uses_only_observation_masks": True,
        "full_wmt26_estimand": False,
        "source_warning": "WMT26 dynamically assigned annotation coverage by model performance; complete-case cores are selected and are not the full benchmark",
        "scoring_warning": "WMT26 cESA scores should not be directly compared numerically with WMT25 human scores",
        "removed_offscreen_observations": excluded_screen,
        "removed_low_reliability_users_by_language": removed_users,
        "language_panels": summaries,
    }
    (args.out / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (args.out / "coverage_audit.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ("language_pair", "n_scored_items", "n_scored_systems", "n_target_overlap_items",
                  "n_complete_panel_items", "n_complete_panel_documents", "retained_fraction_of_scored_items")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for panel in summaries:
            writer.writerow({field: panel[field] for field in fields})


if __name__ == "__main__":
    main()
