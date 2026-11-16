#!/usr/bin/env python3
"""Check paired external-evaluation trial logs and bootstrap randomization-only effects.

Usage: python paired_audit.py results/wmt22_ende --matrix inputs/wmt22_sqm_ende.csv --code-dir .
Both raw matrix and result logs stay on the user's computer. JSON report has
no item-level human annotations. This is NOT an independent acquisition replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

METHODS = ("common_active", "random_pair", "certificate_racing_v2",
           "published_inv_rank2", "published_confusion")
TARGET = "certificate_racing_v2"
MEASURES = {
    "cost_90": "mean_cost90", "cost_all": "mean_costall",
    "resolved_half": "mean_certified_half",
    "posthoc_top_weighted_tau_half": "posthoc_weighted_tau_half",
    "posthoc_top1_half": "posthoc_top1_half",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return float(values[lo] * (1 - (pos - lo)) + values[hi] * (pos - lo))


def paired_interval(base: list[float], candidate: list[float], *, mode: str,
                    resamples: int, seed: int) -> dict:
    if len(base) != len(candidate) or not base:
        raise ValueError("Both policies require matching nonempty replicate sets")
    n = len(base)

    def statistic(indices):
        mb = sum(base[i] for i in indices) / len(indices)
        mc = sum(candidate[i] for i in indices) / len(indices)
        if mode == "relative_saving_percent":
            return 100 * (1 - mc / mb) if mb > 0 else float("nan")
        if mode == "absolute_saving_cells":
            return mb - mc
        if mode == "difference":
            return mc - mb
        raise ValueError(f"Unknown statistic: {mode}")

    observed = statistic(range(n))
    rng = random.Random(seed)
    draws = [statistic([rng.randrange(n) for _ in range(n)]) for _ in range(resamples)]
    if not math.isfinite(observed) or not all(math.isfinite(x) for x in draws):
        raise ValueError("Nonfinite bootstrap statistic")
    return {
        "estimate": observed,
        "percentile_95_interval": [percentile(draws, .025), percentile(draws, .975)],
        "bootstrap_resamples": resamples,
        "interpretation": "Randomization-only, conditional on the fixed human-score matrix; not an independent-language interval",
    }


def verify_and_load(directory: Path, matrix: Path | None, code_dir: Path | None):
    manifest = json.loads((directory / "manifest.json").read_text())
    completion = json.loads((directory / "completion.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    trial_path = directory / "trials.jsonl"
    if not completion.get("complete") or digest(trial_path) != completion.get("trials_sha256"):
        raise ValueError("Completion marker or trial checksum mismatch")
    trials = [json.loads(line) for line in trial_path.read_text().splitlines() if line.strip()]
    if len(trials) != completion.get("rows"):
        raise ValueError("Trial row count mismatch")
    if (matrix is not None and
            digest(matrix) != manifest["input_metadata"].get("input_sha256")):
        raise ValueError("Normalized matrix does not match archived input checksum")
    if code_dir is not None:
        for filename, expected in manifest.get("source_sha256", {}).items():
            source_path = code_dir / filename
            if not source_path.exists() or digest(source_path) != expected:
                raise ValueError(f"Source checksum mismatch: {filename}")

    groups = defaultdict(dict)
    seed_sets = defaultdict(set)
    for row in trials:
        scenario = row["scenario"]
        key = (int(row["replicate"]), row["policy"])
        if key in groups[scenario] or row["policy"] not in METHODS:
            raise ValueError("Duplicate or unrecognized policy/replicate")
        if not (0 <= row["resolved_half"] <= 1 and 0 <= row["resolved_end"] <= 1):
            raise ValueError("Invalid certified-pair fraction")
        possible = row["items"] * row["systems"]
        total_pairs = row["systems"] * (row["systems"] - 1) // 2
        if row["cost_spent"] > possible or not 0 <= row["wrong_certificates"] <= total_pairs:
            raise ValueError("Invalid score-cell cost or pair decision count")
        if bool(row["any_wrong"]) != (row["wrong_certificates"] > 0):
            raise ValueError("Inconsistent error indicator")
        if row["reached_90"] and row["cost_90"] > row["cost_spent"]:
            raise ValueError("90-percent event cost exceeds total spent")
        if row["reached_all"] and row["cost_all"] > row["cost_spent"]:
            raise ValueError("All-pairs event cost exceeds total spent")
        groups[scenario][key] = row
        seed_sets[(scenario, int(row["replicate"]))].add(int(row["seed"]))

    if any(len(seeds) != 1 for seeds in seed_sets.values()):
        raise ValueError("Methods lack matched replicate seeds")

    for scenario, records in groups.items():
        reps = sorted({rep for rep, _ in records})
        if set(records) != {(rep, method) for rep in reps for method in METHODS}:
            raise ValueError("Incomplete policy/replicate grid")
        expected_methods = summary.get(scenario, {})
        if set(expected_methods) != set(METHODS):
            raise ValueError("Missing methods from published summary")
        for method in METHODS:
            rows = [records[(rep, method)] for rep in reps]
            expected = expected_methods[method]
            if len(rows) != expected["runs"]:
                raise ValueError("Summary repetition count mismatch")
            for column, summary_key in MEASURES.items():
                got = statistics.fmean(float(r[column]) for r in rows)
                if not math.isclose(got, float(expected[summary_key]), rel_tol=1e-11, abs_tol=1e-10):
                    raise ValueError(f"Summary arithmetic mismatch: {scenario} {method} {column}")
            if sum(int(r["any_wrong"]) for r in rows) != expected["runs_with_wrong_certificates"]:
                raise ValueError("Summary certificate-error count mismatch")
    return manifest, groups


def evaluate(directory: Path, *, matrix: Path | None = None,
             code_dir: Path | None = None, resamples: int = 10000,
             seed: int = 2026111502):
    if resamples < 100:
        raise ValueError("At least 100 bootstrap resamples required")
    manifest, groups = verify_and_load(directory, matrix, code_dir)
    report = {
        "status": "PASS: input/trial/source checks as supplied; no second policy replay",
        "source_matrix_sha256": manifest.get("input_metadata", {}).get("input_sha256"),
        "matrix_bytes_verified": matrix is not None,
        "runner_source_hashes_verified": code_dir is not None,
        "bootstrap_seed": seed,
        "cohorts": {},
        "interpretation": (
            "Intervals quantify randomized-order variability on one fixed released score "
            "matrix; repetitions are not independent languages, domains, systems or raters. "
            "Published acquisition rules are adapted for pairwise certification. "
            "Posthoc top-1 and weighted tau are not certified decisions. "
            "This script audits logs, not the acquisition code by independent replay."
        ),
    }
    for scenario, records in sorted(groups.items()):
        reps = sorted({rep for rep, _ in records})
        target = [records[(rep, TARGET)] for rep in reps]
        comparisons = {}

        def metric(rows, column):
            return [float(row[column]) for row in rows]

        for i, method in enumerate(METHODS):
            if method == TARGET:
                continue
            base = [records[(rep, method)] for rep in reps]
            comparisons[method] = {
                "cost90_saving_percent": paired_interval(
                    metric(base, "cost_90"), metric(target, "cost_90"),
                    mode="relative_saving_percent", resamples=resamples, seed=seed + i * 101),
                "cost90_saving_cells": paired_interval(
                    metric(base, "cost_90"), metric(target, "cost_90"),
                    mode="absolute_saving_cells", resamples=resamples, seed=seed + i * 101 + 1),
                "posthoc_top1_difference_v2_minus_baseline": paired_interval(
                    metric(base, "posthoc_top1_half"), metric(target, "posthoc_top1_half"),
                    mode="difference", resamples=resamples, seed=seed + i * 101 + 2),
                "posthoc_weighted_tau_difference_v2_minus_baseline": paired_interval(
                    metric(base, "posthoc_top_weighted_tau_half"),
                    metric(target, "posthoc_top_weighted_tau_half"),
                    mode="difference", resamples=resamples, seed=seed + i * 101 + 3),
            }
        report["cohorts"][scenario] = {
            "replicates": len(reps), "total_trial_rows": len(records),
            "wrong_certificates": sum(int(row["wrong_certificates"]) for row in records.values()),
            "comparisons": comparisons,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--matrix", type=Path, help="The exact normalized input CSV")
    parser.add_argument("--code-dir", type=Path, help="Unmodified source folder used for the run")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026111502)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate(args.directory, matrix=args.matrix, code_dir=args.code_dir,
                      resamples=args.resamples, seed=args.seed)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for cohort, data in result["cohorts"].items():
        print(f"{cohort}: {data['replicates']} matched runs, {data['wrong_certificates']} wrong certificates")
        for method, values in data["comparisons"].items():
            cost = values["cost90_saving_percent"]
            top = values["posthoc_top1_difference_v2_minus_baseline"]
            print(f"  vs {method}: cost90 saving {cost['estimate']:+.2f}% "
                  f"[{cost['percentile_95_interval'][0]:+.2f}, "
                  f"{cost['percentile_95_interval'][1]:+.2f}]; "
                  f"posthoc top1 difference {top['estimate']:+.2f} "
                  f"[{top['percentile_95_interval'][0]:+.2f}, "
                  f"{top['percentile_95_interval'][1]:+.2f}]")
    print(result["status"])


if __name__ == "__main__":
    main()
