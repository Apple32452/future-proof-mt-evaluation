"""Evaluate one frozen allocation-only revision on previously exposed data.

Original audited baseline records are reused, not overwritten. The same seeds,
matrices and inference are used. Full-data truths and top-two identities are
available only to this evaluator, never the acquisition controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

from certificate_racing_v2 import POLICY, run_certificate_race
from loader import load_development_csv
from pairwise_racing import POLICIES, UNKNOWN
from validate import SCENARIOS, FRACTIONS, synthetic_matrix, truth_signs, forced_pair_accuracy, binomial_interval

ROOT = Path(__file__).resolve().parent
ALL_POLICIES = (POLICY,) + POLICIES


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_top_pair(record, y):
    """Evaluation-only, highest two finite-matrix means; ID breaks exact ties."""
    a, b = np.triu_indices(y.shape[1], 1)
    top = sorted(range(y.shape[1]), key=lambda s: (-float(np.mean(y[:, s])), s))[:2]
    left, right = sorted(top)
    p = int(np.flatnonzero((a == left) & (b == right))[0])
    result = dict(record)
    result["top_two_pair"] = [left, right]
    result["top_two_true_sign"] = record["true_signs"][p]
    result["top_two_decision_cost"] = record["decision_costs"][p]
    result["top_two_decision_correct"] = record["decisions"][p] == record["true_signs"][p]
    result["top_two_resolved_at_half_budget"] = 0 <= record["decision_costs"][p] <= int(y.size * .5)
    return result


def evaluate_new(y, scenario, rep, seed, ratings_per_cell=None):
    n, k = y.shape
    checkpoints = [int(n * k * f) for f in FRACTIONS]
    queried = set()

    def query(row, system):
        if (row, system) in queried:
            raise AssertionError("duplicate charged cell")
        queried.add((row, system))
        return y[row, system]

    result = run_certificate_race(n, k, query, seed=seed, checkpoints=checkpoints)
    if result["cost_cells"] != len(queried):
        raise AssertionError("cost/query mismatch")
    truth = truth_signs(y)
    decisions = np.asarray(result["decisions"])
    known = decisions != UNKNOWN
    wrong = known & (decisions != truth)
    record = {
        "scenario": scenario, "replicate": rep, "seed": seed, "policy": POLICY,
        "items": n, "systems": k, "pairs": len(truth), "full_cost_cells": n * k,
        "cost_cells": result["cost_cells"],
        "human_rating_equivalents": result["cost_cells"] * ratings_per_cell if ratings_per_cell is not None else None,
        "any_wrong_certificate": bool(wrong.any()), "wrong_certificates": int(wrong.sum()),
        "complete": result["complete"], "resolved_fraction": float(known.mean()),
        "certified_fraction_auc": result["certified_fraction_auc"],
        "resolution_costs": result["resolution_costs"],
        "decisions": result["decisions"], "decision_costs": result["decision_costs"],
        "true_signs": truth.tolist(),
        "reasons_count": {r: result["reasons"].count(r) for r in ("mixture_betting", "deterministic_completion", "exact_census")},
        "snapshots": {}}
    for fraction, budget in zip(FRACTIONS, checkpoints):
        snap = result["snapshots"][str(budget)]
        d = np.array(snap["decisions"])
        resolved = d != UNKNOWN
        correct = resolved & (d == truth)
        record["snapshots"][str(fraction)] = {
            "budget_cells": budget, "actual_cost_cells": snap["cost_cells"],
            "resolved_fraction": float(resolved.mean()),
            "conditional_certified_accuracy": float(correct.sum() / resolved.sum()) if resolved.any() else None,
            "correct_certified_fraction_all_pairs": float(correct.mean()),
            "forced_pair_accuracy": forced_pair_accuracy(snap["sample_means"], truth)}
    return add_top_pair(record, y)


def cost90(record):
    value = record["resolution_costs"]["0.9"]
    return record["full_cost_cells"] if value is None else value


def summarize(records):
    result = {}
    for scenario in sorted({r["scenario"] for r in records}):
        rows = [r for r in records if r["scenario"] == scenario]
        by_policy = {p: sorted([r for r in rows if r["policy"] == p], key=lambda r: r["replicate"])
                     for p in ALL_POLICIES}
        metrics = {}
        for policy, values in by_policy.items():
            errors = sum(r["any_wrong_certificate"] for r in values)
            metrics[policy] = {
                "runs": len(values), "runs_with_any_wrong_certificate": errors,
                "error_rate": errors / len(values), "error_rate_ci95": binomial_interval(errors, len(values)),
                "mean_cost_90_percent": float(np.mean([cost90(r) for r in values])),
                "mean_cost_all_pairs": float(np.mean([r["cost_cells"] for r in values])),
                "complete_runs": sum(r["complete"] for r in values),
                "resolved_at_half_budget": float(np.mean([r["snapshots"]["0.5"]["resolved_fraction"] for r in values])),
                "forced_accuracy_at_half_budget": float(np.mean([r["snapshots"]["0.5"]["forced_pair_accuracy"] for r in values])),
                "mean_certified_fraction_auc": float(np.mean([r["certified_fraction_auc"] for r in values])),
                "mean_top_two_decision_cost": float(np.mean([r["top_two_decision_cost"] for r in values])),
                "top_two_resolved_at_half_budget": float(np.mean([r["top_two_resolved_at_half_budget"] for r in values])),
                "top_two_wrong_decision_runs": sum(not r["top_two_decision_correct"] for r in values)}
        comparisons = {}
        ours = by_policy[POLICY]
        for baseline in POLICIES:
            base = by_policy[baseline]
            assert [(r["replicate"], r["seed"]) for r in ours] == [(r["replicate"], r["seed"]) for r in base]
            x = np.array([cost90(r) for r in ours])
            z = np.array([cost90(r) for r in base])
            rng = np.random.default_rng(20261113)
            indices = rng.integers(0, len(x), (5000, len(x)))
            relative = 1 - x[indices].mean(axis=1) / z[indices].mean(axis=1)
            comparisons[baseline] = {
                "cost90_saving_fraction": float(1 - x.mean() / z.mean()),
                "paired_bootstrap_ci95": np.quantile(relative, [.025, .975]).tolist(),
                "scope": "paired simulation/randomization variability; not independent real-language replication"}
        result[scenario] = {"policies": metrics, "certificate_v2_vs": comparisons}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-csv", type=Path, default=ROOT / "inputs/cs-de_DE_human_scores.csv")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "validation_v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new output directory; previous results are never overwritten")
    old_manifest = json.loads((args.baseline_dir / "run_manifest.json").read_text())
    if sha256(ROOT / "protocol.json") != old_manifest["protocol_sha256"]:
        raise AssertionError("frozen v1 protocol changed")
    for name, checksum in old_manifest["source_hashes_before_run"].items():
        if sha256(ROOT / name) != checksum:
            raise AssertionError(f"frozen v1 source changed: {name}")
    reference = [json.loads(line) for line in (args.baseline_dir / "trials.jsonl").read_text().splitlines()]
    lookup = {(r["scenario"], r["replicate"], r["policy"]): r for r in reference}
    if len(reference) != 2480 or len(lookup) != 2480:
        raise AssertionError("expected the complete frozen v1 reference set")
    real_y, metadata = load_development_csv(args.real_csv)
    protocol = json.loads((ROOT / "protocol_v2.json").read_text())
    source_hashes = {p.name: sha256(p) for p in ROOT.glob("*.py")}
    manifest = {
        "protocol": protocol, "protocol_sha256": sha256(ROOT / "protocol_v2.json"),
        "source_hashes_before_run": source_hashes,
        "baseline_trials_sha256": sha256(args.baseline_dir / "trials.jsonl"),
        "baseline_manifest_sha256": sha256(args.baseline_dir / "run_manifest.json"),
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "real_input": metadata,
        "new_runs_planned": 620, "reference_runs_reused": 2480,
        "scope": "one post-v1 development revision; no untouched validation"}
    args.out.mkdir(parents=True)
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    all_rows, new_rows = [], []
    started = time.monotonic()
    with (args.out / "new_trials.jsonl").open("w") as new_stream, (args.out / "comparison_trials.jsonl").open("w") as combined:
        def record_case(y, scenario, rep, seed, ratings=None):
            new = evaluate_new(y, scenario, rep, seed, ratings)
            new_rows.append(new)
            new_stream.write(json.dumps(new) + "\n")
            new_stream.flush()
            group = [new]
            for policy in POLICIES:
                old = lookup[(scenario, rep, policy)]
                if old["seed"] != seed or old["true_signs"] != new["true_signs"]:
                    raise AssertionError("unpaired input or truth mismatch")
                group.append(add_top_pair(old, y))
            for row in group:
                all_rows.append(row)
                combined.write(json.dumps(row) + "\n")
            combined.flush()

        for scenario_id, scenario in enumerate(SCENARIOS):
            for rep in range(100):
                record_case(synthetic_matrix(scenario, rep), scenario, rep,
                            20261113 + scenario_id * 10000 + rep)
            print(json.dumps({"finished": scenario, "new_runs": len(new_rows),
                              "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
        for rep in range(20):
            record_case(real_y, "cs_de_development", rep, 2026111300 + rep,
                        metadata["human_ratings_per_cell"])
            print(json.dumps({"finished": "cs_de_development", "replicate": rep + 1,
                              "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
    for name, checksum in source_hashes.items():
        if sha256(ROOT / name) != checksum:
            raise AssertionError(f"source changed during validation: {name}")
    result = summarize(all_rows)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "completion.json").write_text(json.dumps({
        "status": "COMPLETE", "new_runs": len(new_rows), "reference_runs": len(reference),
        "new_trials_sha256": sha256(args.out / "new_trials.jsonl"),
        "elapsed_seconds": round(time.monotonic() - started, 2)}, indent=2) + "\n")
    print(json.dumps({"new_runs": len(new_rows), "reference_runs_reused": len(reference),
                      "out": str(args.out), "complete": True}), flush=True)


if __name__ == "__main__":
    main()
