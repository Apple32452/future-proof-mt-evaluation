"""Frozen synthetic validation and optional previously-exposed MT replay.

No independent test-data claim is made. Results include all trials and policies.
Only this outer validator, never the controller, accesses the full score matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.stats import beta

from loader import load_development_csv
from pairwise_racing import POLICIES, UNKNOWN, run_race


ROOT = Path(__file__).resolve().parent
SCENARIOS = ("well_separated", "one_near_tie", "all_tied", "heterogeneous_noise",
             "strong_item_effect", "weak_separation")
FRACTIONS = (0.1, 0.25, 0.5, 0.75)


def synthetic_matrix(scenario: str, replicate: int, seed=20261113):
    scenario_id = SCENARIOS.index(scenario)
    rng = np.random.default_rng(np.random.SeedSequence([seed, scenario_id, replicate]))
    n, k = 120, 6
    if scenario == "well_separated":
        y = np.linspace(.1, .9, k) + rng.normal(0, .02, (n, 1)) + rng.normal(0, .02, (n, k))
    elif scenario == "one_near_tie":
        y = np.array([.2, .38, .55, .72, .85, .852]) + rng.normal(0, .04, (n, k))
    elif scenario == "all_tied":
        # Identical finite means, different item patterns: exact global null.
        values = np.r_[np.zeros(n // 2), np.ones(n // 2)]
        return np.column_stack([rng.permutation(values) for _ in range(k)])
    elif scenario == "heterogeneous_noise":
        y = np.array([.2, .35, .5, .65, .8, .9]) + rng.normal(size=(n, k)) * np.array([.02, .25, .02, .18, .01, .08])
    elif scenario == "strong_item_effect":
        y = rng.uniform(.05, .4, (n, 1)) + np.linspace(.05, .4, k) + rng.normal(0, .01, (n, k))
    elif scenario == "weak_separation":
        y = np.linspace(.48, .52, k) + rng.normal(0, .18, (n, k))
    else:
        raise ValueError(scenario)
    # Clipping defines the synthetic population itself, not an undisclosed
    # transformation selected after inspecting real target data.
    return np.clip(y, 0, 1)


def truth_signs(y):
    a, b = np.triu_indices(y.shape[1], 1)
    means = np.mean(y, axis=0)
    gaps = means[a] - means[b]
    signs = np.sign(gaps).astype(int)
    for p in np.flatnonzero(np.abs(gaps) < 1e-12):
        # Same exact arithmetic convention used only to evaluate census ties.
        from fractions import Fraction
        delta = sum((Fraction.from_float(float(v)) for v in y[:, a[p]]), Fraction())
        delta -= sum((Fraction.from_float(float(v)) for v in y[:, b[p]]), Fraction())
        signs[p] = int(delta > 0) - int(delta < 0)
    return signs


def forced_pair_accuracy(sample_means, true_signs):
    sample_means = np.asarray(sample_means)
    a, b = np.triu_indices(len(sample_means), 1)
    pred = np.sign(sample_means[a] - sample_means[b]).astype(int)
    credit = (pred == true_signs).astype(float)
    credit[(pred == 0) & (true_signs != 0)] = 0.5
    return float(credit.mean())


def evaluate_one(y, scenario, replicate, seed, policy, ratings_per_cell=None):
    n, k = y.shape
    checkpoints = [int(n * k * f) for f in FRACTIONS]
    requested = set()

    def query(i, s):
        if (i, s) in requested:
            raise AssertionError("duplicate charged query")
        requested.add((i, s))
        return y[i, s]

    result = run_race(n, k, query, policy=policy, seed=seed, checkpoints=checkpoints)
    if result["cost_cells"] != len(requested):
        raise AssertionError("query cost mismatch")
    truth = truth_signs(y)
    decisions = np.asarray(result["decisions"])
    resolved = decisions != UNKNOWN
    wrong = resolved & (decisions != truth)
    record = {"scenario": scenario, "replicate": replicate, "seed": seed,
              "policy": policy, "items": n, "systems": k, "pairs": len(truth),
              "full_cost_cells": n * k,
              "cost_cells": result["cost_cells"],
              "human_rating_equivalents": result["cost_cells"] * ratings_per_cell if ratings_per_cell is not None else None,
              "any_wrong_certificate": bool(np.any(wrong)),
              "wrong_certificates": int(wrong.sum()),
              "complete": result["complete"],
              "resolved_fraction": float(resolved.mean()),
              "certified_fraction_auc": result["certified_fraction_auc"],
              "resolution_costs": result["resolution_costs"],
              "decisions": result["decisions"],
              "decision_costs": result["decision_costs"],
              "true_signs": truth.tolist(),
              "reasons_count": {reason: result["reasons"].count(reason) for reason in ("mixture_betting", "deterministic_completion", "exact_census")},
              "snapshots": {}}
    for f, budget in zip(FRACTIONS, checkpoints):
        snap = result["snapshots"][str(budget)]
        d = np.array(snap["decisions"])
        known = d != UNKNOWN
        correct = known & (d == truth)
        record["snapshots"][str(f)] = {
            "budget_cells": budget, "actual_cost_cells": snap["cost_cells"],
            "resolved_fraction": float(known.mean()),
            "conditional_certified_accuracy": float(correct.sum() / known.sum()) if known.any() else None,
            "correct_certified_fraction_all_pairs": float(correct.mean()),
            "forced_pair_accuracy": forced_pair_accuracy(snap["sample_means"], truth)}
    return record


def binomial_interval(errors, n, level=.95):
    alpha = 1 - level
    lo = float(beta.ppf(alpha / 2, errors, n - errors + 1)) if errors else 0.0
    hi = float(beta.ppf(1 - alpha / 2, errors + 1, n - errors)) if errors < n else 1.0
    return [lo, hi]


def summarize(records):
    results = {}
    for scenario in sorted({r["scenario"] for r in records}):
        rows = [r for r in records if r["scenario"] == scenario]
        by_policy = {}
        for policy in POLICIES:
            p = sorted([r for r in rows if r["policy"] == policy], key=lambda x: x["replicate"])
            errors = sum(r["any_wrong_certificate"] for r in p)
            by_policy[policy] = {
                "runs": len(p), "runs_with_any_wrong_certificate": errors,
                "error_rate": errors / len(p), "error_rate_ci95": binomial_interval(errors, len(p)),
                "mean_cost_all_pairs": float(np.mean([r["cost_cells"] for r in p])),
                "mean_cost_90_percent": float(np.mean([r["resolution_costs"]["0.9"] if r["resolution_costs"]["0.9"] is not None else r["full_cost_cells"] for r in p])),
                "complete_runs": sum(r["complete"] for r in p),
                "mean_certified_fraction_auc": float(np.mean([r["certified_fraction_auc"] for r in p])),
                "resolved_at_half_budget": float(np.mean([r["snapshots"]["0.5"]["resolved_fraction"] for r in p])),
                "forced_accuracy_at_half_budget": float(np.mean([r["snapshots"]["0.5"]["forced_pair_accuracy"] for r in p])),
                "mean_betting_certificates": float(np.mean([r["reasons_count"]["mixture_betting"] for r in p]))}
        comparisons = {}
        ours = sorted([r for r in rows if r["policy"] == "shared_racing"], key=lambda x: x["replicate"])
        for baseline in ("random_pair", "common_all", "common_active"):
            base = sorted([r for r in rows if r["policy"] == baseline], key=lambda x: x["replicate"])
            x = np.array([r["resolution_costs"]["0.9"] if r["resolution_costs"]["0.9"] is not None else r["full_cost_cells"] for r in ours])
            z = np.array([r["resolution_costs"]["0.9"] if r["resolution_costs"]["0.9"] is not None else r["full_cost_cells"] for r in base])
            rng = np.random.default_rng(20261113)
            indices = rng.integers(0, len(x), (5000, len(x)))
            relative = 1 - x[indices].mean(axis=1) / z[indices].mean(axis=1)
            comparisons[baseline] = {"cost90_saving_fraction": float(1 - x.mean() / z.mean()),
                                     "paired_bootstrap_ci95": np.quantile(relative, [.025, .975]).tolist(),
                                     "note": "MC variability across matched runs, NOT evidence across independent language populations"}
        results[scenario] = {"policies": by_policy, "shared_racing_vs": comparisons}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-reps", type=int, default=100)
    parser.add_argument("--real-reps", type=int, default=20)
    parser.add_argument("--real-csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.synthetic_reps < 0 or args.real_reps < 0:
        parser.error("replicate counts must be nonnegative")
    if args.out.exists():
        parser.error("choose a new output directory; existing results are never overwritten")
    args.out.mkdir(parents=True)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}
    manifest = {"protocol": protocol, "protocol_sha256": hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest(),
                "source_hashes_before_run": hashes,
                "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "scope": "development and synthetic validation only; no new human labels"}
    if args.real_csv is not None:
        real_y, metadata = load_development_csv(args.real_csv)
        manifest["real_input"] = metadata
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    start = time.monotonic()
    with (args.out / "trials.jsonl").open("w") as stream:
        for scenario_id, scenario in enumerate(SCENARIOS):
            for rep in range(args.synthetic_reps):
                y = synthetic_matrix(scenario, rep)
                seed = 20261113 + scenario_id * 10000 + rep
                for policy in POLICIES:
                    result = evaluate_one(y, scenario, rep, seed, policy)
                    records.append(result)
                    stream.write(json.dumps(result) + "\n")
            print(json.dumps({"finished": scenario, "replicates": args.synthetic_reps, "elapsed_seconds": round(time.monotonic() - start, 2)}), flush=True)
        if args.real_csv is not None:
            for rep in range(args.real_reps):
                for policy in POLICIES:
                    result = evaluate_one(real_y, "cs_de_development", rep, 2026111300 + rep, policy,
                                          ratings_per_cell=metadata["human_ratings_per_cell"])
                    records.append(result)
                    stream.write(json.dumps(result) + "\n")
                print(json.dumps({"finished": "cs_de_development", "replicate": rep + 1, "elapsed_seconds": round(time.monotonic() - start, 2)}), flush=True)
    summary = summarize(records)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"trials": len(records), "elapsed_seconds": round(time.monotonic() - start, 2), "output": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
