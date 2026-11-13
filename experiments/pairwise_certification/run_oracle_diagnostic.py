"""Five fixed, exposed-development hindsight cost-bound problems."""
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
import scipy

from loader import load_development_csv
from certificate_racing_v2 import run_certificate_race
from oracle_costs import build_pair_geometry, resolution_count, solve_oracle, replay_allocation
from pairwise_racing import UNKNOWN

ROOT = Path(__file__).resolve().parent


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new output directory")
    protocol = json.loads((ROOT / "oracle_protocol.json").read_text())
    y, metadata = load_development_csv(ROOT / "inputs/cs-de_DE_human_scores.csv")
    original = [json.loads(line) for line in (ROOT / "validation_v2/comparison_trials.jsonl").read_text().splitlines()]
    original = {(r["replicate"], r["policy"]): r for r in original if r["scenario"] == "cs_de_development"}
    # Do not silently change a frozen source or reinterpret its stored results.
    for version in ("validation_v1", "validation_v2"):
        manifest = json.loads((ROOT / version / "run_manifest.json").read_text())
        for name, digest in manifest["source_hashes_before_run"].items():
            if checksum(ROOT / name) != digest:
                raise AssertionError(f"frozen source changed: {name}")
    n, k = y.shape
    p = k * (k - 1) // 2
    target = math.ceil(.9 * p)
    source_hashes = {name: checksum(ROOT / name) for name in (
        "oracle_costs.py", "run_oracle_diagnostic.py", "oracle_protocol.json",
        "test_oracle_costs.py", "pairwise_racing.py", "certificate_racing_v2.py", "loader.py")}
    args.out.mkdir(parents=True)
    (args.out / "manifest.json").write_text(json.dumps({
        "protocol": protocol, "source_hashes": source_hashes,
        "input": metadata, "python": sys.version, "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(),
        "target_pairs": target, "scope": "five prespecified exposed development permutations only"}, indent=2) + "\n")
    results = []
    with (args.out / "runs.jsonl").open("w") as output:
        for rep, seed in zip(protocol["replicates"], protocol["shared_item_order_seeds"]):
            started = time.monotonic()
            order = np.random.default_rng(np.random.SeedSequence([seed, 1])).permutation(n)
            ordered = y[order]
            geometry = build_pair_geometry(ordered)
            warm = resolution_count(geometry, np.full(k, 4))
            if warm >= target:
                raise AssertionError("prespecified warmup guard failed; cannot use this restricted bound")
            v2_record = original[(rep, "certificate_racing_v2")]
            actual_cost = v2_record["resolution_costs"]["0.9"]
            actual = run_certificate_race(n, k, lambda i, s: y[i, s], seed=seed, budget=actual_cost)
            if actual["resolved_pairs"] < target or resolution_count(geometry, actual["system_counts"]) < target:
                raise AssertionError("geometry does not recognize the observed v2 allocation")
            solved = solve_oracle(geometry, target, warmup_min=4,
                                  time_limit=protocol["time_limit_per_seed_seconds"])
            if solved["lower_bound"] > actual_cost:
                raise AssertionError("reported bound excludes a known feasible allocation")
            replay = None
            if solved["counts"] is not None:
                replay = replay_allocation(ordered, solved["counts"], target_pairs=target)
                if replay["charged_cells"] != solved["incumbent_cost"] or replay["resolved_pairs"] != solved["resolved_pairs"]:
                    raise AssertionError("actual inference disagrees with oracle count model")
                if replay["first_target_cost"] is None:
                    raise AssertionError("incumbent does not reach target")
                truth = np.asarray(v2_record["true_signs"])
                decisions = np.asarray(replay["decisions"])
                replay["wrong_issued_certificates"] = int(((decisions != UNKNOWN) & (decisions != truth)).sum())
                if solved["optimal"] and replay["first_target_cost"] != solved["incumbent_cost"]:
                    raise AssertionError("earlier feasible prefix contradicts claimed optimum")
            upper = min(actual_cost, replay["first_target_cost"] if replay else actual_cost)
            lower = solved["lower_bound"]
            baseline_costs = {policy: original[(rep, policy)]["resolution_costs"]["0.9"]
                              for policy in ("certificate_racing_v2", "shared_racing", "random_pair", "common_active", "common_all")}
            record = {"replicate": rep, "seed": seed, "items": n, "systems": k,
                      "target_pairs": target, "warmup_resolved_pairs": warm,
                      "solver": solved, "incumbent_replay": replay,
                      "known_v2_counts_at_target": actual["system_counts"],
                      "oracle_cost_lower_bound": lower, "oracle_cost_upper_bound": upper,
                      "baseline_cost90": baseline_costs,
                      "possible_saving_fraction_intervals": {
                          name: [1 - upper / cost, 1 - lower / cost]
                          for name, cost in baseline_costs.items()},
                      "elapsed_seconds": round(time.monotonic() - started, 3)}
            results.append(record)
            output.write(json.dumps(record) + "\n")
            output.flush()
            print(json.dumps({"replicate": rep, "status": solved["solver_status"],
                              "lower": lower, "upper": upper, "v2": actual_cost,
                              "seconds": record["elapsed_seconds"]}), flush=True)
    for name, digest in source_hashes.items():
        if checksum(ROOT / name) != digest:
            raise AssertionError(f"source changed during optimization: {name}")
    mean_lower = float(np.mean([r["oracle_cost_lower_bound"] for r in results]))
    mean_upper = float(np.mean([r["oracle_cost_upper_bound"] for r in results]))
    means = {policy: float(np.mean([r["baseline_cost90"][policy] for r in results]))
             for policy in results[0]["baseline_cost90"]}
    summary = {"runs": len(results), "solver_optimal_runs": sum(r["solver"]["optimal"] for r in results),
               "mean_oracle_lower_bound": mean_lower, "mean_oracle_upper_bound": mean_upper,
               "mean_baseline_cost90": means,
               "possible_saving_fraction_intervals": {p: [1 - mean_upper / c, 1 - mean_lower / c] for p, c in means.items()},
               "scope": "hindsight diagnostic under fixed inference and five exposed development item orders; not independent MT generalization",
               "bound_kind": "numerical MILP dual bounds conservatively rounded down, not formal exact-rational certificates",
               "action_class": "arbitrary per-system prefix acquisitions; incumbents are not guaranteed feasible for the narrower pair-bundle policy class",
               "wrong_issued_certificates_in_replayed_incumbents": sum((r["incumbent_replay"] or {}).get("wrong_issued_certificates", 0) for r in results)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "completion.json").write_text(json.dumps({"status": "COMPLETE", "runs": len(results),
        "runs_sha256": checksum(args.out / "runs.jsonl")}, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
