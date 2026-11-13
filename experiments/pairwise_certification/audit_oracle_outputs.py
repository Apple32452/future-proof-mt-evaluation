"""Read-only audit of the five completed hindsight runs; never invokes MILP.

Only the optional audit JSON is written. Frozen sources, protocol, optimization
results, and original v1/v2 data are not modified or re-optimized.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from certificate_racing_v2 import CertificateRacingState, run_certificate_race
from loader import load_development_csv
from oracle_costs import build_pair_geometry, resolution_count
from pairwise_racing import RaceState, UNKNOWN, run_race


ROOT = Path(__file__).resolve().parent
FRACTIONS = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95)
POLICIES = ("certificate_racing_v2", "shared_racing", "random_pair", "common_active", "common_all")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def scalar_tau(differences, pairs, alpha):
    """Independent scalar arithmetic; no oracle or inference helper calls."""
    n_items = len(differences)
    sums = 0.0
    capitals = [[0.0] * len(FRACTIONS) for _ in range(2)]
    peaks = [0.0, 0.0]
    first = n_items + 1
    threshold = math.log(2 * pairs / alpha) + 1e-10
    for n, value in enumerate(differences):
        d = float(value)
        for direction, x in enumerate(((1 + d) / 2, (1 - d) / 2)):
            previous_x = (n + (sums if direction == 0 else -sums)) / 2
            q = (n_items / 2 - previous_x) / (n_items - n)
            ratio = x / q if q > 1e-12 else 1.0
            capitals[direction] = [log_e + math.log(1 + fraction * (ratio - 1))
                                   for fraction, log_e in zip(FRACTIONS, capitals[direction])]
            peak = max(capitals[direction])
            log_mix = peak + math.log(math.fsum(math.exp(v - peak) for v in capitals[direction]) / len(FRACTIONS))
            peaks[direction] = max(peaks[direction], log_mix)
        if first == n_items + 1 and ((peaks[0] > threshold) != (peaks[1] > threshold)):
            first = n + 1
        sums += d
    return first


def direct_predicate(ordered, counts, tau):
    """Evaluate terminal-count definition independently, with accurate sums."""
    n, k = ordered.shape
    counts = list(map(int, counts))
    sums = [math.fsum(map(float, ordered[:counts[s], s])) for s in range(k)]
    answer = []
    for p, (a, b) in enumerate(itertools.combinations(range(k), 2)):
        lo = sums[a] / n - (sums[b] + n - counts[b]) / n
        hi = (sums[a] + n - counts[a]) / n - sums[b] / n
        answer.append(min(counts[a], counts[b]) >= tau[p] or lo > 1e-12 or hi < -1e-12
                      or (counts[a] == n and counts[b] == n))
    return np.asarray(answer, dtype=bool)


def replay(ordered, counts, target, alpha, reverse_systems=False):
    """Use original inference directly, not oracle.replay_allocation."""
    n, k = ordered.shape
    state = RaceState(n, k, alpha)
    first_cost, first_counts = None, None
    system_order = range(k - 1, -1, -1) if reverse_systems else range(k)
    for row in range(max(counts)):
        for s in system_order:
            if row < counts[s]:
                state.observe_next(s, float(ordered[row, s]))
                if first_cost is None and np.count_nonzero(state.decisions != UNKNOWN) >= target:
                    first_cost, first_counts = state.cost, state.counts.tolist()
    return {"charged_cells": state.cost, "first_target_cost": first_cost,
            "first_target_counts": first_counts, "resolved_pairs": int(np.count_nonzero(state.decisions != UNKNOWN)),
            "decisions": state.decisions.tolist(), "counts": state.counts.tolist()}, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "oracle_v1")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        parser.error("audit output already exists; choose a new path")
    completion = read_json(args.run_dir / "completion.json")
    if completion.get("status") != "COMPLETE":
        parser.error("requires a completed oracle run")
    manifest = read_json(args.run_dir / "manifest.json")
    protocol = read_json(ROOT / "oracle_protocol.json")
    summary = read_json(args.run_dir / "summary.json")
    records = read_lines(args.run_dir / "runs.jsonl")
    issues, checks = [], []

    def check(condition, name):
        checks.append(name)
        if not bool(condition):
            issues.append(name)

    def close(actual, expected, name):
        check(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), name)

    check(manifest["protocol"] == protocol, "oracle protocol manifest matches frozen file")
    check(sha(args.run_dir / "runs.jsonl") == completion["runs_sha256"], "completion run checksum")
    check(completion["runs"] == len(records) == summary["runs"] == 5, "exactly five completed records")
    check([(r["replicate"], r["seed"]) for r in records] == list(zip(protocol["replicates"], protocol["shared_item_order_seeds"])),
          "prespecified ordered replicate and seed list")
    tracked = {ROOT / name: digest for name, digest in manifest["source_hashes"].items()}
    for version in ("validation_v1", "validation_v2"):
        old_manifest = read_json(ROOT / version / "run_manifest.json")
        for name, digest in old_manifest["source_hashes_before_run"].items():
            check(sha(ROOT / name) == digest, f"{version} frozen source {name}")
            tracked[ROOT / name] = digest
    for path, digest in tracked.items():
        check(sha(path) == digest, f"oracle tracked source {path.name}")
    protected = {path: sha(path) for path in list(tracked) + list(args.run_dir.glob("*.json*"))}
    comparison_path = ROOT / "validation_v2/comparison_trials.jsonl"
    source_new = ROOT / "validation_v2/new_trials.jsonl"
    source_old = ROOT / "validation_v1/trials.jsonl"
    previous = read_json(ROOT / "validation_v2/run_manifest.json")
    check(sha(source_old) == previous["baseline_trials_sha256"], "original v1 trials checksum")
    check(sha(ROOT / "validation_v1/run_manifest.json") == previous["baseline_manifest_sha256"], "original v1 manifest checksum")
    check(sha(source_new) == read_json(ROOT / "validation_v2/completion.json")["new_trials_sha256"], "original v2 new trials checksum")
    protected.update({p: sha(p) for p in (comparison_path, source_new, source_old)})
    comparison = {(r["replicate"], r["policy"]): r for r in read_lines(comparison_path)
                  if r["scenario"] == "cs_de_development"}
    originals = {(r["replicate"], r["policy"]): r for p in (source_old, source_new)
                 for r in read_lines(p) if r["scenario"] == "cs_de_development"}
    y, metadata = load_development_csv(ROOT / "inputs/cs-de_DE_human_scores.csv")
    check(metadata == manifest["input"], "input checksum and metadata exact match")
    n, k = y.shape
    pairs = list(itertools.combinations(range(k), 2))
    target, alpha, warmup = math.ceil(0.9 * len(pairs)), protocol["alpha"], protocol["warmup_minimum"]
    check(target == manifest["target_pairs"], "target is ceil 90 percent of all pairs")
    exact_totals = [sum((Fraction.from_float(float(v)) for v in y[:, s]), Fraction()) for s in range(k)]
    truth = np.array([(exact_totals[a] > exact_totals[b]) - (exact_totals[a] < exact_totals[b]) for a, b in pairs])
    for method in ("observe_next", "_certify", "completion_bounds", "_census_sign"):
        check(getattr(CertificateRacingState, method) is getattr(RaceState, method), f"v2 unchanged inference method {method}")
    details = []
    for record in records:
        rep, seed, solver = record["replicate"], record["seed"], record["solver"]
        prefix = f"replicate {rep}: "
        for policy in POLICIES:
            old, merged = originals[(rep, policy)], comparison[(rep, policy)]
            check(old["seed"] == seed == merged["seed"], prefix + policy + " paired seed")
            for field in ("resolution_costs", "true_signs", "decisions", "decision_costs", "cost_cells"):
                check(old[field] == merged[field], prefix + policy + " source " + field)
            check(record["baseline_cost90"][policy] == old["resolution_costs"]["0.9"], prefix + policy + " paired baseline cost")
            check(old["true_signs"] == truth.tolist(), prefix + policy + " full-data exact truth")
        check(record["target_pairs"] == solver["target_pairs"] == target, prefix + "target")
        check(solver["warmup_min"] == warmup, prefix + "warmup restriction")
        check(solver["time_limit_seconds"] == protocol["time_limit_per_seed_seconds"], prefix + "time limit")
        check(record["items"] == n and record["systems"] == k, prefix + "dimensions")
        order = np.random.default_rng(np.random.SeedSequence([seed, 1])).permutation(n)
        ordered = y[order]
        geometry = build_pair_geometry(ordered, alpha)
        tau = np.array([scalar_tau(ordered[:, a] - ordered[:, b], len(pairs), alpha) for a, b in pairs])
        check(np.array_equal(tau, geometry["tau"]), prefix + "all pair tau values independently recomputed")
        warm_count = int(direct_predicate(ordered, [warmup] * k, tau).sum())
        check(warm_count == record["warmup_resolved_pairs"] and warm_count < target, prefix + "warmup guard")
        check(solver["variables"] == k * n + len(pairs), prefix + "MILP variable count")
        check(solver["constraints"] == k * (n - 1) + sum(map(len, geometry["forbidden_frontiers"])) + 1,
              prefix + "MILP constraint count")
        counts = solver["counts"]
        check(counts is not None, prefix + "incumbent exists")
        if counts is None:
            continue
        check(len(counts) == k and all(isinstance(c, int) and warmup <= c <= n for c in counts), prefix + "incumbent integral count bounds")
        predicate = direct_predicate(ordered, counts, tau)
        check(predicate.sum() == resolution_count(geometry, counts) == solver["resolved_pairs"], prefix + "independent incumbent geometry")
        check(predicate.sum() >= target, prefix + "incumbent reaches target")
        check(sum(counts) == solver["incumbent_cost"], prefix + "incumbent charged-cell sum")
        expected_lower = max(k * warmup, math.floor(max(0, solver["raw_solver_dual_bound"]) - 1e-7))
        check(solver["lower_bound"] == expected_lower == record["oracle_cost_lower_bound"], prefix + "conservative dual rounding")
        check(solver["optimal"] == (solver["solver_status"] == 0), prefix + "numerical optimal flag")
        if solver["optimal"]:
            close(solver["gap"], 0, prefix + "solver reported zero relative gap")
            close(solver["raw_solver_dual_bound"], solver["incumbent_cost"], prefix + "numerical primal dual agreement")
        replayed, state = replay(ordered, counts, target, alpha)
        for field, value in replayed.items():
            check(value == record["incumbent_replay"][field], prefix + "independent actual inference replay " + field)
        check(np.array_equal(state.decisions != UNKNOWN, predicate), prefix + "actual inference predicate agrees pairwise")
        check(set(state.reasons).issubset({"", "deterministic_completion", "mixture_betting", "exact_census"}), prefix + "no transitive certificate reasons")
        wrong = int(((state.decisions != UNKNOWN) & (state.decisions != truth)).sum())
        check(wrong == record["incumbent_replay"]["wrong_issued_certificates"], prefix + "wrong certificate recount")
        reverse, reverse_state = replay(ordered, counts, target, alpha, reverse_systems=True)
        check(np.array_equal(reverse_state.decisions != UNKNOWN, predicate), prefix + "reverse-system schedule agrees pairwise")
        check(reverse["first_target_cost"] is not None, prefix + "second schedule reaches target")
        # A query guard exposes exactly one next-prefix cell per acquisition;
        # the v2 controller is not passed the full matrix or full-data truth.
        queried_counts = np.zeros(k, dtype=int)
        queries = []

        def query(i, s):
            check(i == int(order[queried_counts[s]]), prefix + f"query {len(queries)} next-prefix only")
            queries.append((i, s))
            queried_counts[s] += 1
            return float(y[i, s])

        baseline = record["baseline_cost90"]["certificate_racing_v2"]
        actual_v2 = run_certificate_race(n, k, query, seed=seed, alpha=alpha, budget=baseline, warmup_rows=warmup)
        check(len(queries) == len(set(queries)) == actual_v2["cost_cells"] == baseline, prefix + "exactly one query per charged cell")
        check(actual_v2["system_counts"] == record["known_v2_counts_at_target"], prefix + "paired v2 allocation reproduced")
        check(actual_v2["resolution_costs"]["0.9"] == baseline, prefix + "paired v2 first target reproduced")
        check(direct_predicate(ordered, actual_v2["system_counts"], tau).sum() >= target, prefix + "known v2 geometry feasible")
        expected_upper = min(baseline, replayed["first_target_cost"])
        check(record["oracle_cost_upper_bound"] == expected_upper, prefix + "feasible upper bound")
        check(record["oracle_cost_lower_bound"] <= expected_upper <= baseline, prefix + "lower <= upper <= paired v2")
        check(all(c >= warmup for c in replayed["first_target_counts"]), prefix + "first-target upper preserves warmup")
        for policy, cost in record["baseline_cost90"].items():
            expected = [1 - expected_upper / cost, 1 - expected_lower / cost]
            for got, want in zip(record["possible_saving_fraction_intervals"][policy], expected):
                close(got, want, prefix + policy + " savings interval")
        details.append({"replicate": rep, "seed": seed, "solver_status": solver["solver_status"],
                        "solver_reported_optimal": solver["optimal"], "solver_reported_relative_gap": solver["gap"],
                        "raw_numerical_dual": solver["raw_solver_dual_bound"], "conservative_lower": expected_lower,
                        "verified_upper": expected_upper, "numerical_bound_width_cells": expected_upper - expected_lower,
                        "v2_cost90": baseline, "incumbent_resolved_pairs": int(predicate.sum()),
                        "wrong_incumbent_certificates": wrong, "independent_scalar_tau_checked_pairs": len(pairs),
                        "incumbent_schedules_replayed": 2, "v2_queries_checked": len(queries),
                        "warmup_resolved_pairs": warm_count})
    mean_lower = math.fsum(r["oracle_cost_lower_bound"] for r in records) / len(records)
    mean_upper = math.fsum(r["oracle_cost_upper_bound"] for r in records) / len(records)
    close(summary["mean_oracle_lower_bound"], mean_lower, "aggregate lower bound")
    close(summary["mean_oracle_upper_bound"], mean_upper, "aggregate upper bound")
    saving_ranges = {}
    for policy in POLICIES:
        mean_cost = math.fsum(r["baseline_cost90"][policy] for r in records) / len(records)
        close(summary["mean_baseline_cost90"][policy], mean_cost, policy + " aggregate baseline")
        saving_ranges[policy] = [1 - mean_upper / mean_cost, 1 - mean_lower / mean_cost]
        for got, want in zip(summary["possible_saving_fraction_intervals"][policy], saving_ranges[policy]):
            close(got, want, policy + " ratio-of-means saving interval")
    check(summary["solver_optimal_runs"] == sum(r["solver"]["optimal"] for r in records), "solver status aggregate")
    check(summary["wrong_issued_certificates_in_replayed_incumbents"] == sum(r["wrong_incumbent_certificates"] for r in details), "wrong certificate aggregate")
    for path, digest in protected.items():
        check(sha(path) == digest, "audit left source/output unchanged: " + str(path.relative_to(ROOT)))
    result = {
        "status": "PASS" if not issues else "FAIL", "checks": len(checks), "issues": issues,
        "optimization_runs_executed_by_audit": 0, "input_sha256": metadata["sha256"],
        "oracle_runs_sha256": completion["runs_sha256"], "auditor_source_sha256": sha(Path(__file__)),
        "records": details, "mean_oracle_lower": mean_lower, "mean_verified_upper": mean_upper,
        "ratio_of_means_saving_intervals": saving_ranges,
        "inference_review": {
            "v2_inference_methods_identical_to_v1": True,
            "certificate_reasons": ["deterministic_completion", "mixture_betting", "exact_census"],
            "deployable_score_access": "dimension-only controller plus one guarded callback call per unique next-prefix acquired cell; full labels and true signs are evaluation-side only",
            "reuse": "one score cell is acquired once and reused only in its observed system/pair prefixes",
            "forecast": "v2 fictional continuation acts on copied observed betting state; no forecast is passed to observe_next",
            "static_review_scope": "reviewed frozen source; observed query traces are checked, not a formal information-flow proof"
        },
        "limitations": [
            "These are five prespecified permutations of the same already-exposed development matrix, not five independent MT datasets or new human evaluations.",
            "Optimal means solver status 0 and numerical primal/dual agreement; this audit does not independently verify an exact-rational branch-and-bound proof.",
            "Lower bound is conservatively floor(raw dual minus 1e-7), so each reported interval remains one cell wide even when the numerical solver reports zero gap.",
            "The optimized class permits arbitrary per-system prefix acquisitions but RETAINS at least four acquired cells per system. It relaxes pair-bundle scheduling, not the warmup restriction.",
            "The zero-certificate all-counts-four check justifies applying the bound to the frozen policies after mandatory warmup; it does not justify applying this lower bound to arbitrary no-warmup algorithms.",
            "The feasible hindsight allocation need not be realizable by the narrower online pair-bundle action class and is not deployable without unobserved labels.",
            "The savings intervals bound the hindsight optimum relative to paired baselines on these five orders under this fixed inference protocol; they are not confidence intervals or attainable online improvements.",
            "No transitive closure or proxy judgments enter actual certificates. Full labels are intentionally available only to the hindsight diagnostic and outer evaluation."
        ]
    }
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
