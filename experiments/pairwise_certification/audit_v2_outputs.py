"""Independent consistency audit of the frozen allocation-only v2 experiment.

Requires completion.json before reading performance outputs. Prints a JSON
report; never edits algorithms, protocols, inputs, or validation outputs.
Truth and metrics are recomputed independently of validate_v2.py. Two traced
real replays also feed only acquired values to the unmodified v1 RaceState,
checking that hypothetical forecast values did not become evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta

from audit_outputs import exact_signs, reconstruct_synthetic
from certificate_racing_v2 import run_certificate_race
from pairwise_racing import RaceState


ROOT = Path(__file__).resolve().parent
SCENARIOS = ("well_separated", "one_near_tie", "all_tied", "heterogeneous_noise",
             "strong_item_effect", "weak_separation")
OLD_POLICIES = ("shared_racing", "random_pair", "common_all", "common_active")
NEW_POLICY = "certificate_racing_v2"
POLICIES = (NEW_POLICY,) + OLD_POLICIES
UNKNOWN = 2
TOP_FIELDS = {"top_two_pair", "top_two_true_sign", "top_two_decision_cost",
              "top_two_decision_correct", "top_two_resolved_at_half_budget"}
RECORD_FIELDS = {
    "scenario", "replicate", "seed", "policy", "items", "systems", "pairs",
    "full_cost_cells", "cost_cells", "human_rating_equivalents",
    "any_wrong_certificate", "wrong_certificates", "complete", "resolved_fraction",
    "certified_fraction_auc", "resolution_costs", "decisions", "decision_costs",
    "true_signs", "reasons_count", "snapshots",
} | TOP_FIELDS
PINNED_REAL_SHA = "f6de1e8da36fb67c7d99ec354e8807a016045c2d5b420f3ab8aeb72816bc279d"
PINNED_BASELINE_SHA = "cea91c6ddfafa1edf6b640973a18847c2321e0939cfd1331b8041d57ba146873"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(record):
    return record["scenario"], record["replicate"], record["policy"]


def exact_top_two(matrix):
    sums = [sum((Fraction.from_float(float(v)) for v in matrix[:, s]), Fraction())
            for s in range(matrix.shape[1])]
    # Equal row counts make the ordering of exact column sums the mean ordering.
    best = sorted(range(len(sums)), key=lambda s: (-sums[s], s))[:2]
    return sorted(best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "validation_v2")
    parser.add_argument("--baseline", type=Path, default=ROOT / "validation_v1")
    parser.add_argument("--real-csv", type=Path, default=ROOT / "inputs/cs-de_DE_human_scores.csv")
    args = parser.parse_args()
    if not (args.results / "completion.json").is_file():
        raise RuntimeError("V2 is not complete: completion.json is absent; no performance audit performed")
    completion = json.loads((args.results / "completion.json").read_text())
    if completion.get("status") != "COMPLETE":
        raise RuntimeError("completion.json does not declare COMPLETE")

    issues, checked = [], Counter()

    def require(condition, message):
        checked["assertions"] += 1
        if not bool(condition):
            issues.append(message)

    def close(actual, expected, message, tol=1e-12):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, abs_tol=tol, rel_tol=tol), message)

    manifest = json.loads((args.results / "run_manifest.json").read_text())
    old_manifest = json.loads((args.baseline / "run_manifest.json").read_text())
    require(manifest["new_runs_planned"] == completion["new_runs"] == 620, "New-run count is not 620")
    require(manifest["reference_runs_reused"] == completion["reference_runs"] == 2480,
            "Reference-run count is not 2480")
    protected = {ROOT / name for name in manifest["source_hashes_before_run"]}
    protected |= {ROOT / name for name in old_manifest["source_hashes_before_run"]}
    protected |= {ROOT / "protocol.json", ROOT / "protocol_v2.json", ROOT / "output_audit.json"}
    protected |= {args.results / name for name in ("run_manifest.json", "new_trials.jsonl",
                    "comparison_trials.jsonl", "summary.json", "completion.json")}
    protected |= {args.baseline / name for name in ("run_manifest.json", "trials.jsonl", "summary.json")}
    protected.add(args.real_csv)
    before = {str(p): sha256(p) for p in protected}
    for label, source in (("v2", manifest), ("v1", old_manifest)):
        for name, expected in source["source_hashes_before_run"].items():
            require(sha256(ROOT / name) == expected, f"{label} frozen source changed: {name}")
        protocol_name = "protocol_v2.json" if label == "v2" else "protocol.json"
        require(sha256(ROOT / protocol_name) == source["protocol_sha256"], f"{label} protocol hash changed")
        require(json.loads((ROOT / protocol_name).read_text()) == source["protocol"],
                f"{label} embedded protocol differs from file")
    baseline_sha = sha256(args.baseline / "trials.jsonl")
    require(baseline_sha == manifest["baseline_trials_sha256"] == PINNED_BASELINE_SHA,
            "Baseline trial checksum differs from the independently audited v1 set")
    require(sha256(args.baseline / "run_manifest.json") == manifest["baseline_manifest_sha256"],
            "Baseline manifest checksum changed")
    require(sha256(args.results / "new_trials.jsonl") == completion["new_trials_sha256"],
            "Completion checksum does not match new trial records")

    originals = [json.loads(line) for line in (args.baseline / "trials.jsonl").read_text().splitlines()]
    new_rows = [json.loads(line) for line in (args.results / "new_trials.jsonl").read_text().splitlines()]
    rows = [json.loads(line) for line in (args.results / "comparison_trials.jsonl").read_text().splitlines()]
    old_lookup = {identity(r): r for r in originals}
    new_lookup = {identity(r): r for r in new_rows}
    lookup = {identity(r): r for r in rows}
    require(len(originals) == len(old_lookup) == 2480, "Original baseline records are missing or duplicated")
    require(len(new_rows) == len(new_lookup) == 620, "New records are missing or duplicated")
    require(len(rows) == len(lookup) == 3100, "Combined records are missing or duplicated")
    expected = {(s, rep, p) for s in SCENARIOS for rep in range(100) for p in POLICIES}
    expected |= {("cs_de_development", rep, p) for rep in range(20) for p in POLICIES}
    require(set(lookup) == expected, "Unexpected or missing combined trial identities")
    require(set(new_lookup) == {key for key in expected if key[2] == NEW_POLICY},
            "Unexpected or missing new-policy identities")
    for record in rows:
        key = identity(record)
        require(set(record) == RECORD_FIELDS, f"{key}: unexpected record schema")
        if key[2] == NEW_POLICY:
            require(record == new_lookup.get(key), f"{key}: combined record differs from new-only record")
            checked["new_records_matched"] += 1
        else:
            original_fields = {k: v for k, v in record.items() if k not in TOP_FIELDS}
            require(original_fields == old_lookup.get(key), f"{key}: original baseline fields changed")
            checked["baseline_records_matched"] += 1

    # Read the historical matrix independently of the loader and validator.
    real_sha = sha256(args.real_csv)
    require(real_sha == PINNED_REAL_SHA == manifest["real_input"]["sha256"], "Historical input hash mismatch")
    with args.real_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        systems = [name for name in reader.fieldnames if name not in ("sentence_id", "source_text")]
        raw_records = list(reader)
    real = np.array([[float(r[s]) / 100.0 for s in systems] for r in raw_records])
    require(real.shape == (231, 20), "Real matrix shape differs from 231 by 20")
    require(np.isfinite(real).all() and (real >= 0).all() and (real <= 1).all(), "Invalid bounded real matrix")
    require(systems == manifest["real_input"]["systems"], "Real system order differs from manifest")
    require(len({r["sentence_id"] for r in raw_records}) == 231, "Real item IDs are not unique")
    require(len({r["sentence_id"].rsplit("_#_", 1)[0] for r in raw_records}) == 128, "Real document count mismatch")
    require(manifest["real_input"]["human_ratings_per_cell"] == 2, "Incorrect historical ratings per mean cell")
    real_truth, real_top = exact_signs(real), exact_top_two(real)
    truths, tops = {}, {}
    for scenario in SCENARIOS:
        for rep in range(100):
            matrix = reconstruct_synthetic(scenario, rep)
            truths[scenario, rep] = exact_signs(matrix)
            tops[scenario, rep] = exact_top_two(matrix)
    for rep in range(20):
        truths["cs_de_development", rep] = real_truth
        tops["cs_de_development", rep] = real_top

    grouped, wrong_runs, wrong_edges = defaultdict(list), Counter(), Counter()
    for record in rows:
        scenario, rep, policy = identity(record)
        label = f"{scenario}/{rep}/{policy}"
        grouped[scenario, policy].append(record)
        n, k = (231, 20) if scenario == "cs_de_development" else (120, 6)
        total, pairs = n * k, k * (k - 1) // 2
        seed = 2026111300 + rep if scenario == "cs_de_development" else 20261113 + SCENARIOS.index(scenario) * 10000 + rep
        require((record["items"], record["systems"], record["pairs"], record["full_cost_cells"], record["seed"])
                == (n, k, pairs, total, seed), label + ": dimensions or paired seed mismatch")
        require(record["true_signs"] == truths[scenario, rep], label + ": incorrect exact finite-matrix truth")
        decision = np.array(record["decisions"])
        costs = np.array(record["decision_costs"])
        truth = np.array(truths[scenario, rep])
        require(decision.shape == costs.shape == truth.shape, label + ": malformed pair arrays")
        require(np.isin(decision, [-1, 0, 1, UNKNOWN]).all(), label + ": invalid decision code")
        known = decision != UNKNOWN
        wrong = known & (decision != truth)
        cost = record["cost_cells"]
        require(type(cost) is int and 0 <= cost <= total, label + ": acquisition cost outside census cap")
        require(bool(known.all()) == record["complete"], label + ": inconsistent completion flag")
        require(record["complete"] is True, label + ": full-budget run remained incomplete")
        close(record["resolved_fraction"], float(known.mean()), label + ": resolved fraction mismatch")
        require(record["any_wrong_certificate"] == bool(wrong.any()), label + ": wrong-certificate event mismatch")
        require(record["wrong_certificates"] == int(wrong.sum()), label + ": wrong-certificate count mismatch")
        require((costs[known] >= 1).all() and (costs[known] <= cost).all(), label + ": invalid decision cost")
        require((costs[~known] == -1).all(), label + ": unresolved pair has a decision cost")
        require(sum(record["reasons_count"].values()) == int(known.sum()), label + ": reason counts mismatch")
        if known.all():
            require(int(costs.max()) == cost, label + ": observations continued after full resolution")
        close(record["certified_fraction_auc"], float(np.sum(total - costs[known]) / (total * pairs)),
              label + ": independently integrated certificate AUC mismatch")
        ordered_costs = sorted(costs[known].tolist())
        for fraction in (.5, .9, .95, 1.0):
            threshold = math.ceil(fraction * pairs)
            expected_cost = ordered_costs[threshold - 1] if len(ordered_costs) >= threshold else None
            require(record["resolution_costs"][str(fraction)] == expected_cost,
                    label + f": {fraction} resolution cost mismatch")
            if expected_cost is not None:
                require(expected_cost <= cost <= total, label + ": resolution cost exceeds final/census cost")
        last_resolved = 0.0
        for fraction in (.1, .25, .5, .75):
            snap = record["snapshots"][str(fraction)]
            budget = int(total * fraction)
            actual = min(budget, cost)
            require((snap["budget_cells"], snap["actual_cost_cells"]) == (budget, actual),
                    label + ": checkpoint/early-stop cost is incorrect")
            prefix_known = known & (costs <= actual)
            prefix_correct = prefix_known & (decision == truth)
            close(snap["resolved_fraction"], float(prefix_known.mean()), label + ": checkpoint resolution mismatch")
            close(snap["correct_certified_fraction_all_pairs"], float(prefix_correct.mean()),
                  label + ": checkpoint correct fraction mismatch")
            require(snap["resolved_fraction"] >= last_resolved, label + ": nonmonotone certificate count")
            last_resolved = snap["resolved_fraction"]
            if prefix_known.any():
                close(snap["conditional_certified_accuracy"], float(prefix_correct.sum() / prefix_known.sum()),
                      label + ": checkpoint conditional accuracy mismatch")
            else:
                require(snap["conditional_certified_accuracy"] is None, label + ": empty certificate accuracy not null")
            require(0 <= snap["forced_pair_accuracy"] <= 1, label + ": invalid forced accuracy")
        require(record["human_rating_equivalents"] == (2 * cost if scenario == "cs_de_development" else None),
                label + ": historical rating-equivalent cost mismatch")
        pair_list = [(a, b) for a in range(k) for b in range(a + 1, k)]
        top = tops[scenario, rep]
        top_idx = pair_list.index(tuple(top))
        require(record["top_two_pair"] == top, label + ": wrong full-data top-two identity")
        require(record["top_two_true_sign"] == int(truth[top_idx]), label + ": wrong top-two truth")
        require(record["top_two_decision_cost"] == int(costs[top_idx]), label + ": wrong top-two resolution cost")
        require(record["top_two_decision_correct"] == bool(known[top_idx] and decision[top_idx] == truth[top_idx]),
                label + ": wrong top-two correctness")
        require(record["top_two_resolved_at_half_budget"] == bool(known[top_idx] and costs[top_idx] <= int(total * .5)),
                label + ": wrong top-two half-budget resolution")
        wrong_runs[scenario, policy] += bool(wrong.any())
        wrong_edges[scenario, policy] += int(wrong.sum())
        checked["trial_records"] += 1

    published = json.loads((args.results / "summary.json").read_text())
    require(set(published) == set(SCENARIOS) | {"cs_de_development"}, "Summary scenario set mismatch")
    ratios = {}
    for (scenario, policy), values in grouped.items():
        values.sort(key=lambda r: r["replicate"])
        out = published[scenario]["policies"][policy]
        number, errors = len(values), wrong_runs[scenario, policy]
        require(out["runs"] == number, f"{scenario}/{policy}: summary count mismatch")
        require(out["runs_with_any_wrong_certificate"] == errors, f"{scenario}/{policy}: summary errors mismatch")
        close(out["error_rate"], errors / number, f"{scenario}/{policy}: summary error frequency mismatch")
        lower = float(beta.ppf(.025, errors, number - errors + 1)) if errors else 0.0
        upper = float(beta.ppf(.975, errors + 1, number - errors)) if errors < number else 1.0
        for i, expected_bound in enumerate((lower, upper)):
            close(out["error_rate_ci95"][i], expected_bound, f"{scenario}/{policy}: binomial interval mismatch")
        aggregate = {
            "mean_cost_90_percent": np.mean([r["resolution_costs"]["0.9"] for r in values]),
            "mean_cost_all_pairs": np.mean([r["cost_cells"] for r in values]),
            "resolved_at_half_budget": np.mean([r["snapshots"]["0.5"]["resolved_fraction"] for r in values]),
            "forced_accuracy_at_half_budget": np.mean([r["snapshots"]["0.5"]["forced_pair_accuracy"] for r in values]),
            "mean_certified_fraction_auc": np.mean([r["certified_fraction_auc"] for r in values]),
            "mean_top_two_decision_cost": np.mean([r["top_two_decision_cost"] for r in values]),
            "top_two_resolved_at_half_budget": np.mean([r["top_two_resolved_at_half_budget"] for r in values]),
        }
        for key, expected_value in aggregate.items():
            close(out[key], float(expected_value), f"{scenario}/{policy}: summary {key} mismatch")
        require(out["complete_runs"] == sum(r["complete"] for r in values), f"{scenario}/{policy}: summary completion mismatch")
        require(out["top_two_wrong_decision_runs"] == sum(not r["top_two_decision_correct"] for r in values),
                f"{scenario}/{policy}: summary top-two wrong count mismatch")
        checked["aggregate_policy_groups"] += 1
    for scenario in published:
        candidate = grouped[scenario, NEW_POLICY]
        candidate_costs = np.array([r["resolution_costs"]["0.9"] for r in candidate])
        for base in OLD_POLICIES:
            reference = grouped[scenario, base]
            require([(r["replicate"], r["seed"]) for r in candidate]
                    == [(r["replicate"], r["seed"]) for r in reference], f"{scenario}/{base}: unpaired summary")
            baseline_costs = np.array([r["resolution_costs"]["0.9"] for r in reference])
            saving = float(1 - candidate_costs.mean() / baseline_costs.mean())
            out = published[scenario]["certificate_v2_vs"][base]
            close(out["cost90_saving_fraction"], saving, f"{scenario}/{base}: saving ratio mismatch")
            rng = np.random.default_rng(20261113)
            selections = rng.integers(len(candidate), size=(5000, len(candidate)))
            bootstrap = 1 - np.mean(candidate_costs[selections], axis=1) / np.mean(baseline_costs[selections], axis=1)
            bounds = np.percentile(bootstrap, [2.5, 97.5])
            for i, expected_bound in enumerate(bounds):
                close(out["paired_bootstrap_ci95"][i], float(expected_bound), f"{scenario}/{base}: bootstrap interval mismatch")
            ratios[f"{scenario}/{base}"] = saving
            checked["bootstrap_comparisons"] += 1

    # Re-run two fixed real seeds with explicit callbacks, then replay ONLY the
    # observed trace through v1 inference, independent of v2 forecasting.
    replay_checks = []
    pair_a, pair_b = np.triu_indices(20, 1)
    full_gaps = np.mean(real[:, pair_a] - real[:, pair_b], axis=0)
    for rep in (0, 1):
        record = lookup["cs_de_development", rep, NEW_POLICY]
        calls = []
        seen = set()
        def query(row, system):
            require((row, system) not in seen, f"replay/{rep}: duplicate query")
            require(0 <= row < 231 and 0 <= system < 20, f"replay/{rep}: invalid query coordinates")
            seen.add((row, system))
            calls.append((row, system))
            return real[row, system]
        checkpoints = [int(real.size * f) for f in (.1, .25, .5, .75)]
        result = run_certificate_race(231, 20, query, seed=record["seed"], checkpoints=checkpoints, record_trace=True)
        label = f"replay/{rep}/{NEW_POLICY}"
        require(len(calls) == len(seen) == result["cost_cells"] == len(result["trace"]), label + ": trace/cost mismatch")
        order = np.random.default_rng(np.random.SeedSequence([record["seed"], 1])).permutation(231)
        counts = np.zeros(20, dtype=int)
        evidence = RaceState(231, 20, .05)
        evidence_snapshots = {}
        for index, (event, (row, system)) in enumerate(zip(result["trace"], calls), 1):
            require(row == int(order[counts[system]]), label + ": nonprefix query")
            counts[system] += 1
            require(event == {"cost": index, "row": row, "system": system, "value": real[row, system]},
                    label + ": trace differs from callback")
            evidence.observe_next(system, float(real[row, system]))
            lo, hi = evidence.completion_bounds()
            require(np.all(lo <= full_gaps + 1e-12) and np.all(hi >= full_gaps - 1e-12),
                    label + ": deterministic completion bounds exclude finite truth")
            if index in checkpoints:
                evidence_snapshots[index] = evidence.summary()
        evidence_final = evidence.summary()
        require(counts.tolist() == result["system_counts"], label + ": final system counts mismatch")
        require(np.minimum(counts[pair_a], counts[pair_b]).tolist() == result["paired_prefix_counts"],
                label + ": final paired prefix counts mismatch")
        require(sum(result["system_counts"]) == len(calls), label + ": counts differ from cost")
        for key in ("cost_cells", "decisions", "decision_costs", "resolution_costs", "complete"):
            require(result[key] == record[key], label + ": frozen-output replay mismatch for " + key)
        for key in ("cost_cells", "decisions", "decision_costs", "reasons", "system_counts", "paired_prefix_counts", "sample_means"):
            require(result[key] == evidence_final[key], label + ": fictional forecast affected actual evidence for " + key)
        close(result["certified_fraction_auc"], record["certified_fraction_auc"], label + ": AUC replay mismatch")
        for fraction, budget in zip((.1, .25, .5, .75), checkpoints):
            snap = result["snapshots"][str(budget)]
            independent_snap = evidence_snapshots.get(budget, evidence_final)
            require(snap == independent_snap, label + ": checkpoint differs from acquired-only v1 inference")
            observed = record["snapshots"][str(fraction)]
            means = snap["sample_means"]
            predictions = [int(means[a] > means[b]) - int(means[a] < means[b])
                           for a in range(20) for b in range(a + 1, 20)]
            credit = [1.0 if pred == true else .5 if pred == 0 and true != 0 else 0.0
                      for pred, true in zip(predictions, real_truth)]
            close(observed["forced_pair_accuracy"], float(np.mean(credit)), label + ": forced-accuracy replay mismatch")
        replay_checks.append({"replicate": rep, "seed": record["seed"], "query_events": len(calls),
                              "frozen_output_replay": "matched",
                              "acquired_only_v1_inference_replay": "matched"})
        checked["replayed_query_events"] += len(calls)
    require(len(replay_checks) == 2, "Expected two traced real-data replays")
    after = {str(p): sha256(p) for p in protected}
    require(before == after, "A protected file changed during the output audit")

    report = {
        "status": "PASS" if not issues else "FAIL",
        "scope": "Independent v2 record/metric audit and two traced development replays; no untouched data or theorem replacement",
        "checked": dict(checked),
        "expected_new_records": 620, "expected_reused_baseline_records": 2480,
        "expected_combined_records": 3100,
        "protected_hashes_unchanged_during_audit": before == after,
        "new_trials_sha256": sha256(args.results / "new_trials.jsonl"),
        "comparison_trials_sha256": sha256(args.results / "comparison_trials.jsonl"),
        "baseline_trials_sha256": baseline_sha,
        "protocol_v2_sha256": sha256(ROOT / "protocol_v2.json"),
        "real_input_sha256": real_sha,
        "independent_real_truth_pairs": len(real_truth), "independent_synthetic_populations": 600,
        "wrong_certificate_runs": {f"{s}/{p}": wrong_runs[s, p] for s, p in sorted(grouped)},
        "wrong_certificates": {f"{s}/{p}": wrong_edges[s, p] for s, p in sorted(grouped)},
        "recomputed_cost90_savings": ratios,
        "replays": replay_checks,
        "issues": issues,
        "limitations": [
            "Real replay is the same exposed fixed matrix; Monte Carlo intervals are not cross-language generalization evidence.",
            "Two trace replays use the frozen v2 scheduler and v1 inference; independent arithmetic/truth checks do not independently prove the stopping theorem.",
            "Forced ranking accuracy is independently replayed for two new-policy seeds; other forced-accuracy fields are range/aggregate checked or matched to audited original records.",
            "Top-two identities are evaluator-only full-data diagnostics, not evidence that the controller knew or certified the best system.",
        ],
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
