"""Independent consistency audit of frozen v1 trial outputs.

Reads all trial records, reconstructs finite-matrix truth independently, and
replays two real-data seeds with full query traces. Prints JSON; does not modify
the algorithm, protocol, inputs, or validation outputs.
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

from pairwise_racing import run_race


SCENARIOS = ("well_separated", "one_near_tie", "all_tied", "heterogeneous_noise",
             "strong_item_effect", "weak_separation")
POLICIES = ("shared_racing", "random_pair", "common_all", "common_active")
UNKNOWN = 2
ROOT = Path(__file__).resolve().parent


def exact_signs(matrix):
    sums = [sum((Fraction.from_float(float(v)) for v in matrix[:, s]), Fraction())
            for s in range(matrix.shape[1])]
    return [int(sums[a] > sums[b]) - int(sums[a] < sums[b])
            for a in range(len(sums)) for b in range(a + 1, len(sums))]


def reconstruct_synthetic(scenario, replicate):
    rng = np.random.default_rng(np.random.SeedSequence([20261113, SCENARIOS.index(scenario), replicate]))
    n, k = 120, 6
    if scenario == "well_separated":
        matrix = np.linspace(.1, .9, k) + rng.normal(0, .02, (n, 1)) + rng.normal(0, .02, (n, k))
    elif scenario == "one_near_tie":
        matrix = np.array([.2, .38, .55, .72, .85, .852]) + rng.normal(0, .04, (n, k))
    elif scenario == "all_tied":
        population = np.r_[np.zeros(n // 2), np.ones(n // 2)]
        return np.column_stack([rng.permutation(population) for _ in range(k)])
    elif scenario == "heterogeneous_noise":
        matrix = np.array([.2, .35, .5, .65, .8, .9]) + rng.normal(size=(n, k)) * np.array([.02, .25, .02, .18, .01, .08])
    elif scenario == "strong_item_effect":
        matrix = rng.uniform(.05, .4, (n, 1)) + np.linspace(.05, .4, k) + rng.normal(0, .01, (n, k))
    elif scenario == "weak_separation":
        matrix = np.linspace(.48, .52, k) + rng.normal(0, .18, (n, k))
    else:
        raise ValueError(scenario)
    return np.clip(matrix, 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "validation_v1")
    parser.add_argument("--real-csv", type=Path, default=ROOT / "inputs/cs-de_DE_human_scores.csv")
    args = parser.parse_args()
    manifest = json.loads((args.results / "run_manifest.json").read_text())
    if not (args.results / "summary.json").is_file():
        raise RuntimeError("Validation is not yet complete: summary.json is absent")
    issues = []
    checked = Counter()

    def require(condition, message):
        checked["assertions"] += 1
        if not condition:
            issues.append(message)

    def close(actual, expected, message, tol=1e-12):
        require(math.isfinite(actual) and math.isclose(actual, expected, abs_tol=tol, rel_tol=tol), message)

    for name, expected in manifest["source_hashes_before_run"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
                f"Source changed since frozen run: {name}")
    require(hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest() == manifest["protocol_sha256"],
            "Protocol changed since frozen run")
    rows = [json.loads(line) for line in (args.results / "trials.jsonl").read_text().splitlines()]
    expected_synthetic_reps = manifest["arguments"]["synthetic_reps"]
    expected_real_reps = manifest["arguments"]["real_reps"]
    expected_rows = (len(SCENARIOS) * expected_synthetic_reps + expected_real_reps) * len(POLICIES)
    require(len(rows) == expected_rows, "Wrong number of trial records")
    identities = [(r["scenario"], r["replicate"], r["policy"]) for r in rows]
    require(len(set(identities)) == len(identities), "Duplicate trial identity")
    expected_identities = {(s, i, p) for s in SCENARIOS for i in range(expected_synthetic_reps) for p in POLICIES}
    expected_identities |= {("cs_de_development", i, p) for i in range(expected_real_reps) for p in POLICIES}
    require(set(identities) == expected_identities, "Missing/unexpected trial identity")

    # Read independently of loader.py and check the pinned historical identity.
    input_bytes = args.real_csv.read_bytes()
    real_sha256 = hashlib.sha256(input_bytes).hexdigest()
    require(real_sha256 == "f6de1e8da36fb67c7d99ec354e8807a016045c2d5b420f3ab8aeb72816bc279d",
            "Real matrix checksum differs from frozen historical input")
    with args.real_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        systems = [c for c in reader.fieldnames if c not in ("sentence_id", "source_text")]
        real_matrix = np.array([[float(row[c]) / 100 for c in systems] for row in reader])
    require(real_matrix.shape == (231, 20), "Unexpected real-data shape")
    require(systems == manifest["real_input"]["systems"], "System ordering differs from manifest")
    real_truth = exact_signs(real_matrix)
    truths = {("cs_de_development", r): real_truth for r in range(expected_real_reps)}
    for s in SCENARIOS:
        for rep in range(expected_synthetic_reps):
            truths[s, rep] = exact_signs(reconstruct_synthetic(s, rep))

    wrong_runs = Counter()
    wrong_edges = Counter()
    grouped = defaultdict(list)
    for record in rows:
        s, rep, policy = record["scenario"], record["replicate"], record["policy"]
        label = f"{s}/{rep}/{policy}"
        grouped[s, policy].append(record)
        n, k, pairs, total = record["items"], record["systems"], record["pairs"], record["full_cost_cells"]
        require(pairs == k * (k - 1) // 2, label + ": wrong pair count")
        require(total == n * k, label + ": wrong full cost")
        require(record["true_signs"] == truths[s, rep], label + ": incorrect finite-matrix truth")
        decision = np.array(record["decisions"])
        costs = np.array(record["decision_costs"])
        truth = np.array(truths[s, rep])
        require(decision.shape == truth.shape and costs.shape == truth.shape, label + ": invalid decision dimensions")
        require(all(x in (-1, 0, 1, UNKNOWN) for x in decision), label + ": invalid decision code")
        known = decision != UNKNOWN
        wrong = known & (decision != truth)
        full_cost = record["cost_cells"]
        require(isinstance(full_cost, int) and 0 <= full_cost <= total, label + ": cost exceeds census")
        require(bool(known.all()) == record["complete"], label + ": inconsistent completion flag")
        close(record["resolved_fraction"], float(known.mean()), label + ": resolved fraction mismatch")
        require(int(wrong.sum()) == record["wrong_certificates"], label + ": wrong-edge count mismatch")
        require(bool(wrong.any()) == record["any_wrong_certificate"], label + ": wrong-run flag mismatch")
        require(np.all(costs[known] >= 1) and np.all(costs[known] <= full_cost), label + ": invalid decision time")
        require(np.all(costs[~known] == -1), label + ": unresolved decision charged time")
        require(sum(record["reasons_count"].values()) == int(known.sum()), label + ": reason counts inconsistent")
        if known.all():
            require(max(costs) == full_cost, label + ": run failed to stop when complete")
        auc = float(np.sum(total - costs[known]) / (total * pairs))
        close(record["certified_fraction_auc"], auc, label + ": certificate AUC mismatch")
        ranked_costs = sorted(costs[known].tolist())
        for fraction in (.5, .9, .95, 1.0):
            target = int(math.ceil(fraction * pairs))
            expected = ranked_costs[target - 1] if len(ranked_costs) >= target else None
            observed = record["resolution_costs"][str(fraction)]
            require(observed == expected, label + f": cost at {fraction} resolution mismatch")
            if observed is not None:
                require(observed <= full_cost <= total, label + ": resolution cost exceeds final/census")
        for fraction in (.1, .25, .5, .75):
            snapshot = record["snapshots"][str(fraction)]
            budget = int(total * fraction)
            actual = min(budget, full_cost)
            require(snapshot["budget_cells"] == budget, label + ": wrong requested checkpoint")
            require(snapshot["actual_cost_cells"] == actual, label + ": dishonest early-stop/checkpoint cost")
            prefix_known = known & (costs <= actual)
            prefix_correct = prefix_known & (decision == truth)
            close(snapshot["resolved_fraction"], float(prefix_known.mean()), label + ": checkpoint resolution mismatch")
            close(snapshot["correct_certified_fraction_all_pairs"], float(prefix_correct.mean()), label + ": checkpoint correct fraction mismatch")
            if prefix_known.any():
                close(snapshot["conditional_certified_accuracy"], float(prefix_correct.sum() / prefix_known.sum()), label + ": checkpoint conditional accuracy mismatch")
            else:
                require(snapshot["conditional_certified_accuracy"] is None, label + ": empty certificate accuracy must be null")
            require(0 <= snapshot["forced_pair_accuracy"] <= 1, label + ": forced accuracy range violation")
        require(record["human_rating_equivalents"] == (2 * full_cost if s == "cs_de_development" else None),
                label + ": historical rating-equivalent cost mismatch")
        wrong_runs[s, policy] += bool(wrong.any())
        wrong_edges[s, policy] += int(wrong.sum())
        checked["trial_records"] += 1

    # Independent aggregate checks (the percentile bootstrap is not reimplemented).
    published = json.loads((args.results / "summary.json").read_text())
    for (s, policy), records in grouped.items():
        out = published[s]["policies"][policy]
        require(out["runs"] == len(records), f"{s}/{policy}: summary run count")
        require(out["runs_with_any_wrong_certificate"] == wrong_runs[s, policy], f"{s}/{policy}: summary errors")
        close(out["mean_cost_all_pairs"], float(np.mean([r["cost_cells"] for r in records])), f"{s}/{policy}: summary final cost")
        close(out["mean_cost_90_percent"], float(np.mean([r["resolution_costs"]["0.9"] for r in records])), f"{s}/{policy}: summary 90-percent cost")
        close(out["mean_certified_fraction_auc"], float(np.mean([r["certified_fraction_auc"] for r in records])), f"{s}/{policy}: summary AUC")
    for s in published:
        ours = np.mean([r["resolution_costs"]["0.9"] for r in grouped[s, "shared_racing"]])
        for base in ("random_pair", "common_all", "common_active"):
            reference = np.mean([r["resolution_costs"]["0.9"] for r in grouped[s, base]])
            close(published[s]["shared_racing_vs"][base]["cost90_saving_fraction"],
                  float(1 - ours / reference), f"{s}: summary relative saving against {base}")

    replay_checks = []
    for record in rows:
        if record["scenario"] != "cs_de_development" or record["replicate"] not in (0, 1):
            continue
        calls = []
        def query(row, system):
            calls.append((row, system))
            return real_matrix[row, system]
        checkpoints = [int(real_matrix.size * f) for f in (.1, .25, .5, .75)]
        replay = run_race(231, 20, query, policy=record["policy"], seed=record["seed"],
                          checkpoints=checkpoints, record_trace=True)
        label = f"replay/{record['replicate']}/{record['policy']}"
        require(len(calls) == len(set(calls)) == replay["cost_cells"] == len(replay["trace"]),
                label + ": duplicated or uncharged query")
        require(sum(replay["system_counts"]) == len(calls), label + ": counts differ from query log")
        order = np.random.default_rng(np.random.SeedSequence([record["seed"], 1])).permutation(231)
        for system in range(20):
            seen = [row for row, s in calls if s == system]
            require(seen == order[:len(seen)].tolist(), label + ": nonprefix query order")
        for index, (event, (row, system)) in enumerate(zip(replay["trace"], calls), 1):
            require(event == {"cost": index, "row": row, "system": system, "value": real_matrix[row, system]},
                    label + ": trace differs from callback")
        for key in ("cost_cells", "decisions", "decision_costs", "resolution_costs", "complete"):
            require(replay[key] == record[key], label + ": nondeterministic " + key)
        close(replay["certified_fraction_auc"], record["certified_fraction_auc"], label + ": AUC replay mismatch")
        for fraction, budget in zip((.1, .25, .5, .75), checkpoints):
            original = record["snapshots"][str(fraction)]
            snap = replay["snapshots"][str(budget)]
            means = snap["sample_means"]
            predictions = [int(means[a] > means[b]) - int(means[a] < means[b]) for a in range(20) for b in range(a + 1, 20)]
            credit = [1.0 if pred == true else .5 if pred == 0 and true != 0 else 0.0
                      for pred, true in zip(predictions, real_truth)]
            close(original["forced_pair_accuracy"], float(np.mean(credit)), label + ": forced-accuracy replay mismatch")
        replay_checks.append({"replicate": record["replicate"], "policy": record["policy"],
                              "seed": record["seed"], "query_events": len(calls),
                              "decision_and_cost_replay": "matched"})
        checked["replayed_query_events"] += len(calls)
    require(len(replay_checks) == 8, "Expected eight traced real-data replays")
    report = {
        "status": "PASS" if not issues else "FAIL",
        "scope": "Independent output/metric recomputation and eight traced development replays; not a proof or new independent dataset",
        "checked": dict(checked),
        "expected_trial_records": expected_rows,
        "trials_sha256": hashlib.sha256((args.results / "trials.jsonl").read_bytes()).hexdigest(),
        "real_input_sha256": real_sha256,
        "independent_real_truth_pairs": len(real_truth),
        "independent_synthetic_populations": len(SCENARIOS) * expected_synthetic_reps,
        "wrong_certificate_runs": {f"{s}/{p}": wrong_runs[s, p] for s, p in sorted(grouped)},
        "wrong_certificates": {f"{s}/{p}": wrong_edges[s, p] for s, p in sorted(grouped)},
        "replays": replay_checks,
        "issues": issues,
        "limitations": ["Bootstrap percentile intervals not independently recomputed.",
                        "Trace replays use the frozen controller; arithmetic/record checks and finite-matrix truth are independent.",
                        "Development matrix means are not noise-free estimates of latent human preferences."],
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
