"""Read-only behavioral diagnosis of frozen v1 on ten fixed existing seeds.

No policy settings or evidence tests are modified. Instrumentation records
proposed pair actions and actual label acquisitions. All replays must match the
already-published v1 output before their diagnostic measurements are accepted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pairwise_racing import RaceState, UNKNOWN, run_race
from validate import synthetic_matrix


POLICIES = ("shared_racing", "common_active", "random_pair")
REPLICATES = tuple(range(10))
NEAR_SYSTEMS = (4, 5)
NEAR_PAIR_INDEX = 14
CHECKPOINTS = (72, 180, 360, 540)


def acquisition_diagnostics(state):
    """Recompute the unchanged v1 utility solely to explain its decisions."""
    n = state.pair_n.astype(float)
    means = np.divide(state.sum_d, n, out=np.zeros_like(n), where=n > 0)
    variance = np.ones_like(n)
    enough = n > 1
    variance[enough] = np.maximum(0, (state.sum_d2[enough] - state.sum_d[enough] ** 2 / n[enough]) / (n[enough] - 1))
    difficulty = (variance + 1e-4) / (means ** 2 + .01 ** 2)
    gain = difficulty / ((n + 1) * (n + 2))
    gain[state.decisions != UNKNOWN] = 0
    ca, cb = state.counts[state.pair_a], state.counts[state.pair_b]
    benefit = np.zeros(state.n_systems)
    np.add.at(benefit, state.pair_a[ca < cb], gain[ca < cb])
    np.add.at(benefit, state.pair_b[cb < ca], gain[cb < ca])
    utility = np.where(ca < cb, benefit[state.pair_a],
                       np.where(cb < ca, benefit[state.pair_b],
                                (benefit[state.pair_a] + benefit[state.pair_b] + gain) / 2))
    return means, variance, difficulty, gain, utility


def one_run(matrix, replicate, policy, reference):
    pair_actions = []
    acquisitions = []
    original_action = RaceState.action_for_pair
    original_observe = RaceState.observe_next

    def trace_action(state, pair):
        action = original_action(state, pair)
        means, variance, difficulty, gain, utility = acquisition_diagnostics(state)
        unresolved = np.flatnonzero(state.decisions == UNKNOWN)
        other = unresolved[unresolved != NEAR_PAIR_INDEX]
        if policy == "shared_racing":
            if not np.isclose(utility[pair], np.max(utility[unresolved]), rtol=1e-12, atol=1e-15):
                raise AssertionError("Instrumented utility does not explain selected action")
        pair_actions.append({
            "cost_before": state.cost, "selected_pair_index": int(pair),
            "selected_pair": [int(state.pair_a[pair]), int(state.pair_b[pair])],
            "planned_systems": list(action), "near_pair_selected": int(pair) == NEAR_PAIR_INDEX,
            "resolved_pairs_before": int(np.count_nonzero(state.decisions != UNKNOWN)),
            "near_pair_samples_before": int(state.pair_n[NEAR_PAIR_INDEX]),
            "near_pair_decision_before": int(state.decisions[NEAR_PAIR_INDEX]),
            "near_gap_estimate": float(means[NEAR_PAIR_INDEX]),
            "near_variance_estimate": float(variance[NEAR_PAIR_INDEX]),
            "near_difficulty": float(difficulty[NEAR_PAIR_INDEX]),
            "selected_utility": float(utility[pair]),
            "near_utility": float(utility[NEAR_PAIR_INDEX]),
            "max_other_unresolved_utility": float(np.max(utility[other])) if len(other) else None,
        })
        return action

    def trace_observe(state, system, value):
        original_observe(state, system, value)
        acquisitions.append({
            "cost": state.cost, "system": int(system),
            "is_near_system": int(system) in NEAR_SYSTEMS,
            "system_counts": state.counts.tolist(),
            "resolved_pairs": int(np.count_nonzero(state.decisions != UNKNOWN)),
            "near_pair_samples": int(state.pair_n[NEAR_PAIR_INDEX]),
            "near_pair_decision": int(state.decisions[NEAR_PAIR_INDEX]),
        })

    seed = 20261113 + 10000 + replicate
    with patch.object(RaceState, "action_for_pair", trace_action), patch.object(RaceState, "observe_next", trace_observe):
        result = run_race(120, 6, lambda row, system: matrix[row, system], policy=policy,
                          seed=seed, checkpoints=CHECKPOINTS, record_trace=True)
    for field in ("decisions", "decision_costs", "resolution_costs", "cost_cells", "complete"):
        if result[field] != reference[field]:
            raise AssertionError(f"Instrumented replay differs from frozen output: {policy}/{replicate}/{field}")
    if not np.isclose(result["certified_fraction_auc"], reference["certified_fraction_auc"], rtol=0, atol=1e-13):
        raise AssertionError("Certificate AUC changed under instrumentation")
    for index, event in enumerate(pair_actions):
        event["cost_after"] = pair_actions[index + 1]["cost_before"] if index + 1 < len(pair_actions) else result["cost_cells"]
    costs90 = result["resolution_costs"]["0.9"]

    def at_budget(budget):
        actual = min(budget, result["cost_cells"])
        observed = acquisitions[:actual]
        near_labels = sum(q["is_near_system"] for q in observed)
        events = [a for a in pair_actions if a["cost_before"] < actual]
        targeted = [a for a in events if a["near_pair_selected"]]
        targeted_label_cost = sum(min(a["cost_after"], actual) - a["cost_before"] for a in targeted)
        latest = observed[-1] if observed else {"resolved_pairs": 0, "system_counts": [0] * 6,
                                               "near_pair_samples": 0, "near_pair_decision": UNKNOWN}
        return {
            "budget_cells": budget, "actual_cost_cells": actual,
            "near_system_labels": near_labels, "other_system_labels": actual - near_labels,
            "near_system_share": near_labels / actual if actual else 0.0,
            "near_pair_samples": latest["near_pair_samples"],
            "system_counts": latest["system_counts"],
            "resolved_pairs": latest["resolved_pairs"],
            "resolved_fraction": latest["resolved_pairs"] / 15,
            "near_pair_decision": latest["near_pair_decision"],
            "pair_target_actions": len(events) if policy != "common_active" else None,
            "near_pair_target_actions": len(targeted) if policy != "common_active" else None,
            "near_pair_target_action_share": len(targeted) / len(events) if events else None,
            "labels_bought_by_near_pair_target_actions": targeted_label_cost if policy != "common_active" else None,
        }

    output = {
        "replicate": replicate, "seed": seed, "policy": policy,
        "true_near_pair_gap": float(matrix[:, 4].mean() - matrix[:, 5].mean()),
        "cost90": costs90, "cost_all_pairs": result["cost_cells"],
        "near_pair_resolution_cost": result["decision_costs"][NEAR_PAIR_INDEX],
        "near_pair_resolved_by_cost90": result["decision_costs"][NEAR_PAIR_INDEX] <= costs90,
        "near_pair_resolution_reason": result["reasons"][NEAR_PAIR_INDEX],
        "snapshots": {str(b): at_budget(b) for b in CHECKPOINTS},
        "at_cost90": at_budget(costs90), "at_final": at_budget(result["cost_cells"]),
        "first_post_warmup_action": pair_actions[0] if pair_actions else None,
        "matches_frozen_validation_record": True,
    }
    trace = {"replicate": replicate, "seed": seed, "policy": policy,
             "pair_actions": pair_actions, "acquisition_diagnostics": acquisitions,
             "query_trace": result["trace"]}
    return output, trace


def summarize(records):
    summary = {}
    for policy in POLICIES:
        group = [r for r in records if r["policy"] == policy]
        out = {"runs": len(group)}
        for key in ("cost90", "cost_all_pairs", "near_pair_resolution_cost"):
            out["mean_" + key] = float(np.mean([r[key] for r in group]))
        out["near_pair_resolved_by_cost90_runs"] = sum(r["near_pair_resolved_by_cost90"] for r in group)
        for location in (*map(str, CHECKPOINTS), "at_cost90", "at_final"):
            entries = [r[location] if location.startswith("at_") else r["snapshots"][location] for r in group]
            stats = {}
            for key in ("actual_cost_cells", "near_system_labels", "other_system_labels", "near_system_share",
                        "near_pair_samples", "resolved_pairs", "resolved_fraction", "pair_target_actions",
                        "near_pair_target_actions", "near_pair_target_action_share", "labels_bought_by_near_pair_target_actions"):
                values = [entry[key] for entry in entries]
                stats["mean_" + key] = float(np.mean(values)) if all(v is not None for v in values) else None
            stats["mean_system_counts"] = np.mean([entry["system_counts"] for entry in entries], axis=0).tolist()
            if policy != "common_active":
                events = sum(e["pair_target_actions"] for e in entries)
                stats["pooled_near_pair_target_action_share"] = sum(e["near_pair_target_actions"] for e in entries) / events if events else None
            out[location] = stats
        if policy != "common_active":
            first = [r["first_post_warmup_action"] for r in group]
            out["first_post_warmup_near_pair_selected_runs"] = sum(e["near_pair_selected"] for e in first)
            ratios = [e["near_utility"] / e["max_other_unresolved_utility"] for e in first]
            out["first_post_warmup_near_vs_best_other_utility_ratios"] = ratios
            out["first_post_warmup_median_near_vs_best_other_utility_ratio"] = float(np.median(ratios))
        summary[policy] = out
    comparison = {}
    for baseline in ("common_active", "random_pair"):
        comparison[baseline] = {
            "mean_cost90_excess_cells": summary["shared_racing"]["mean_cost90"] - summary[baseline]["mean_cost90"],
            "cost90_saving_fraction": 1 - summary["shared_racing"]["mean_cost90"] / summary[baseline]["mean_cost90"],
            "mean_all_pairs_cost_difference_cells": summary["shared_racing"]["mean_cost_all_pairs"] - summary[baseline]["mean_cost_all_pairs"],
        }
    return {"by_policy": summary, "shared_racing_vs": comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "diagnostics/near_tie_v1")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refuse to overwrite prior diagnosis; choose a new output directory")
    protected = ("pairwise_racing.py", "validate.py", "protocol.json", "validation_v1/trials.jsonl",
                 "validation_v1/summary.json", "output_audit.json")
    before = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in protected}
    manifest = json.loads((ROOT / "validation_v1/run_manifest.json").read_text())
    for name in ("pairwise_racing.py", "validate.py"):
        if before[name] != manifest["source_hashes_before_run"][name]:
            raise ValueError(f"Frozen v1 source has changed: {name}")
    original = [json.loads(line) for line in (ROOT / "validation_v1/trials.jsonl").read_text().splitlines()]
    references = {(r["replicate"], r["policy"]): r for r in original if r["scenario"] == "one_near_tie"}
    args.out.mkdir(parents=True)
    records = []
    with (args.out / "traces.jsonl").open("w") as stream:
        for replicate in REPLICATES:
            matrix = synthetic_matrix("one_near_tie", replicate)
            for policy in POLICIES:
                record, trace = one_run(matrix, replicate, policy, references[replicate, policy])
                records.append(record)
                stream.write(json.dumps(trace, allow_nan=False) + "\n")
    after = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in protected}
    if before != after:
        raise AssertionError("Protected frozen v1 file changed")
    report = {
        "status": "diagnostic_complete",
        "scope": "posthoc mechanism analysis of already-used synthetic development cases; no tuning or unseen-data validation",
        "scenario": "one_near_tie", "replicates": list(REPLICATES),
        "seed_range_inclusive": [20271113, 20271122], "near_systems": list(NEAR_SYSTEMS),
        "near_pair_index": NEAR_PAIR_INDEX, "n_pairs": 15,
        "cost90_definition": "14 of 15 pairs certified (90 percent rounds up); the near pair need not be among them",
        "pair_action_definition": "one call to unchanged action_for_pair; warmup excluded; common_active does not select a target pair",
        "protected_hashes_before_and_after_match": before == after,
        "protected_hashes": before,
        "all_30_replays_match_frozen_outputs": all(r["matches_frozen_validation_record"] for r in records),
        "summary": summarize(records), "runs": records,
    }
    (args.out / "diagnostic_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.out), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
