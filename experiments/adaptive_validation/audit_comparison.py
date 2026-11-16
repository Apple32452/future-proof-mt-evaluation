"""Arithmetic/provenance audit of the adaptive MT policy trial summaries.

Recalculates trial-log summaries and checks score constraints.
Does NOT independently re-execute acquisition policies.
"""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    out = args.directory
    completion = json.loads((out / "completion.json").read_text())
    assert completion["complete"] is True
    raw = (out / "trials.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == completion["trials_sha256"]
    rows = [json.loads(line) for line in raw.splitlines() if line]
    assert len(rows) == completion["rows"]
    summary = json.loads((out / "summary.json").read_text())
    grouped = defaultdict(list)
    incorrect = 0
    for row in rows:
        assert 0 <= row["resolved_half"] <= 1
        assert 0 <= row["resolved_end"] <= 1
        assert row["cost_spent"] <= row["items"] * row["systems"]
        assert 0 <= row["wrong_certificates"] <= row["systems"] * (row["systems"] - 1) // 2
        assert row["any_wrong"] == (row["wrong_certificates"] > 0)
        if row["reached_90"]:
            assert row["cost_90"] <= row["cost_spent"]
        if row["reached_all"]:
            assert row["cost_all"] <= row["cost_spent"]
        grouped[(row["scenario"], row["policy"])].append(row)
        incorrect += row["wrong_certificates"]
    checks = 0
    for scenario, methods in summary.items():
        for policy, expected in methods.items():
            group = grouped[(scenario, policy)]
            assert len(group) == expected["runs"]
            for column, name in [
                ("cost_90", "mean_cost90"),
                ("cost_all", "mean_costall"),
                ("resolved_half", "mean_certified_half"),
                ("posthoc_top_weighted_tau_half", "posthoc_weighted_tau_half"),
                ("posthoc_top1_half", "posthoc_top1_half"),
            ]:
                assert abs(float(np.mean([row[column] for row in group])) -
                           expected[name]) < 1e-9
                checks += 1
            assert sum(row["any_wrong"] for row in group) == expected["runs_with_wrong_certificates"]
            checks += 1
    result = {
        "audit": "PASS", "trial_rows": len(rows), "summary_checks": checks,
        "total_wrong_certificates": incorrect,
        "independent_recalculation": "summary/constraints, no second policy replay",
        "note": "External-human evidence only when source matrix has been verified",
    }
    (out / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
