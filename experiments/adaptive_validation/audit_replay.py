"""Deterministic re-execution of every synthetic policy/seed/matrix in a run.

Uses the same implementation; this verifies reproducibility, not independent
statistical proof or performance on a fresh dataset.
"""
from pathlib import Path
import argparse
import hashlib
import json
from run_comparison import matrices, SEED
from published_adaptive_comparison import run_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    raw = (args.results / "trials.jsonl").read_bytes()
    manifest = json.loads((args.results / "manifest.json").read_text())
    for filename, expected in manifest["source_sha256"].items():
        path = Path(__file__).parent / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Original run source changed: {filename}")
    rows = [json.loads(line) for line in raw.splitlines() if line]
    lookup = {(r["scenario"], r["replicate"], r["policy"]): r for r in rows}
    if len(lookup) != len(rows):
        raise AssertionError("Duplicate policy/scenario/repetition")
    checked = 0
    for scenario, rep, matrix in matrices():
        for policy in manifest["protocol"]["policies"]:
            original = lookup[(scenario, rep, policy)]
            replay = run_one(matrix, policy, seed=SEED+rep, warmup_rows=15)
            if replay != {k:v for k,v in original.items()
                          if k not in ("scenario", "replicate")}:
                raise AssertionError((scenario, rep, policy))
            checked += 1
    assert checked == len(rows)
    result = {
        "status": "PASS",
        "checked_exact_trial_replays": checked,
        "independence": "Same synthetic data and code; deterministic reproducibility only",
    }
    (args.results / "replay_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
