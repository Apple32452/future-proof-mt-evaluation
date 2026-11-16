#!/usr/bin/env python3
"""Re-execute archived adaptive-evaluation trials on the exact original score matrix.

Uses the original policy source files, not an independently implemented
statistical procedure. Designed to strengthen the summary-only audit.
Example:
 python replay_check.py results/wmt22_ende --matrix inputs/wmt22_sqm_ende.csv \
   --runner-dir . --replicates 0 1 --out results/wmt22_ende/replay_check.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

METRICS = (
    "seed", "items", "systems", "posthoc_top_weighted_tau_half",
    "posthoc_top1_half", "cost_90", "reached_90", "cost_all",
    "reached_all", "cost_spent", "resolved_half",
    "wrong_certificates", "any_wrong", "resolved_end",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(out: Path, matrix: Path, runner: Path, replicates: list[int]):
    manifest = json.loads((out / "manifest.json").read_text())
    completion = json.loads((out / "completion.json").read_text())
    if sha(out / "trials.jsonl") != completion["trials_sha256"]:
        raise ValueError("Trial checksum differs from completion record")
    if sha(matrix) != manifest["input_metadata"]["input_sha256"]:
        raise ValueError("Matrix checksum differs from original run manifest")
    for name, expected in manifest["source_sha256"].items():
        path = runner / name
        if not path.exists() or sha(path) != expected:
            raise ValueError(f"Original runner code checksum mismatch: {name}")

    archived = [json.loads(line)
                for line in (out / "trials.jsonl").read_text().splitlines()
                if line.strip()]
    selected = [row for row in archived if int(row["replicate"]) in replicates]
    if len(selected) != len(replicates) * 5:
        raise ValueError("Missing one or more methods in selected paired replicates")
    sys.path.insert(0, str(runner.resolve()))
    from run_comparison import load_matrix
    from published_adaptive_comparison import run_one
    y, meta = load_matrix(matrix)
    if meta["input_sha256"] != manifest["input_metadata"]["input_sha256"]:
        raise ValueError("Matrix parser has inconsistent checksum")
    checked = []
    for old in selected:
        fresh = run_one(y, old["policy"], seed=int(old["seed"]),
                        warmup_rows=int(manifest["protocol"]["warmup_rows_per_system"]))
        for field in METRICS:
            a, b = old[field], fresh[field]
            same = (math.isclose(a, b, rel_tol=0, abs_tol=1e-10)
                    if isinstance(a, (int, float)) and not isinstance(a, bool)
                    else a == b)
            if not same:
                raise ValueError(f"Replay mismatch on replicate {old['replicate']}, "
                                 f"{old['policy']}, {field}: {a!r} vs {b!r}")
        checked.append({"replicate": old["replicate"], "policy": old["policy"],
                        "seed": old["seed"], "matches": True})
        print(f"MATCH rep {old['replicate']} {old['policy']}", flush=True)
    return {
        "status": "PASS",
        "original_matrix_sha256": sha(matrix),
        "original_code_sha256_verified": True,
        "checked": checked,
        "limitation": "Exact same-code deterministic replay, not independent algorithm verification",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--runner-dir", type=Path, required=True)
    p.add_argument("--replicates", type=int, nargs="+", default=[0, 1])
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    result = replay(a.directory, a.matrix, a.runner_dir, a.replicates)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "replays": len(result["checked"])}, indent=2))


if __name__ == "__main__":
    main()
