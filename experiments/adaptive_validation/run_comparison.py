"""Matched certificate comparisons: v2, random allocation and published policies.

Run on a public human-score matrix with columns item_id, doc_id, one column
per system, and normalized scores within the predeclared [0,1] bounds.
If --matrix is omitted, run frozen synthetic development scenarios only.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
from published_adaptive_comparison import POLICIES, run_one

SEED = 2026111501
PROTO = {
    "study": "preliminary_stronger_published_baselines_v1",
    "publication": "Zouhar et al., Findings EMNLP 2026, 2026.findings-emnlp.360",
    "same_rule_for_all": "shared sequential familywise pair certificates, alpha=0.05",
    "warmup_rows_per_system": 15,
    "cost_unit": "unique system-item mean score cell",
    "policies": list(POLICIES),
    "primary": "cost to certify 90% of all pairs, censored at N*K",
    "secondary": ["correctness", "all-pairs cost", "certified fraction at half budget"],
    "note": "published rules adapted to new stopping objective, not exact published-paper benchmark",
    "fixed_seed": SEED,
    "selection": "only observed cells; no full-data oracle",
}


def matrices():
    scenario_id = {"well_separated": 1, "near_tie": 2, "item_effect": 3}
    for scenario in scenario_id:
        for rep in range(12):
            rng = np.random.default_rng(np.random.SeedSequence(
                [SEED, rep, scenario_id[scenario]]))
            n, k = 120, 8
            mu = np.linspace(.82, .26, k)
            if scenario == "near_tie":
                mu[:2] = [.80, .795]
            shock = rng.normal(0, .13 if scenario == "item_effect" else .045, (n, 1))
            noise = rng.normal(0, .13 if scenario != "item_effect" else .07, (n, k))
            y = np.clip(mu[None, :] + shock + noise, 0, 1)
            yield scenario, rep, y


def load_matrix(path):
    import pandas as pd
    df = pd.read_csv(path)
    if not {"item_id", "doc_id"}.issubset(df.columns):
        raise ValueError("Missing item_id/doc_id columns")
    if df.item_id.isna().any() or df.item_id.duplicated().any():
        raise ValueError("Duplicate or null item IDs")
    systems = [col for col in df.columns if col not in ("item_id", "doc_id")]
    if len(systems) < 2:
        raise ValueError("At least two MT systems required")
    y = df[systems].to_numpy(dtype=float)
    if len(df) < 15 or not np.all(np.isfinite(y)) or np.any((y < 0) | (y > 1)):
        raise ValueError("Insufficient rows, missing or unbounded human scores")
    return y, {
        "n_items": int(y.shape[0]),
        "systems": systems,
        "input_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "score_bounds": [0, 1],
    }


def run_cases(cases, out, kind, metadata):
    if out.exists():
        raise FileExistsError("Choose a new directory; outputs cannot be overwritten")
    out.mkdir(parents=True)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in Path(__file__).parent.glob("*.py")}
    manifest = {
        "protocol": PROTO,
        "environment": {"python": sys.version, "numpy": np.__version__,
                        "platform": platform.platform()},
        "data_role": kind,
        "input_metadata": metadata,
        "source_sha256": hashes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    allresults = []
    with (out / "trials.jsonl").open("w") as handle:
        for scenario, rep, y in cases:
            seed = SEED + rep + (1000 if kind == "external" else 0)
            for policy in POLICIES:
                row = {"scenario": scenario, "replicate": rep,
                       **run_one(y, policy, seed=seed, warmup_rows=15)}
                handle.write(json.dumps(row) + "\n")
                allresults.append(row)
            handle.flush()
            print(f"completed {scenario} repeat {rep+1}", flush=True)
    summary = {}
    for scenario in sorted({r["scenario"] for r in allresults}):
        group = [r for r in allresults if r["scenario"] == scenario]
        summary[scenario] = {}
        for policy in POLICIES:
            vals = [r for r in group if r["policy"] == policy]
            summary[scenario][policy] = {
                "runs": len(vals),
                "mean_cost90": float(np.mean([r["cost_90"] for r in vals])),
                "mean_costall": float(np.mean([r["cost_all"] for r in vals])),
                "mean_certified_half": float(np.mean([r["resolved_half"] for r in vals])),
                "posthoc_weighted_tau_half": float(np.mean(
                    [r["posthoc_top_weighted_tau_half"] for r in vals])),
                "posthoc_top1_half": float(np.mean([r["posthoc_top1_half"] for r in vals])),
                "runs_with_wrong_certificates": sum(r["any_wrong"] for r in vals),
                "reached90": sum(r["reached_90"] for r in vals),
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "completion.json").write_text(json.dumps({
        "complete": True,
        "rows": len(allresults),
        "trials_sha256": hashlib.sha256((out / "trials.jsonl").read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, help="Normalized external human-score CSV")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()
    if args.matrix:
        y, meta = load_matrix(args.matrix)
        cases = ((args.matrix.stem, rep, y) for rep in range(args.reps))
        summary = run_cases(cases, args.out, "external_unverified_until_cohort_independence_reviewed", meta)
    else:
        summary = run_cases(matrices(), args.out, "synthetic_dev_only",
                            {"scenario_count": 3, "replicates_per_scenario": 12})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
