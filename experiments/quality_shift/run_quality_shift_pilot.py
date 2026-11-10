#!/usr/bin/env python3
"""Weaker-to-stronger MT evaluation pilot."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

METHODS = ("Random", "Difficulty-stratified", "Hard-only", "All-dev variance",
           "Upper-dev variance", "Source-length-stratified")
STOCHASTIC = {"Random", "Difficulty-stratified", "Source-length-stratified"}
EPS = 1e-12


def json_write(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def order_with_ties(values: np.ndarray, rng: np.random.Generator,
                    descending: bool = False) -> np.ndarray:
    """Tie breaking is randomized without inspecting target outcomes."""
    permutation = rng.permutation(len(values))
    signed = -values if descending else values
    return permutation[np.argsort(signed[permutation], kind="stable")]


def stratified_indices(order: np.ndarray, budget: int, rng: np.random.Generator,
                       bins: int = 5) -> np.ndarray:
    groups = np.array_split(order, bins)
    take = np.full(bins, budget // bins, dtype=int)
    # In the planned budgets, the remainder is always zero.
    take[: budget % bins] += 1
    if any(t > len(g) for t, g in zip(take, groups)):
        raise ValueError("A stratum is too small for this allocation; reduce the budget.")
    selected = np.concatenate([rng.choice(g, size=int(t), replace=False)
                               for g, t in zip(groups, take)])
    assert len(selected) == budget and len(np.unique(selected)) == budget
    return selected


def select_indices(method: str, dev_scores: np.ndarray, source_lengths: np.ndarray,
                   dev_calibration_means: np.ndarray, budget: int,
                   rng: np.random.Generator, bins: int = 5,
                   upper_count: int = 3) -> np.ndarray:
    """The selector API deliberately has no target-score argument."""
    n, d = dev_scores.shape
    if not 1 <= budget <= n:
        raise ValueError("Budget must be within 1..number of evaluation items.")
    if method == "Random":
        return rng.choice(n, size=budget, replace=False)
    if method == "Difficulty-stratified":
        order = order_with_ties(-dev_scores.mean(axis=1), rng)
        return stratified_indices(order, budget, rng, bins)
    if method == "Source-length-stratified":
        order = order_with_ties(source_lengths, rng)
        return stratified_indices(order, budget, rng, bins)
    if method == "Hard-only":
        utility = -dev_scores.mean(axis=1)
    elif method == "All-dev variance":
        utility = dev_scores.var(axis=1, ddof=1)
    elif method == "Upper-dev variance":
        if not 2 <= upper_count <= d:
            raise ValueError("upper_count must be between 2 and the development group size.")
        upper = np.argsort(dev_calibration_means, kind="stable")[-upper_count:]
        utility = dev_scores[:, upper].var(axis=1, ddof=1)
    else:
        raise ValueError(f"Unknown method: {method}")
    return order_with_ties(utility, rng, descending=True)[:budget]


def score_batch(estimates: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate T x K estimated system scores. Ties get half pairwise credit."""
    estimates = np.atleast_2d(estimates)
    if estimates.shape[1] != len(reference):
        raise ValueError("Estimate/reference dimensions do not match.")
    a, b = np.triu_indices(len(reference), 1)
    dr = reference[a] - reference[b]
    de = estimates[:, a] - estimates[:, b]
    sr = np.where(abs(dr) <= EPS, 0.0, np.sign(dr))
    se = np.where(abs(de) <= EPS, 0.0, np.sign(de))
    nonzero = sr != 0
    out: dict[str, np.ndarray] = {}
    for threshold, name in [(0.0, "pairwise_accuracy"), (1.0, "pairwise_accuracy_gap1")]:
        keep = abs(dr) > max(threshold, EPS)
        if keep.any():
            credit = (se[:, keep] == sr[keep]).astype(float)
            credit[se[:, keep] == 0] = 0.5
            out[name] = credit.mean(axis=1)
        else:
            out[name] = np.full(len(estimates), np.nan)
    rank_ref = rankdata(reference) - (len(reference) + 1) / 2
    rank_est = rankdata(estimates, axis=1) - (len(reference) + 1) / 2
    denom = np.linalg.norm(rank_ref) * np.linalg.norm(rank_est, axis=1)
    out["spearman"] = np.divide(rank_est @ rank_ref, denom,
                                 out=np.full(len(estimates), np.nan), where=denom > EPS)
    denominator = np.sqrt(nonzero.sum() * (se != 0).sum(axis=1))
    out["kendall_tau_b"] = np.divide((se * sr).sum(axis=1), denominator,
                                      out=np.full(len(estimates), np.nan), where=denominator > EPS)
    true_best = np.isclose(reference, reference.max(), rtol=0, atol=EPS)
    est_best = np.isclose(estimates, estimates.max(axis=1)[:, None], rtol=0, atol=EPS)
    out["top1_accuracy"] = (est_best & true_best).sum(axis=1) / est_best.sum(axis=1)
    err = estimates - reference[None, :]
    out["mae"] = abs(err).mean(axis=1)
    out["bias"] = err.mean(axis=1)
    return out


def run_pilot(df: pd.DataFrame, output: Path, splits: int, trials: int,
              seed: int, budgets: list[int], synthetic: bool = False) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    required = {"sentence_id", "source_text"}
    if not required <= set(df):
        raise ValueError("CSV must contain sentence_id and source_text columns.")
    if df.sentence_id.isna().any() or df.sentence_id.duplicated().any():
        raise ValueError("Missing or duplicated item IDs; audit the input first.")
    systems = sorted(c for c in df if c not in required and c != "refA")
    if len(systems) < 12:
        raise ValueError("This pilot needs at least 12 systems: six target and six development.")
    y = df[systems].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all() or df.source_text.isna().any():
        raise ValueError("Incomplete source/score matrix; no silent imputation is performed.")
    ids = df.sentence_id.astype(str).to_numpy()
    # Existing WMT identifiers have LP_#_domain_#_document_#_segment.
    # Strip only the final segment component to keep whole documents together.
    groups = np.array([s.rsplit("_#_", 1)[0] if "_#_" in s else s for s in ids])
    unique_groups = np.unique(groups)
    lengths = df.source_text.astype(str).str.len().to_numpy(dtype=float)
    rows, audits, manifests = [], [], []
    for split in range(splits):
        rng = np.random.default_rng(np.random.SeedSequence([seed, split, 10]))
        n_cal_groups = math.ceil(0.25 * len(unique_groups))
        cal_groups = rng.permutation(unique_groups)[:n_cal_groups]
        cal = np.flatnonzero(np.isin(groups, cal_groups))
        evaluation = np.flatnonzero(~np.isin(groups, cal_groups))
        assert not set(groups[cal]) & set(groups[evaluation])
        if len(evaluation) < max(budgets):
            raise ValueError(f"Split {split} has only {len(evaluation)} evaluation items; reduce budgets.")
        cal_mean = y[cal].mean(axis=0)
        ordered = np.argsort(cal_mean, kind="stable")
        target = ordered[-6:]
        pool = ordered[:-6]
        groups_of_models = {
            "Lower-six": pool[:6],
            "Upper-six": pool[-6:],
            "Random-six": np.sort(rng.choice(pool, size=6, replace=False)),
        }
        ref = y[evaluation][:, target].mean(axis=0)
        full_pair_diffs = abs(ref[:, None] - ref[None, :])[np.triu_indices(6, 1)]
        target_scores = y[evaluation][:, target]
        base_manifest = {"split": split, "calibration_ids": ids[cal].tolist(),
                         "evaluation_ids": ids[evaluation].tolist(),
                         "target_systems": [systems[j] for j in target],
                         "development_groups": {k: [systems[j] for j in v]
                                                 for k, v in groups_of_models.items()}}
        manifests.append(base_manifest)
        for regime_index, (regime, dev) in enumerate(groups_of_models.items()):
            assert not set(dev) & set(target)
            dev_mean_eval = y[evaluation][:, dev].mean(axis=0)
            audits.append({"split": split, "regime": regime,
                           "n_calibration_items": len(cal), "n_evaluation_items": len(evaluation),
                           "calibration_mean_quality_gap": float(cal_mean[target].mean() - cal_mean[dev].mean()),
                           "evaluation_mean_quality_gap": float(ref.mean() - dev_mean_eval.mean()),
                           "evaluation_min_target_minus_max_dev": float(ref.min() - dev_mean_eval.max()),
                           "target_score_range": float(np.ptp(ref)),
                           "target_pair_median_gap": float(np.median(full_pair_diffs)),
                           "eligible_pairs_gap1": int((full_pair_diffs > 1).sum())})
            # Pass only development-system evaluation scores to selection.
            dev_y = y[evaluation][:, dev]
            for method_index, method in enumerate(METHODS):
                repetitions = trials if method in STOCHASTIC else 1
                for budget in budgets:
                    indices = []
                    for trial in range(repetitions):
                        # Random and source-only controls use identical subsets across regimes.
                        r_index = 0 if method in {"Random", "Source-length-stratified"} else regime_index
                        ss = np.random.SeedSequence([seed, split, r_index, method_index, budget, trial, 70])
                        chosen = select_indices(method, dev_y, lengths[evaluation], cal_mean[dev],
                                                budget, np.random.default_rng(ss))
                        indices.append(chosen)
                    selected = np.stack(indices)
                    estimates = target_scores[selected].mean(axis=1)
                    metrics = score_batch(estimates, ref)
                    for trial in range(repetitions):
                        row = {"split": split, "regime": regime, "method": method,
                               "budget": budget, "trial": trial,
                               "synthetic": synthetic, "n_evaluation_items": len(evaluation)}
                        row.update({k: float(v[trial]) for k, v in metrics.items()})
                        row["selected_indices"] = json.dumps(evaluation[selected[trial]].tolist())
                        rows.append(row)
        print(f"split {split + 1}/{splits}: calibration={len(cal)}, evaluation={len(evaluation)}", flush=True)
    trials_df = pd.DataFrame(rows)
    trials_df.to_csv(output / "trials.csv", index=False)
    metrics = list(score_batch(np.array([[0., 1.]]), np.array([0., 1.])))
    split_mean = trials_df.groupby(["split", "regime", "method", "budget"], as_index=False)[metrics].mean()
    split_mean.to_csv(output / "split_summary.csv", index=False)
    summary = split_mean.groupby(["regime", "method", "budget"], as_index=False)[metrics].mean()
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "quality_gap_audit.csv", index=False)
    json_write(output / "split_manifest.json", manifests)
    cfg = {"synthetic_only": synthetic, "n_items": len(df), "n_systems": len(systems),
           "n_documents": len(unique_groups), "splits": splits, "trials_stochastic": trials,
           "trials_deterministic": 1, "seed": seed, "budgets": budgets,
           "methods": list(METHODS), "calibration_document_fraction": .25,
           "target_systems": 6, "dev_systems": 6, "upper_dev_count": 3,
           "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "status": "synthetic test only" if synthetic else "item-level input supplied and pilot executed",
           "caution": "Repeated overlapping splits are not independent language-level replications."}
    json_write(output / "run_config.json", cfg)
    return cfg


def self_test(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    # Exact counterexample: hardest items misrank two stronger systems.
    dev = np.array([[10.,30.], [25.,35.], [75.,85.], [85.,95.]])
    target = np.array([[90.,92.], [90.,92.], [100.,94.], [100.,94.]])
    reference = target.mean(axis=0)
    all_subsets = np.array(list(itertools.combinations(range(4), 2)))
    random_acc = score_batch(target[all_subsets].mean(axis=1), reference)["pairwise_accuracy"].mean()
    hard = np.argsort(dev.mean(axis=1))[:2]
    hard_acc = score_batch(target[hard].mean(axis=0), reference)["pairwise_accuracy"][0]
    balanced = np.array([(a,b) for a in [0,1] for b in [2,3]])
    balanced_acc = score_batch(target[balanced].mean(axis=1), reference)["pairwise_accuracy"].mean()
    assert np.isclose(random_acc, 5/6) and hard_acc == 0 and balanced_acc == 1
    exact = score_batch(reference, reference)
    assert exact["mae"][0] == 0 and exact["pairwise_accuracy"][0] == 1
    tied = score_batch(np.array([[1.,1.]]), np.array([0.,1.]))
    assert tied["pairwise_accuracy"][0] == .5
    # Purely synthetic end-to-end pipeline smoke test, never a WMT result.
    rng = np.random.default_rng(812)
    n, k = 180, 20
    abilities = np.linspace(0, 12, k)
    difficulty = rng.normal(0, 6, n)
    y = np.clip(70 + abilities[None, :] - difficulty[:,None] + rng.normal(0, 4, (n,k)), 0,100)
    data = {"sentence_id": [f"synthetic_#_doc{i//3}_#_{i%3}" for i in range(n)],
            "source_text": ["x" * int(10+i%67) for i in range(n)]}
    data.update({f"model_{j:02d}": y[:,j] for j in range(k)})
    run_pilot(pd.DataFrame(data), output / "synthetic_smoke_test", splits=2, trials=4,
              seed=41, budgets=[10,25], synthetic=True)
    # Selector invariance under changing target data: API never accepts target data.
    s1=select_indices("Upper-dev variance", y[:50,:6], np.arange(50), abilities[:6], 10,
                      np.random.default_rng(90))
    y_changed=y.copy(); y_changed[:,6:]=rng.normal(200,100,y_changed[:,6:].shape)
    s2=select_indices("Upper-dev variance", y_changed[:50,:6], np.arange(50), abilities[:6], 10,
                      np.random.default_rng(90))
    assert np.array_equal(s1,s2)
    json_write(output / "self_test_report.json", {
        "status": "passed", "data_kind": "constructed synthetic examples; not WMT results",
        "checks": ["exact 5/6 uniform ranking accuracy", "hard-only reversal",
                   "balanced subset ordering", "full-evaluation identity", "tie credit",
                   "document disjointness", "system disjointness", "target-score isolation",
                   "end-to-end output generation"],
        "toy_reference_A": 95.0, "toy_reference_B": 93.0,
        "toy_random_pairwise_accuracy": float(random_acc),
        "toy_hard_pairwise_accuracy": float(hard_acc),
        "toy_balanced_pairwise_accuracy": float(balanced_acc),
        "warning": "No new WMT weaker-to-stronger selection experiment was run in this package."})
    print("SELF-TEST PASSED (synthetic data only)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path, default=Path("quality_shift_results"))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--budgets", type=int, nargs="+", default=[10,25,50,100])
    args = ap.parse_args()
    if args.self_test:
        self_test(args.out); return
    if args.data is None or not args.data.is_file():
        ap.error("Provide the existing full item-level score CSV with --data; the ranking CSV is insufficient.")
    if min(args.splits, args.trials, *args.budgets) < 1:
        ap.error("All counts and budgets must be positive.")
    if args.out.exists() and any(args.out.iterdir()):
        ap.error("Output directory is not empty; choose a new path to avoid overwriting results.")
    cfg = run_pilot(pd.read_csv(args.data), args.out, args.splits, args.trials, args.seed,
                    args.budgets)
    cfg["input_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    json_write(args.out / "run_config.json", cfg)


if __name__ == "__main__":
    main()
