# WMT26 ranking-aware hybrid: frozen exploratory protocol

**Frozen plan created before inspecting any sampling outcomes from the twelve
new direction panels.** This is still a retrospective, exploratory extension:
we have already inspected the previous seven WMT26 directions and their
human-score outcomes. It is not a registered prospective human trial.

## Question

Does weighted, development-score-based oversampling of ranking-informative
sentences improve the ranking of six strong MT systems **without the severe
absolute-score bias** seen with unweighted hard-only selection?

## Ranking-aware hybrid, budget 25

1. Use only six *development* systems' item-level human scores on the
   evaluation pool. Rank every item by an equal-weight mean of its
   percentile difficulty (negative mean dev cESA) and percentile
   between-dev-system disagreement (standard deviation).
2. Partition items into five approximately equal-sized utility bins.
3. Randomly draw ten representative items across all five bins (two/bin).
4. Allocate fifteen additional draws to the two highest-utility bins,
   alternating bin priorities and respecting no-replacement capacity.
5. Estimate each target's mean via the unbiased-by-design stratified
   estimate: sum over bins of (N_bin / N_total) × mean sampled target cESA
   in that bin. **No target evaluation score affects selection or weights.**

For other budgets B: start with at least one per bin and approximately
40% representative allocation; distribute the remainder to the highest
utility two bins with a documented capacity fallback. The exact rule is
in [`rank_hybrid_methods.py`](rank_hybrid_methods.py).

## Experimental setup

- Exact same filtered, 18-system WMT26 complete-case panels as the prior
  quality-calibrated study. For each of 20 document-disjoint splits, choose
  the six top-rated targets and six bottom-rated development systems
  using **only the 25% calibration-document cESA ratings**.
- Controls: Random, Hard-only, Difficulty-stratified, All-development
  variance, and the new cluster-proportional method from the prior study.
  All methods share the same split-specific targets, development systems,
  held-out evaluation pool and judgment budget.
- **Primary generalization cohort:** 12 *additional* WMT26 directions
  selected **using observation counts only** and not previously used in
  sampling-method comparisons: Czech→German, English→Belarusian, Czech,
  German, Estonian, Armenian, Icelandic, Kazakh, Ligurian, Northern Sámi,
  Simplified Chinese, Traditional Chinese. Primary budget **10 selected
  sentences for each of six targets** (60 system--sentence scores), because
  most selected panels have only 50--84
  eligible segments. Budget 25 is supplementary (often close to
  full-evaluation coverage on those small panels).
- **Secondary exploratory cohort:** the seven previously examined
  WMT26 directions (English→Egyptian Arabic, Russian, Japanese, Ukrainian,
  Indonesian, Thai; Chinese→Japanese), budget **25**; additional budgets
  10/50/100 where feasible.
- 20 seeded whole-document splits and 100 trials per stochastic method;
  deterministic methods one. Fixed seed `20261111`.
- Metrics: primary **pairwise ranking accuracy** (target pairs whose true
  cESA means differ), supplementary accuracy for >1-point pairs, top-one
  correctness, Spearman/Kendall, MAE and signed bias. Equal language weight
  in the macro average. Bootstrap uncertainty over language directions is
  exploratory because languages/splits are not fully independent.
- **Practical-effect reference threshold:** +2 percentage points in
  pairwise accuracy against random. The method must also be compared with
  hard-only and cluster-proportional, and its bias disclosed. This
  threshold is not guaranteed and is not a significance test.

## Interpretation limits

1. Stronger targets mean top six *on calibration documents* among the
   18 coverage-selected systems; not necessarily the top WMT26 models.
   Never filter on realized evaluation quality gaps.
2. Complete-case cESA observation coverage is nonrandom; WMT26 uses
   dynamic allocation of human evaluation. Retrospective source and
   version details: [`WMT26_EXTERNAL.md`](WMT26_EXTERNAL.md).
3. **Incremental budgets exclude prior human annotations** needed for
   all 18 calibration models and all six development models over the
   full evaluation pool. This is a conditional target-judgment efficiency
   study, **not** end-to-end human-annotation cost savings. One sampled
   sentence counts as six target-system scores (budget 25 = 150 such
   scores). No new human labels were commissioned: the study simulates
   selective evaluation using previously collected WMT26 annotations.
4. The 12 new directions supply a meaningful **method-outcome holdout**
   but are **not independent human annotator/data collection campaigns**.
   Small item cores and shared year/system families limit generalization.
5. This study is not tuned after inspecting its twelve-direction results.
   Any further changes must be separate, transparently labeled experiments.

## Reproduction

Prepare WMT26 matrices as documented in `WMT26_EXTERNAL.md`, then run:

```bash
python3 -m unittest discover -s experiments/quality_shift -p 'test_*.py' -v
python3 -u experiments/quality_shift/run_wmt26_rank_hybrid.py \
  --matrix-dir data/wmt26_transfer_matrices \
  --out results/wmt26_rank_hybrid_20261110
python3 -u experiments/quality_shift/audit_wmt26_rank_hybrid.py \
  --matrix-dir data/wmt26_transfer_matrices \
  --results results/wmt26_rank_hybrid_20261110
```

Original source data, item-level trial logs and large split-ID manifests
are not committed; the compact summaries, checksums and auditor reports are.
