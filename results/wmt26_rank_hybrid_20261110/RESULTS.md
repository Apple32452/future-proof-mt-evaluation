# WMT26 ranking-aware hybrid: complete 19-direction follow-up

The **predeclared ranking-aware hybrid did not improve system rankings**.
On the twelve additional WMT26 language directions, it lost to random
sampling in **all 12 directions** at the primary 10-item budget. The seven
previously examined directions provide a consistent negative secondary result.
This is an informative failed hypothesis, **not a new ranking-method gain**.

## Predeclared primary comparison: 12 previously untested directions

An item budget of 10 is **10 sentences per each of six strong target
systems**, or 60 target system--sentence scores. The same 10 sentences
are scored for all six targets. All numbers below average trials within
document-disjoint split, then average the 20 splits within each language,
then weight each language equally.

| Sampling method | Pairwise accuracy | Accuracy on >1-point pairs | Top-1 accuracy | Score MAE | Signed score bias |
|---|---:|---:|---:|---:|---:|
| Random | 77.66% | 80.26% | 52.55% | 4.70 | +0.04 |
| Difficulty-stratified | **78.24%** | **80.86%** | **53.84%** | **4.46** | -0.17 |
| Development-cluster-proportional | 77.48% | 80.05% | 52.44% | 4.60 | +0.01 |
| Hard-only | 77.34% | 79.69% | 59.38% | 8.88 | -7.65 |
| All-development variance | 77.30% | 79.44% | 50.62% | 7.24 | +4.41 |
| **Rank-aware hybrid** | **75.09%** | **77.45%** | **47.97%** | **5.78** | **-0.01** |

Against random, the new hybrid's paired language-average ranking accuracy
change is **-2.57 percentage points**, exploratory 95% language-resampling
interval **[-3.01, -2.16] percentage points**. Every one of the 12
directions is negative (range -3.87 to -1.58 points). The corresponding
change for difficulty-stratified selection is **+0.58 percentage points**,
interval **[+0.33, +0.86]**, with 11/12 directions positive. This is
well below the predeclared practical threshold of **+2 percentage points**.

The weighted hybrid has almost zero *signed* bias, as expected from its
fixed-stratum mean estimator, but its **absolute estimation error and
ranking error are both greater**. Oversampling a few difficulty/disagreement
bins appears to spend too much of a small budget in those bins, leaving
only one representative observation in other bins at budget 10. This
mechanism is an interpretation of the observed pattern, not a proven
causal mechanism.

### Secondary sample-size check: same 12 directions, budget 25

| Method | Pairwise accuracy | Score MAE |
|---|---:|---:|
| Random | 87.76% | 2.21 |
| Difficulty-stratified | 87.91% | 2.11 |
| Development-cluster-proportional | 87.77% | 2.13 |
| Rank-aware hybrid | 83.68% | 3.19 |

The hybrid again loses to random in **12/12** directions (mean -4.08
percentage points). Many held-out panels contain only 25--50 evaluation
sentences in some splits, so budget 25 is not a comparable low-cost
regime and must not replace the declared 10-item primary endpoint.

## Secondary exploratory comparison: seven previously inspected directions

At the separately declared **25-item budget**:

| Method | Pairwise accuracy | Score MAE | Signed score bias |
|---|---:|---:|---:|
| Random | 76.53% | 3.29 | -0.01 |
| Difficulty-stratified | 76.94% | 2.98 | -0.05 |
| Development-cluster-proportional | **77.22%** | **2.72** | +0.01 |
| Hard-only | 75.04% | 31.08 | -30.78 |
| Rank-aware hybrid | **73.94%** | 4.20 | +0.06 |

The hybrid is below random by **2.59 points** (all 7/7 directions),
exploratory language-resampling interval **[-3.27, -1.78] points**.
The development-cluster baseline improves by **+0.69 points** versus
random, interval **[+0.28, +1.15] points** (6/7 directions), while
reducing MAE from 3.29 to 2.72 (approximately **17.2%**). Neither
small ranking gain meets the prespecified +2-point practical target.
The hybrid also loses to random at budgets 10, 50 and 100 on the
supported subsets; see the full CSVs for budget-specific language counts.

## Data quality and limitations

- **Dataset:** finalized WMT26 cESA human-score annotations; pinned
  source SHA-256 `6023795cc592864a9d67ed6daaaada283041d0a8c87c4bfb4c3e5ae276c73b28`.
  These observations are retrospective, not newly commissioned labels.
- **Models:** among an 18-system coverage-selected panel in each direction,
  the six highest-calibration-mean systems become targets, and the six
  lowest become development systems. Roles are frozen within each split
  using only calibration documents, never target evaluation outcomes.
- **Quality shift:** across the 240 holdout document splits, the average
  target-minus-development calibration score gap is **38.21** points;
  on evaluation documents it is **30.22** points. The entire target
  group scores above the entire development group on evaluation in
  57.9% of holdout splits; calibration-only ranking is not an absolute
  guarantee of a disjoint quality gap.
- **Budgets:** the six target systems must each receive one judgment
  on every sampled sentence, so a budget of 10 items equals **60 target
  system--item scores**. This **excludes** the human setup labels for
  18 models on calibration documents and the six development models
  across the full evaluation item pool. Consequently these are
  conditional *incremental* target-label costs, **not total human
  evaluation savings**.
- **Source sampling:** WMT26 annotation allocation and the 18-system
  complete-case filter are nonrandom, restricting the estimand. Language
  directions and overlapping document splits are not independent new
  annotation studies. Language-level bootstrap intervals are
  **exploratory**, not a formal prospective statistical guarantee.
- **Method selection:** the 12-direction panel was chosen using
  observation counts only, and the hybrid formula/allocation were
  frozen before inspecting its outcomes. The earlier seven directions
  remain exploratory; no posthoc method adjustment was made.

## Reproduction and independent evidence

The implementation, frozen protocol and auditor live in
`experiments/quality_shift/`. Run with WMT26 matrices prepared according
to `experiments/quality_shift/WMT26_EXTERNAL.md`:

```bash
python -m unittest discover -s experiments/quality_shift -p 'test_*.py' -v
python -u experiments/quality_shift/run_wmt26_rank_hybrid.py \
  --matrix-dir data/wmt26_transfer_matrices \
  --out results/wmt26_rank_hybrid_20261110
python -u experiments/quality_shift/audit_wmt26_rank_hybrid.py \
  --matrix-dir data/wmt26_transfer_matrices \
  --results results/wmt26_rank_hybrid_20261110
```

The full run used seed `20261111`, 20 disjoint-document splits and 100
trials per stochastic sampler; each deterministic selector had one trial.
**All 41 synthetic/unit tests passed**. The auditor checks file digests,
calibration-only model roles, document identities, every sampled subset
and sampling weight, independently re-scores trial-zero ranking metrics,
and reaggregates the saved summaries. See `audit_manifest.json` and
per-language `audit_report.json` for the final checks.

Raw WMT26 annotations, full item-level selection traces (`trials.csv`)
and large sentence-ID split manifests are intentionally excluded from
GitHub. Their hashes are retained in per-language `completion.json`;
they can be regenerated for a fresh audit.

## Next research step

**Do not present this hybrid as an improvement.** A genuinely new ranking
method should allocate samples according to estimated *pairwise score-
difference variance*, possibly learned on calibration documents, rather
than heuristically oversampling general difficulty/disagreement. Freeze
such a proposal before testing on another unexamined data source.
Retain difficulty-stratified and development-cluster-proportional methods
as strong, inexpensive baselines. The class-project contribution is still
usefully broader than the original paper in model quality, settings,
alternative samplers, and transparent negative evidence; a claim of a
convincing superior ranking algorithm is **not** supported here.
