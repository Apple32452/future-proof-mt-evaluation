# Calibration-driven pairwise-variance sampling: WMT24/WMT26 results

**Outcome: ranking hypothesis not supported on fresh WMT24 data.**
The method and the practical-success threshold were frozen before inspecting
WMT24 ranking outcomes; no post-result changes to the sampling rule were made.
The entire 16-direction run and all its selection traces passed independent
replay and aggregation audits.

## Main result: ranking four high-calibration systems

The primary metric is pairwise accuracy among four calibration-selected
target systems at **25 evaluation items per target**, or 100 incremental
target-system/item judgments. Macro averages give each language direction
equal weight. Differences below are percentage points relative to random.

| Cohort | Sampling method | Pairwise accuracy | Change vs random | 95% paired language bootstrap interval | Top-one accuracy | Score MAE |
|:--|:--|--:|--:|:--|--:|--:|
| **WMT24 fresh nine** | Random | 76.88% | baseline | — | 61.54% | 2.530 |
| WMT24 fresh nine | **Cal-pairvar-Neyman** | **76.79%** | **−0.09 pp** | **[−0.60, +0.31] pp** | 61.38% | **2.453** |
| WMT24 fresh nine | Cal-pairvar-proportional | 76.63% | −0.25 pp | [−0.54, +0.03] pp | 61.39% | 2.493 |
| WMT24 fresh nine | Difficulty-stratified | 76.68% | −0.20 pp | [−0.61, +0.18] pp | 61.38% | 2.452 |
| WMT24 fresh nine | Dev-cluster-proportional | 76.71% | −0.16 pp | [−0.65, +0.30] pp | 61.11% | 2.455 |
| **WMT26 exploratory seven** | Random | 75.68% | baseline | — | 57.11% | 3.283 |
| WMT26 exploratory seven | **Cal-pairvar-Neyman** | **76.09%** | **+0.41 pp** | **[−0.05, +0.82] pp** | 57.87% | **3.129** |
| WMT26 exploratory seven | Cal-pairvar-proportional | 76.03% | +0.35 pp | [+0.01, +0.72] pp | 57.80% | 3.164 |
| WMT26 exploratory seven | Difficulty-stratified | 76.12% | +0.44 pp | [+0.09, +0.74] pp | 57.40% | 2.993 |
| WMT26 exploratory seven | Dev-cluster-proportional | **76.43%** | **+0.74 pp** | [+0.27, +1.22] pp | 58.09% | **2.830** |

The prespecified success target was **at least +2.00 pp** over random on
WMT24 at budget 25. The observed **−0.09 pp** does not satisfy it. The
pairwise variance-based allocation beats its own proportional control by
only **+0.16 pp** on WMT24 and **+0.06 pp** on WMT26. Five of nine fresh
WMT24 languages improve against random, while four deteriorate. The
WMT26 exploratory result is positive but small and below the
stronger development-cluster-proportional baseline.

Calibration-driven allocation reduced score MAE versus random by **0.077
score points** on WMT24 (8 of 9 languages; exploratory paired CI
[+0.028, +0.134]) and **0.154 points** on WMT26 (6 of 7 languages;
CI [+0.071, +0.242]). These are score-estimation improvements, *not*
evidence of a new reliable ranking advantage. Score levels must not be
compared directly across years: WMT24 uses ESA and WMT26 uses cESA.

## Budget sensitivity (pairwise accuracy)

| Cohort | Budget per target | Random | Cal-pairvar-Neyman | Change vs random | Directions |
|:--|--:|--:|--:|--:|--:|
| WMT24 fresh | 10 | 69.94% | 69.52% | −0.43 pp | 9 |
| WMT24 fresh | **25** | **76.88%** | **76.79%** | **−0.09 pp** | **9** |
| WMT24 fresh | 50 | 82.03% | 82.21% | +0.18 pp | 9 |
| WMT24 fresh | 100 | 87.34% | 87.31% | −0.03 pp | 9 |
| WMT26 exploratory | 10 | 68.59% | 68.59% | +0.01 pp | 7 |
| WMT26 exploratory | **25** | **75.68%** | **76.09%** | **+0.41 pp** | **7** |
| WMT26 exploratory | 50 | 82.43% | 82.57% | +0.14 pp | 7 |
| WMT26 exploratory | 100 | 87.26% | 87.31% | +0.06 pp | **6** |

The WMT26 English-to-Ukrainian panel cannot consistently support 100
evaluation items after document-disjoint calibration, so budget 100 covers
six rather than seven languages. Larger budgets are **secondary analyses**.

## Interpretation and a revealing hard-only control

Hard-only selection attained **79.91%** pairwise accuracy on WMT24 at
budget 25 (+3.03 pp vs random), but produced mean absolute score error
**13.02** and signed score bias **−12.95** (versus random MAE **2.53**).
On WMT26 it fell to **72.50%** pairwise accuracy (−3.18 pp vs random)
and MAE **27.25**. Its WMT24 ranking gain thus cannot justify a reliable
general evaluation estimator, and it did not transfer to the newer cohort.

One possible explanation for the variance method's weak ranking gain is
that Neyman allocation minimizes a weighted *variance of estimated pair
differences*, which need not optimize the *probability of correct pair
signs*, especially when calibration variance is noisy, differences are
small or score distributions are non-Gaussian. That is an interpretation,
not a separately verified causal mechanism.

## Data, cost and inference boundaries

- WMT24 primary: official ESA waves at
  [wmt-conference/wmt24-news-systems](https://github.com/wmt-conference/wmt24-news-systems/tree/fabe1f81d65449b3e3f9f8d37da280fa6cf3650a),
  9 language directions with **285–991** complete jointly scored items
  each, 10 coverage-selected machine systems. Only **role-calibration**
  judgments select the four target and four development identities;
  variance-calibration documents are separate from role calibration and
  the evaluation document split.
- WMT26 exploratory: seven previously inspected cESA language directions,
  18 coverage-selected model candidates with calibration-defined high-four
  targets and low-four development controls. The primary WMT24 test is
  genuinely new *method-outcome* data, whereas WMT26 is not.
- There are **20 seeded whole-document splits** and **100 independent
  draws per stochastic method/budget**. All results condition on a
  complete-case, jointly annotated pool; document and annotation
  coverage may not be representative of the whole benchmark.
- **Important uncounted setup cost:** The mean WMT24 split requires about
  **2,259 prior system-item labels per language** (10-system role
  calibration, four-system variance calibration and full evaluation-pool
  four-system development scores), in addition to the **100 incremental
  target labels** at budget 25. The analogous WMT26 mean is about
  **2,365 prior system-item labels**, excluding those 100 targets.
  Some target-system labels are explicitly required on calibration documents,
  so the method is not target-free in the strict *never-before-seen-model*
  sense. These costs can be amortized only if the labels already exist.
- Bootstrap intervals resample the nine or seven **language-level paired
  differences**, not individual trial draws; overlapping splits and some
  common source-language documents imply correlated observations.
  Intervals are exploratory and are **not** an independent study-level
  significance test.
- The top-four WMT24 systems (selected on role calibration only) average
  **8.7 ESA points** above development models on evaluation documents;
  their weakest-minus-strongest development mean is around **−1.2**, so
  they are not uniformly stronger. The corresponding WMT26 averages are
  **+22.7** and **+8.9 cESA points**, respectively.

## Audit and reproducibility

Independent auditing completed for **16/16 language directions**:

- **631,260** exact selection/weight replays, using calibration/development
  scores and frozen RNG seeds;
- **7,560** independently recomputed trial-zero metric vectors;
- **320** disjoint role/variance/evaluation document partitions;
- **48** independently checked macro cells and **336** paired comparisons;
- full input checksum, method code checksum, per-language means and
  complete-run signature validation.

See [audit_manifest.json](audit_manifest.json), [macro_summary.csv](macro_summary.csv),
[paired_language_comparisons.csv](paired_language_comparisons.csv),
and [language_summary.csv](language_summary.csv). Language-specific
summaries and role-quality gaps are under `languages/<cohort>/<direction>/`.
Raw official human judgments, source-derived matrices, large trial-ID
selections and exact split-ID manifests are intentionally not tracked.

Reproduce using the pinned official annotation files (see
[`build_wmt24_esa_matrices.py`](../../experiments/quality_shift/build_wmt24_esa_matrices.py)),
the frozen [protocol](../../experiments/quality_shift/WMT24_PAIRVAR_PROTOCOL.md),
[`run_wmt24_pairvar.py`](../../experiments/quality_shift/run_wmt24_pairvar.py)
and independent [`audit_wmt24_pairvar.py`](../../experiments/quality_shift/audit_wmt24_pairvar.py).

**Decision:** This experiment does not justify adopting Cal-pairvar-Neyman
as a convincing new ranking method. If pursuing another approach, formulate
and preregister a genuinely ranking-oriented decision rule that accounts
for its calibration label costs, and test it on another untouched panel.
