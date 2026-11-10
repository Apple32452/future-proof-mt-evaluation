# WMT26: Quality-calibrated stronger targets and new sampling methods

**Date:** November 10, 2026. **Status:** exploratory, retrospective.

This experiment is separate from the WMT25 pilot and the **identity-fixed**
WMT26 newer-model check. Within each split, the six top- and six bottom-
quality systems are assigned by **human cESA scores on held-out calibration
documents only**, then tested on disjoint evaluation documents. Every method
uses the same split-specific target/development groups. We report the success
and failures of both new and previously investigated methods.

## Main result: primary four language directions

Equal-weight four-direction macro average, budget **25 additional target
sentence judgments**, 20 document-disjoint calibration/evaluation splits:

| Sentence-selection method | Target ranking pairwise accuracy | MAE (cESA points) | Signed bias (cESA points) |
|:--|--:|--:|--:|
| Random | 77.95% | 2.828 | -0.048 |
| **Dev-cluster-proportional (new)** | **78.40%** | **2.356** | -0.020 |
| Dev-feature-mean-balance (new) | 72.99% | 3.085 | -1.108 |
| Difficulty-stratified | 78.23% | 2.641 | -0.033 |
| Source-length-stratified | 78.03% | 2.693 | +0.007 |
| Hard-only | 81.20% | 17.515 | -17.245 |
| All-dev variance | 79.27% | 8.317 | -0.378 |
| Upper-dev variance | 76.39% | 6.877 | -1.504 |

Compared with random, **cluster-proportional sampling** adds about **0.45
percentage points** to target pairwise ranking accuracy while reducing
MAE by **0.473 cESA points**, or **16.7%**. Across the four language
directions, the paired bootstrap for the pairwise-accuracy difference gives
an exploratory 95% interval of **[-0.00385, 0.01191]** (includes zero).
For the MAE *difference* (method minus random), the exploratory interval
is **[-0.989, -0.198]**. Four overlapping language directions do not
constitute independent confirmatory replication.

Hard-only sampling has **higher primary ranking accuracy** but extremely
poor absolute-score estimation, with a strong negative bias. The new
mean-balanced coreset performs worse than random on both primary accuracy
and MAE. Those are genuine negative findings and should remain visible.

## Seven-direction external check (including supplementary directions)

| Sentence-selection method | Ranking accuracy | MAE |
|:--|--:|--:|
| Random | 76.60% | 3.298 |
| **Dev-cluster-proportional (new)** | **76.89%** | **2.745** |
| Difficulty-stratified | 76.71% | 3.010 |
| Source-length-stratified | 76.54% | 3.117 |
| Dev-feature-mean-balance (new) | 72.21% | 3.388 |
| Hard-only | 75.69% | 30.720 |
| All-dev variance | 75.82% | 8.972 |
| Upper-dev variance | 73.55% | 7.959 |

The cluster method decreases MAE by about **16.8%** across seven
directions, with a small **+0.29 percentage-point** ranking-accuracy
difference. It reduces MAE *in every evaluated direction*, but accuracy
improvements vary by language. The hard-only ranking advantage in the
primary four directions does **not** survive the supplementary directions.

### Cluster-proportional versus random, by direction

| Direction | Random accuracy | Cluster accuracy | Random MAE | Cluster MAE |
|:--|--:|--:|--:|--:|
| English → Egyptian Arabic | 77.23% | 78.64% | 2.785 | 1.540 |
| English → Russian | 76.63% | 76.90% | 2.960 | 2.739 |
| English → Japanese | 77.61% | 76.77% | 2.642 | 2.459 |
| English → Ukrainian | 80.31% | 81.29% | 2.926 | 2.685 |
| English → Indonesian | 70.64% | 70.77% | 3.654 | 3.453 |
| English → Thai | 83.53% | 83.44% | 3.350 | 3.215 |
| Chinese → Japanese | 70.25% | 70.45% | 4.771 | 3.125 |

## Did targets really have higher quality?

The calibration-defined targets are the **top six among the 18 eligible
systems**, not automatically the best systems in the full 46-system WMT26
roster. The realized held-out **mean target-minus-development gap** was
positive in **140 of 140** language/split evaluations. We never filtered
or changed the study based on these realized held-out gaps.

| Direction | Mean held-out target-minus-development gap | Strictly all targets above all devs (splits/20) |
|:--|--:|--:|
| English → Egyptian Arabic | +10.28 cESA | 10 |
| English → Russian | +11.00 cESA | 6 |
| English → Japanese | +13.84 cESA | 9 |
| English → Ukrainian | +14.12 cESA | 4 |
| English → Indonesian | +23.04 cESA | 20 |
| English → Thai | +46.80 cESA | 20 |
| Chinese → Japanese | +22.18 cESA | 20 |

This is strong evidence of a **mean quality contrast within the chosen
complete-case panels**, but *not* a guarantee of every-target-better-than-
every-development separation for the primary four directions.

## Essential qualification: what does the annotation budget count?

The 25-item budget counts only **additional held-out target-system
judgments per evaluation trial**, *conditional on*: (i) the availability of
human scores for **all 18 systems** on calibration documents, and (ii)
human scores for **all six development systems** over the entire eligible
evaluation pool. In a fresh study these inputs would also incur annotation
costs. Consequently, **we do not claim a 16.8% net human-label saving**:
16.8% is a *reduction in score estimation MAE at the same incremental
target-label budget*. WMT26 dynamic human-annotation coverage, the
preselected complete-case cohort, retrospective re-use of judged items,
and different WMT25/WMT26 protocols further limit generalization.

## Reproduction and integrity

All seven independent audits passed: **218,160 saved selections** were
replayed, and **4,320 trial-zero metric vectors** were recalculated using
an independent scalar/SciPy metric oracle. The full automated test suite
passed **36 tests**. The auditor shares the production selector functions
for replay but independently checks saved metrics, aggregate tables,
calibration assignments, quality gaps, and cryptographic file hashes.

See [`WMT26_QUALITY_CALIBRATED.md`](../../experiments/quality_shift/WMT26_QUALITY_CALIBRATED.md)
for method definitions, source-provenance hashes, complete reproduction
commands, selection-seed conventions, and the independent replay auditor.
Per-split method means, per-direction summaries, quality-gap audit files,
signed paired-method comparisons, and cryptographic completion/audit
records are preserved next to this note. Large item-level trial logs,
source matrices and document-ID manifests are excluded from GitHub but can
be regenerated from the pinned public source.

**Course-project assessment:** this does meaningfully extend the prior
NAACL study to a calibrated higher-quality comparison *and* two alternative
sampling methods. It supplies a usable tradeoff (ranking preservation
versus human-score bias) but not a confirmatory new-method superiority
claim or a full end-to-end annotation-cost result.
