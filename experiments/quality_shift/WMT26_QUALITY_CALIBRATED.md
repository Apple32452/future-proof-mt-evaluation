# WMT26 quality-calibrated targets and alternative selection methods

This is a **new exploratory follow-up**, deliberately kept separate from the
completed WMT25 pilot and the six *identity-fixed* WMT26 external evaluation.
The identity-fixed experiment found that newer models were not always better
than its development systems. We therefore ask a **different question**:

> When a calibration set identifies six stronger and six weaker translation
> systems, can we estimate the *relative ranking of the stronger six* using
> fewer held-out human judgments? Do new multivariate development-score
> samplers outperform equal-budget random sampling?

**This experiment was designed after inspecting the WMT26 coverage and the
previous pilot's quality gaps. It must be treated as exploratory and cannot be
described as a preregistered/confirmatory external validation.**

## Data and separation of roles

We reuse the WMT26 complete-case human-annotation matrices built in
[`build_wmt26_transfer_matrices.py`](build_wmt26_transfer_matrices.py).
Their original six named systems and twelve coverage-selected development
systems form an 18-model *candidate panel*; no models outside this panel can
be selected. The original panel and item core were constructed under WMT26
dynamic annotation coverage, so the selected population is not representative
of every WMT26 segment. Official source and SHA-256 details are in
[`WMT26_EXTERNAL.md`](WMT26_EXTERNAL.md).

For each direction and each of **20** document-disjoint random splits:

1. Reserve 25% of **whole documents** as calibration; the rest are evaluation.
2. Average each of the 18 systems' *human* cESA scores on calibration docs.
3. Define the six highest-scoring systems as **targets**; define the six
   lowest-scoring systems as **development**. Exclude the middle six. This
   assignment is fixed for that split, based on calibration scores only.
4. On evaluation documents, make the entire development-system score vectors
   and source lengths available to the sentence selector. Keep all six target
   systems' evaluation judgments hidden until after sentences are chosen.
5. Compare each method's estimated six-target ranking to the ranking computed
   using **all target evaluation scores on the retained item core**.

We **do not** filter splits, directions or results by the observed evaluation
quality advantage. We report both the calibration quality gap and the held-out
quality gap to check whether "stronger targets" was achieved. The target
identities may differ across splits, but all methods receive the **same
target/development assignment and evaluation items in each split**.

### New samplers

In addition to the six frozen WMT25 samplers, this study introduces:

- **Dev-feature-mean-balance (deterministic coreset):** represent each
  evaluation sentence by its six-dimensional development score vector,
  standardized using **calibration-only** development means and standard
  deviations. Greedily choose the next unused sentence to minimize the
  squared distance between the selected vectors' mean and the full evaluation
  pool's development-vector mean. Estimate targets by the unweighted mean
  of their chosen human scores. No target evaluation scores enter selection.
- **Dev-cluster-proportional:** cluster the standardized six-dimensional
  development vectors with seeded k-means (five clusters). Randomly sample
  sentences from *every nonempty cluster*, approximately proportional to
  its size. A cluster-population-weighted estimate gives each selected
  target judgment weight `n_cluster / (n_total * n_sampled_cluster)`;
  this corrects for deliberate stratification and totals exactly one.

Random, difficulty-stratified, source-length-stratified, and
cluster-proportional each have 100 independent deterministic-seed trials
per split/budget; deterministic methods have one. The budgets are 10, 25,
50 and 100 additional evaluation-item judgments, restricted to those
available across all scheduled splits. **Primary metric:** pairwise ranking
accuracy at budget 25, macro-averaged equally across the four overlapping
WMT25/WMT26 directions (English → Egyptian Arabic, Russian, Japanese,
Ukrainian). The other three directions (English → Indonesian, Thai;
Chinese → Japanese) and MAE, signed bias, top-one accuracy and other budgets
are supplementary.

## Explicit limits on interpretation

1. **Historical development and calibration labels are not free.** The 25%
   calibration-document subset requires human scores for *all 18 models*,
   and the evaluation pool requires human scores for *all six development
   models* **on every eligible item** before selecting target labels.
   Reported budgets count only **additional target-item judgments**, conditional
   on both sets of already-available labels. The study does *not* demonstrate
   a net reduction in total annotation cost including those inputs.
2. **Strong quality is relative.** Targets are the top six among 18
   coverage-selected systems, **not necessarily the six best models in all
   WMT26**. The realized evaluation gaps are descriptive evidence, not
   grounds for selecting or dropping a split.
3. **Complete-case coverage is nonrandom.** We cannot generalize to the
   full WMT26 test set or unjudged sentences without additional assumptions.
4. **Retrospective evidence only.** All labels already exist; no new human
   judgments are collected. This is not real-time human-in-the-loop testing.
5. **Small multilingual sample.** A bootstrap over four language directions
   is exploratory, not formal confirmation. We report losses, regressions,
   and null results as well as successes.
6. **Different score scales.** WMT26 cESA units cannot be numerically
   equated with WMT25 human ratings, even when language pairs overlap.

## Running the experiment

From the GitHub repository root, after producing the WMT26 score matrices
by following the prior [`WMT26_EXTERNAL.md`](WMT26_EXTERNAL.md) instructions:

```bash
python3 -m unittest discover -s experiments/quality_shift -p 'test_*.py' -v

python3 -u experiments/quality_shift/run_wmt26_quality_calibrated.py \
  --matrix-dir data/wmt26_transfer_matrices \
  --out results/wmt26_quality_calibrated_20261110

python3 -u experiments/quality_shift/audit_wmt26_quality_calibrated.py \
  --matrix-dir data/wmt26_transfer_matrices \
  --results results/wmt26_quality_calibrated_20261110
```

To resume interrupted complete language runs, pass **identical arguments**
plus `--resume`. Existing hashes and protocol signatures are verified.
The `.gitignore` omits large trial-level selection logs and raw human labels;
it also omits the bulky per-split sentence-ID manifests. The committed
per-language summaries, explicit quality-gap audits, compact provenance
manifests, and independent audit reports are sufficient to inspect outcomes.
Full replays require regenerating the original matrices, split manifests,
and ignored trials locally.

Once results finish, their analysis is in
[`results/wmt26_quality_calibrated_20261110/RESULTS.md`](../../results/wmt26_quality_calibrated_20261110/RESULTS.md).
