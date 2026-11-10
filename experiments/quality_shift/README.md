# Stronger Unseen MT Systems: Quality-Shift Pilot

## Research question

Can evaluation items selected using lower-quality MT systems
reliably rank stronger unseen MT systems?

This is a new class-project extension of the earlier
difficulty-stratified evaluation study.

For the later **calibration-driven pairwise-variance** sampling experiment,
see its [frozen protocol](WMT24_PAIRVAR_PROTOCOL.md) and
[WMT24/WMT26 audited results](../../results/wmt24_pairvar_20261110/RESULTS.md).
The primary WMT24 fresh-data test did not improve ranking accuracy over
random; the WMT26 check remained exploratory and all human-label
calibration costs are reported separately.

## Dataset

- WMT25 Czech-German human evaluation
- 231 items, 20 MT systems, 128 source documents
- 20 document-separated calibration/evaluation splits

## Experimental methods

1. Random sampling
2. Difficulty-stratified sampling
3. Hard-only selection
4. All-development-system variance
5. Upper-development-system variance
6. Source-length stratification

Development conditions: Lower-six, Random-six, Upper-six.

Budgets: 10, 25, 50, 100 items.
Seed: 20261106.
Stochastic trials: 100 per split and budget.

## Preliminary results

At 25 items, pairwise ranking accuracy:

| Method | Lower-six | Random-six | Upper-six |
|---|---:|---:|---:|
| Random | 75.47% | 75.47% | 75.47% |
| Difficulty-stratified | 75.33% | 75.49% | 76.05% |
| Hard-only | 71.00% | 81.67% | 86.17% |

Hard-only performs substantially differently depending on
the development-system population. Its ranking gains also
come with large absolute-score estimation bias.

Results are descriptive: this is one language pair with
20 overlapping splits, not independent replications.

## Reproduction

Install dependencies:

    pip install -r experiments/quality_shift/requirements.txt

Run:

    python experiments/quality_shift/run_quality_shift_pilot.py \
      --data results/score_matrices/cs-de_DE_human_scores.csv \
      --out results/quality_shift_reproduction \
      --splits 20 --trials 100 \
      --budgets 10 25 50 100 --seed 20261106

The saved result directory contains aggregate and split-level
results, quality-gap audits, and the run configuration.

Large trial-level outputs are not committed by default.

## WMT26 ranking-aware hybrid follow-up

The [frozen WMT26 ranking protocol](WMT26_RANK_HYBRID_PROTOCOL.md) compares
random, hard-only, variance, difficulty-stratified, development-cluster,
and **rank-aware hybrid** sampling. The last method oversamples items that
are hard or disagreement-heavy for lower-quality development systems while
reweighting strata to avoid hard-only mean-score bias. The twelve additional
WMT26 directions are its primary method-outcome holdout at budget 10;
seven previously inspected directions are reported separately. Scores on
high-quality targets are never used for item selection.

## Multilingual extension

The fixed pilot can now be run across the 14-language WMT25 roster. The 13
non-pilot directions form the primary new-language summary; Czech-German is
retained as an exact reproduction check. See [MULTILINGUAL.md](MULTILINGUAL.md)
for the protocol, budget-coverage rule, independent output audit, and commands.
