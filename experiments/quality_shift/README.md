# Stronger Unseen MT Systems: Quality-Shift Pilot

## Research question

Can evaluation items selected using lower-quality MT systems
reliably rank stronger unseen MT systems?

This is a new class-project extension of the earlier
difficulty-stratified evaluation study.

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

## Multilingual extension

The fixed pilot can now be run across the 14-language WMT25 roster. The 13
non-pilot directions form the primary new-language summary; Czech-German is
retained as an exact reproduction check. See [MULTILINGUAL.md](MULTILINGUAL.md)
for the protocol, budget-coverage rule, independent output audit, and commands.
