# Quality-shift multilingual extension

This extension tests whether the Czech-German ranking result generalizes to the
other 13 language pairs from the existing WMT25 study. The original pilot script
and its committed results are unchanged. This is a retrospective quality-group
experiment, not an evaluation of chronologically newer model generations.

## Fixed protocol

- Six selectors: Random, Difficulty-stratified, Hard-only, All-dev variance,
  Upper-dev variance, and Source-length-stratified.
- Twenty seeded whole-document partitions per language; 25% of documents,
  rounded upward, are reserved for calibration.
- Six targets are selected by calibration scores. Lower-six, Upper-six, and
  Random-six development groups exclude those targets.
- Target evaluation scores and translations are not selector inputs.
- One hundred trials for stochastic methods, one selection for deterministic
  methods, five strata, seed `20261106`.
- Pairwise ranking accuracy at 25 items is primary. Budgets 10, 50, and 100 are
  sensitivity analyses. MAE and signed bias diagnose score distortion.

The primary summary excludes Czech-German because that pilot was already used
to develop this extension. All 14 directions and the pilot alone are reported
separately. These directions were used in earlier project experiments; they are
not a newly collected independent dataset.

### Budget coverage rule

Whole-document splits can have very unequal item counts. Before running any new
selection outcomes, the runner inspects every scheduled partition. A
language-budget combination is eligible only if **all 20** seeded evaluation
partitions contain enough items. An ineligible combination is omitted in full,
with the reason recorded in `budget_coverage.csv`; no split is resampled or
selectively dropped. Every language supports the primary 25-item comparison.

### Aggregation and uncertainty

Average trials within splits, then splits within language pairs, then give
language pairs equal weight. Pair each method with Random within each language.
Report pairwise-accuracy differences in percentage points and MAE differences in
score units. Positive values in `paired_language_comparisons.csv` favor the
method. Keep undefined metrics missing and report finite trial/split counts.

The 10,000-resample language-pair bootstrap is exploratory. Language directions
share source documents and model families, so the intervals are not proof of
independent-dataset generalization. No p-values or multiplicity-adjusted claims
are made. No interval is reported for fewer than two valid languages.

Do not filter a split because its targets fail to remain stronger on evaluation
items. The quality-gap audit reports those outcomes. Changing development
groups changes system identity and family composition as well as average
quality; this does not identify a causal effect of the numerical quality gap.

## Reproduction

Install the dependencies from `experiments/quality_shift/requirements.txt`.
Obtain the real Git LFS contents of the official WMT25 human-evaluation JSONL,
not its small pointer file. The data used for this run came from
`wmt-conference/wmt25-general-mt` commit
`2fc0874d85278d8c7e0807951e54d8422e7957e4`, file
`data/wmt25-genmt-humeval.jsonl`, SHA-256
`a36ebce10c9ea26f71449933721de3a87e27f6a5b09d96b5daa026eaf1d197ab`.

From the project root:

```bash
python3 -m unittest discover -s experiments/quality_shift -p 'test_*.py' -v

python3 experiments/quality_shift/build_quality_shift_matrices.py \
  --data ../wmt25-general-mt/data/wmt25-genmt-humeval.jsonl \
  --out data/quality_shift_matrices \
  --source-revision 2fc0874d85278d8c7e0807951e54d8422e7957e4

python3 experiments/quality_shift/run_quality_shift_multilingual.py \
  --matrix-dir data/quality_shift_matrices \
  --out results/quality_shift_multilingual_reproduction
```

The output directory must be new or empty. Add `--resume` only to continue the
same input, code, and configuration; completed outputs are hash-verified before
reuse. Interrupted temporary language directories are preserved, not silently
overwritten. The matrix builder keeps the full human-score universe, preserves
source-file item order, averages non-null annotations, excludes `refA`, and
rejects incomplete matrices instead of silently filtering or imputing them.

## Audit and evidence boundaries

Each language saves its exact split manifest, trial selections, aggregate and
split-level results, quality-gap audits, and runtime versions. The independent
audit checks every saved selection and independently recalculates trial-zero
metrics for every split/regime/method/budget. All trial metrics are reaggregated
and compared with the saved tables. This is not an independent reimplementation
of all six selectors. Trial CSVs are retained locally but ignored by git because
they are large and regenerable; compact manifests, summaries, hashes, and audit
reports are suitable for committing.

The Czech-German full-default run must reproduce the original summary tables
and exact partition/system manifest before proceeding. The pilot script hash is
pinned by the wrapper to prevent accidental scientific changes.
