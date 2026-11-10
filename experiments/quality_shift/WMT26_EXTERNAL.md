# WMT26: externally evaluating newer high-capacity MT systems

This is a **separate** 2026 evaluation, not a modification of the original
WMT25 pilot or its 14-language retrospective extension. It tests whether
sentence-selection heuristics selected during the WMT25 project still work for
named newer MT systems evaluated in WMT26. **No 2026 outcomes were used to
revise the six frozen selectors** (Random, difficulty-stratified, hard-only,
all-development variance, upper-development variance and source-length
stratification). WMT26 cESA is a different human evaluation scale; compare
methods *within* WMT26, not WMT25 and WMT26 score values directly.

## Official sources

- Official WMT26 GenMT repository: https://github.com/wmt-conference/wmt26-general-mt
- Frozen repository commit: `63aa5d24311a7e7e3e8173528f3771c7e8b4be78`.
- Final human annotation asset (435,274,392 bytes):
  https://github.com/wmt-conference/wmt26-general-mt/releases/download/humeval/annotations.json
- Final asset SHA-256:
  `6023795cc592864a9d67ed6daaaada283041d0a8c87c4bfb4c3e5ae276c73b28`.
- Original task paper: https://aclanthology.org/2026.wmt-main.48/

WMT26 evaluated 46 systems on 23 language directions and used dynamic
contrastive Error Span Annotation (cESA). This project's adapter invokes the
official WMT26 `humeval/07-build_synthetic_references.py` routines to handle
user RESET actions, tutorials/attention checks, annotations for models not
shown on the screen, and low-reliability annotators. Repeated cESA scores for
each displayed (direction, segment, system) are averaged. Only a segment with
valid observations for *every* model in its panel is retained. The data must
never be back-filled with model predictions or automatic metrics.

## Fixed external protocol

The **six fixed target model identities** are GPT 5.5, Gemini 3.1 Pro,
DeepSeek V4 Pro, Gemma 4 - 31B, Mistral Medium 3.5 and Qwen 3.6 - 27b.
These are not picked by their realized evaluation scores; do not assume they
beat every development model. The quality-gap audit records realized gaps.

For each WMT26 direction, select twelve *other* models by maximum overlap of
**observed-score availability masks** with all six targets (tie by model name).
This is chosen before method outcomes, but WMT26's dynamically allocated
human evaluation is **not missing at random**: the resulting cohort is a
coverage-selected subset, not the entire WMT26 test set. Both the item
retention fraction and development-model identities are recorded in the
matrix manifest. Human references, synthetic references and PseudoRef are
ineligible as development systems.

The experiment uses 20 seeded whole-document splits: 25% of documents for
calibration, the rest for evaluation, with no document overlap. Development
groups are Lower-six, Upper-six and Random-six according to **development
calibration scores only**. The six frozen target identities never change.
Selection uses only development evaluation scores (or source length), never
target evaluation judgments. Stochastic methods use 100 trials; others use
one. Budgets 10, 25, 50 and 100 are included only where all 20 scheduled
evaluation splits have at least that many items. Nothing is dropped based on
the observed target quality gap or on method performance.

Primary outcome: **pairwise ranking accuracy at budget 25** among fixed
targets in the four WMT25-overlapping directions with substantial WMT26 human
coverage: English to Egyptian Arabic, Russian, Japanese and Ukrainian.
Supplementary directions: English to Indonesian, English to Thai, and Chinese
to Japanese. MAE, signed bias, rank correlations, top-one and other budgets
are diagnostic/sensitivity analyses. Average trials within splits, then
splits within each language, then language directions equally. Paired
bootstrap intervals across language directions are exploratory only.

## Reproduction from the repository root

The source data are downloaded into sibling directories **outside this git
repository**; raw annotations and derived matrices are not committed.

```bash
git clone https://github.com/wmt-conference/wmt26-general-mt.git ../wmt26-general-mt
git -C ../wmt26-general-mt checkout 63aa5d24311a7e7e3e8173528f3771c7e8b4be78

mkdir -p ../wmt26-data
curl -fL --retry 2 \
  https://github.com/wmt-conference/wmt26-general-mt/releases/download/humeval/annotations.json \
  -o ../wmt26-data/annotations.json
shasum -a 256 ../wmt26-data/annotations.json

python3 -m pip install -r experiments/quality_shift/requirements.txt 'sacrebleu>=2.4'
python3 -m unittest discover -s experiments/quality_shift -p 'test_*.py' -v

python3 -u experiments/quality_shift/build_wmt26_transfer_matrices.py \
  --annotations ../wmt26-data/annotations.json \
  --wmt26-repo ../wmt26-general-mt \
  --out data/wmt26_transfer_matrices

python3 -u experiments/quality_shift/run_wmt26_external.py \
  --matrix-dir data/wmt26_transfer_matrices \
  --out results/wmt26_transfer_20261110

python3 -u experiments/quality_shift/audit_wmt26_external.py \
  --matrix-dir data/wmt26_transfer_matrices \
  --results results/wmt26_transfer_20261110
```

For a resumed run, use the **identical** command plus `--resume` and the same
output directory. Finished language records and their file hashes are
rechecked; changed inputs, code or protocol fail closed. The large per-trial
selection logs are retained locally but git-ignored. The audit replays every
recorded selection; independently recalculates a trial-zero metric vector for
each split, regime, method and budget; and checks every saved trial through
independent split and language reaggregation. Consequently, the committed
compact summaries and audit records are useful for reporting, but a full
reproduction requires regenerating the ignored per-trial records from the
verified upstream annotations and matrices.

Results, including primary per-direction performance, are recorded in
[`results/wmt26_transfer_20261110/RESULTS.md`](../../results/wmt26_transfer_20261110/RESULTS.md).

## Evidence and limitations

1. The evaluated population is the *jointly scored cESA segment core* for
   each frozen target/development panel. Dynamic sampling favored strong
   systems, and coverage may correlate with quality. The retained fraction
   is published; estimates must not be called full-WMT26 performance.
2. Modern model identities are predetermined. Positive target-versus-dev
   quality gaps are an empirical outcome, not a precondition for inclusion.
3. Different human judgment protocols, language samples, domains, and
   composition mean raw score/MAE values from 2025 and 2026 are **not**
   directly comparable. These are within-year sampling comparisons.
4. All WMT26 judgments are retrospective; the experiment is an external
   *year/dataset and model-generation holdout*, not a prospective deployed
   evaluation in which fresh labels arrive in real time.
5. WMT25 and WMT26 share language directions/system families and WMT26
   directions can share source documents. Language bootstrap intervals are
   exploratory, not proof of independent replication or formal control of
   multiple testing.
6. We did not add human judgments, run paid models or claim that the sampled
   subset improves absolute MT quality. No artificial score imputation.

The correct research question is: **On the same genuinely newer WMT26
human-scored segment panels, do frozen WMT25-era sampling rules preserve the
ranking of six named new model systems better than equal-budget random
sampling?**
