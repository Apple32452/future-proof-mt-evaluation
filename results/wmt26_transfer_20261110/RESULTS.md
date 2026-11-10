# WMT26 newer-model external check (November 10, 2026)

## Scope and outcome

This separate retrospective experiment evaluates **six frozen WMT25-era
sentence-selection heuristics** on a WMT26 human-scored panel of six newer
named systems. It is **not** a demonstration that the new targets are
substantially higher quality than the development systems: the realized
quality-gap checks frequently show the opposite. This is a newer-model,
new-judgment-protocol external setting, not a successful strict-quality-shift
replication.

- Fixed target identities: GPT 5.5, Gemini 3.1 Pro, DeepSeek V4 Pro,
  Gemma 4 - 31B, Mistral Medium 3.5, Qwen 3.6 - 27b.
- Primary: English-to-Egyptian-Arabic, Russian, Japanese, Ukrainian;
  supplementary: English-to-Indonesian, Thai, Chinese-to-Japanese.
- Per direction: twenty seeded 25%/75% document-disjoint
  calibration/evaluation splits; three calibrated development regimes;
  six sampling methods; budgets supported by all splits; 100 stochastic
  draws or one deterministic evaluation per method/split/budget.
- Main estimand: average pairwise target ranking accuracy at 25
  human-scored evaluation items. Each WMT26 language direction has equal
  weight in the four-language primary macro average.
- Scores are WMT26 contrastive Error Span Annotation (cESA), **not** the
  WMT25 human score scale. MAE differences across years are not comparable.

## Primary four-direction results (budget 25, lower-six development)

| Selection method | Target pairwise accuracy | MAE (cESA points) | Signed bias (cESA points) |
| :-- | --: | --: | --: |
| Random | 88.19% | 3.652 | +0.058 |
| Difficulty-stratified | **88.56%** | **3.402** | -0.068 |
| Source-length-stratified | 88.51% | 3.548 | -0.033 |
| Hard-only | 84.29% | 21.157 | -20.978 |
| All-development variance | 85.92% | 12.298 | -8.223 |
| Upper-development variance | 85.17% | 9.968 | -5.820 |

Difficulty stratification exceeds random by **0.37 percentage points**
of pairwise ranking accuracy (+0.00365 on the [0,1] scale), and reduces
MAE by **0.250 cESA points** (about 6.8%). The rank gain is small. A paired
bootstrap over only four language directions gives an *exploratory*
95% interval of +0.00029 to +0.00671 on the pairwise-accuracy scale;
this is not evidence of formal confirmatory significance.

### Per-language accuracy (same primary setup)

| Language direction | Random | Difficulty-stratified | Change (percentage points) |
| :-- | --: | --: | --: |
| English → Egyptian Arabic | 89.43% | 89.84% | +0.41 |
| English → Russian | 90.42% | 90.33% | -0.10 |
| English → Japanese | 89.79% | 90.18% | +0.38 |
| English → Ukrainian | 83.14% | 83.90% | +0.77 |

## Stronger-model assumption fails the audit

The preselected modern target systems were **not** uniformly better than
the observed development panels. Across each language's 20 splits × three
development regimes (60 comparisons), the target panel's average evaluation
quality was higher than the selected development panel's mean in:

| Direction | Target mean higher, out of 60 | Mean target-minus-development quality gap |
| :-- | --: | --: |
| English → Egyptian Arabic | 9 | -3.38 cESA |
| English → Russian | 0 | -6.48 cESA |
| English → Japanese | 28 | -2.02 cESA |
| English → Ukrainian | 33 | +0.18 cESA |

Across none of these 240 panel/split comparisons was the *lowest-quality*
target above the *highest-quality* development system. This disqualifies an
unqualified conclusion that the six targets test substantially stronger models.
Changing the target roster after seeing this result would make a different,
exploratory experiment, which should be documented separately.

## Evidence and reproduction limits

Results use only WMT26 segments with valid human judgments for *all* six
targets and the twelve coverage-selected development systems; WMT26's
dynamic evaluation means missingness depends on system identity/performance.
Do not interpret reported target means as full-WMT26 system quality or
extrapolate to unjudged samples. Full trial-level selection logs (~160 MB),
raw human annotations (~435 MB) and derived score matrices are excluded
from git; their provenance, hashed manifests and reproduction instructions
are in [`WMT26_EXTERNAL.md`](../../experiments/quality_shift/WMT26_EXTERNAL.md).

The retained complete-case fraction is **55.6%** for English→Egyptian
Arabic (384 of 691 human-scored segments), **37.6%** for English→Russian
(260 of 691), **46.3%** for English→Japanese (320 of 691) and just
**22.6%** for English→Ukrainian (156 of 691). This is a meaningful
restriction on what the primary results generalize to.

The official source is <https://github.com/wmt-conference/wmt26-general-mt>
at commit `63aa5d24311a7e7e3e8173528f3771c7e8b4be78`; the frozen final
annotation asset SHA-256 is
`6023795cc592864a9d67ed6daaaada283041d0a8c87c4bfb4c3e5ae276c73b28`.
All trial and split summaries, direction-level summaries, source-coverage
audits, manifests and the independent replay-audit report are located
alongside this file.

## Implications for the class project

This is one meaningful **new dataset, model generation and human-judgment
setting** beyond the NAACL submission, but its *ranking* improvement is
modest and it does **not yet** fulfill the professor's suggested test of
genuinely much-higher-quality target systems. A separately declared study
with strictly stronger quality-calibrated targets and an alternative
sampling method is the next logical step. It should be frozen before
inspecting its held-out ranking outcomes.
