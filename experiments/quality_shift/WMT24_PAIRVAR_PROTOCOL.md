# Calibration-driven pairwise-variance sampling: frozen experiment protocol

**Protocol locked before computing ranking outcomes on the nine WMT24 ESA
directions.** Only input coverage, document conflicts and duplicate-annotation
checks were inspected before locking. This is a retrospective experiment on
public, previously collected human judgments, not a prospective annotation
campaign. The previously examined WMT26 directions are exploratory.

## Hypothesis

When the objective is **ranking strong translation systems**, the sampler
should allocate labels toward strata with large *variance of differences
between strong systems*, not simply toward difficult sentences.

## Exact method: Cal-pairvar-Neyman

For each of 20 seeded document-disjoint splits:

1. Assign the first 25% of documents to a calibration pool, the rest to
   evaluation. Split the calibration documents in two **disjoint halves**:
   role calibration and variance calibration. All item/score operations
   subsequently respect those three disjoint document sets.
2. Select a pool of **ten machine translation systems** based solely on
   human-annotation availability (never quality values). On **role
   calibration** documents only, select the four highest-mean systems as
   unseen evaluation *targets*, four lowest-mean as *development*, and
   discard the two middle systems.
3. Use the four development systems' human judgments across the evaluation
   pool to form **three equal-count strata by average development-system
   difficulty** (negative mean score). Map variance-calibration items into
   these strata using the evaluation development-score tertile boundaries.
4. For each of the six target-system pairs `(a,b)`, estimate its difference
   variance from **variance-calibration** items in each stratum. Shrink each
   estimate toward the corresponding global variance with five pseudo-
   observations. All model pairs are weighted by the inverse square of
   `(5 + absolute role-calibration mean score gap)`; normalize weights to
   one. This emphasizes close pairs, using **role-calibration data only**.
5. Let `s_h` be the square root of the pair-weighted, shrunk difference
   variance in stratum `h`. Allocate an item budget `B` with a bounded,
   deterministic integer approximation to Neyman allocation
   `n_h proportional to N_h * s_h`, where `N_h` is the full evaluation
   stratum size, with **at least one item per nonempty stratum**, sampling
   without replacement. Use proportional allocation if all variances vanish.
6. Sample within each stratum uniformly. Estimate each target's full
   evaluation mean as a **population-weighted** average of stratum means,
   with per-item weights `N_h/(N_total*n_h)`. This estimator is unbiased
   conditional on the fixed strata and allocated counts. Rank targets
   using these weighted estimated means.

No evaluation **target** score appears in the selector, variance fit,
allocation or weights. Target evaluation labels are revealed only after
the chosen sentence IDs are finalized.

## Comparator methods

- Random item sampling (uniform mean).
- Difficulty-stratified (five equal-population difficulty strata).
- Development-cluster-proportional (five development-feature clusters,
  proportional allocation and population weights).
- Hard-only (deterministic high-difficulty diagnostic; unweighted and biased).
- **Cal-pairvar-proportional**: the *same three strata as the new method*,
  but sample proportional to population size; isolates whether learned
  pairwise variances improve allocation.
- **Cal-pairvar-Neyman**: the new calibrated variance allocation.

## Predeclared datasets and outcomes

**Primary new-data test:** All nine WMT24 official ESA directions where
ten non-reference systems have a complete jointly scored item panel:
`ces-ukr`, `eng-ces`, `eng-hin`, `eng-isl`, `eng-jpn`, `eng-rus`,
`eng-spa`, `eng-ukr`, `eng-zho`. Source is the three official, corrected
ESA wave files at `wmt-conference/wmt24-news-systems` commit
`fabe1f81d65449b3e3f9f8d37da280fa6cf3650a`, with file checksums
recorded in the built matrix manifest. Exclude `refA`, tutorial rows,
non-test documents and non-TGT rows; average repeated scores per
annotator-item-system first, then average across annotators. Strip only
the official annotation suffixes `#dup` and `#incomplete` when grouping
whole source documents. Source text is not needed by any compared sampler.

**Primary budget:** 25 sentences **per each of four target systems**
(100 target system--sentence judgments, excluding calibration and full-pool
development judgments). Secondary budgets 10/50/100 where supported.

**Secondary exploratory transfer:** Seven previously inspected WMT26
cESA directions (`eng_Latn---arz_Arab`, `rus_Cyrl`, `jpn_Jpan`,
`ukr_Cyrl`, `ind_Latn`, `tha_Thai` and `zho_Hans---jpn_Jpan`), reusing
the fixed 18-system coverage-selected panel but forming **new four-target,
four-development calibration-defined roles**. Compare budget 25 across
these seven, and other budgets only as supplementary. Do **not** combine
raw ESA 2024 and cESA 2026 score levels or mean absolute errors.

20 document-level splits, 100 stochastic trials per method/budget,
deterministic Hard-only once. Seed `20261112`. No tuning after WMT24
rankings are examined. Equal language weights, with paired language-level
bootstrap intervals for exploratory uncertainty; overlapping splits and
related language directions are **not** independent human studies.

**Primary metric:** pairwise ranking accuracy among the four selected
targets, averaged across the nine WMT24 languages at budget 25.
Secondary metrics: pairwise accuracy on >1-score-point pairs, top-one
identification, Kendall/Spearman, mean absolute score error, signed bias.
The prespecified practical success reference is **+2 percentage points
pairwise accuracy vs Random**, with explicit comparison to both
proportional and stronger previously developed baselines.

## Critical cost and data limitations

- Human scores for **all ten models on role-calibration documents** and
  **target pairs on variance-calibration documents**, plus full-pool
  development-system judgments, are assumed preexisting. Target-item
  budgets omit these nontrivial costs: *no claim of end-to-end human
  annotation savings* follows from the reported budgets.
- WMT24 ESA and WMT26 cESA are different human protocols and cannot
  be compared on absolute score scales. WMT24 role-selected top four
  are not necessarily modern state-of-the-art systems; WMT26 covers
  newer systems separately.
- Annotation coverage and complete-case inclusion can depend on system;
  the estimand is the jointly judged pool, not all benchmark segments.
- Calibration variance estimation on limited documents can be unstable,
  and these strata may not predict variance of strong-system differences.
  Report a failed hypothesis honestly if rankings do not improve.

## Reproduction

```bash
python -m unittest discover -s experiments/quality_shift -p 'test_*.py' -v
python -u experiments/quality_shift/build_wmt24_esa_matrices.py \
  --source-dir data/wmt24_esa_source_data --out data/wmt24_esa_matrices
python -u experiments/quality_shift/run_wmt24_pairvar.py \
  --wmt24-matrix-dir data/wmt24_esa_matrices \
  --wmt26-matrix-dir data/wmt26_transfer_matrices \
  --out results/wmt24_pairvar_20261110
python -u experiments/quality_shift/audit_wmt24_pairvar.py \
  --wmt24-matrix-dir data/wmt24_esa_matrices \
  --wmt26-matrix-dir data/wmt26_transfer_matrices \
  --results results/wmt24_pairvar_20261110
```

Raw source annotations, per-trial selected item traces and full split-ID
manifests remain uncommitted. Checksums, compact aggregated results,
code and independent auditor records are committed for reproduction.
