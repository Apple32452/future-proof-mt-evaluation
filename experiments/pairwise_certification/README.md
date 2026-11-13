# Pairwise MT certification: development validation and hindsight cost bound

**Status:** development study; not an independently validated improvement, and not a claim of a new statistical theorem.

This branch records the validated shared-prefix pairwise racing study from November 13, 2026. The complete code-and-results package was produced separately as `MT_Pairwise_Certification_Study_2026-11-13.zip` (SHA-256: `114d4c9a96349791187258022583703a3fd96e56cc706b8af27d9675a6f92afd`). To avoid overstating publication status, the full executable files must still be imported from that package; use `import_study.py` in this directory.

## Research question and estimand

For a fixed finite population of N evaluation segments and M systems with bounded human scores Y[i,s] in [0,1], certify signs of all differences between fixed-test-set means:

`Delta(a,b) = N^{-1} sum_i (Y[i,a] - Y[i,b])`.

A certificate is a statistically justified ordering. It is not a guarantee about new documents, annotator noise, or future systems. The experiment controls simultaneous sequential error at familywise alpha=0.05 under the documented sampling assumptions. It distinguishes 90% pair coverage, all-pairs coverage, and the leading-pair decision.

## Frozen study sequence

- **v1:** shared random segment order; adaptively select system scores; reuse acquired scores across eligible comparisons; sequentially valid pairwise inference.
- **v2:** change the *acquisition rule only* to target comparisons approaching certification; keep the inferential test unchanged.
- **Hindsight allocation diagnostic:** expose all development scores to an offline optimizer and bound the minimal final-prefix costs under the same certification machinery. This is an oracle diagnostic, not a deployable method or a statistical confidence interval.

The v1/v2 paired development replay used Czech-to-German human-score entries from an already examined WMT25 panel. Each published score cell is the mean of two historical ratings; this distinction must be retained when discussing annotation costs. The oracle used five prespecified exposed development orders. The primary target was certification of at least 171 of 190 system pairs.

## Main costs (mean score cells)

| Policy | v2 study: 90% coverage, 20 orders | Five-order oracle diagnostic |
|:--|--:|--:|
| Revised v2 racing | 4,336.1 | 4,323.0 |
| Random-pair allocation | 4,438.8 | 4,424.0 |
| Common-active random sampling | 4,550.4 | 4,551.8 |
| Offline hindsight optimum | Not evaluated | 4,114.2 |
| Numerical lower bound | Not evaluated | 4,113.2 |

The v2 development gains at 90% coverage are 2.31% relative to random-pair allocation and 4.71% relative to common-active. The numerical oracle result permits at most about 4.85% further mean savings compared with v2 *in those five runs under those exact constraints*. It is not a general impossibility statement.

The v1/v2 tests passed 63 automated tests in the final package, and the separate oracle-output auditor passed 22,122 checks. Those checks validate reproducibility and arithmetic for these runs, not future generalization.

## Interpretation and next experiment

Further adjustments to acquisition alone have limited headroom **under the frozen certification rule and sampled development case**. The next controlled experiment should put the same observed-score stream into two inference procedures, comparing the current betting certificate against a variance-adaptive sequentially valid alternative. Keep the desired familywise error guarantee, handle ties, compare cost-to-90%-coverage and complete-ranking costs, and only later evaluate acquisition jointly.

Do **not** present the partial-rank coverage metric as identification of the best system: 90% of pairs may be certified while a decisive leading-system pair is unresolved.

## Reproduction

The complete ZIP contains `METHOD.md`, `METHOD_V2.md`, `METHOD_ORACLE.md`, executable Python, 63 tests, protocols, summaries, audit records and raw regenerable trial traces. No private data or raw human-score matrix is required to be uploaded to GitHub.

1. Download the ZIP from the project conversation/Library.
2. Verify its SHA-256 against the hash above.
3. Use `python experiments/pairwise_certification/import_study.py /path/to/MT_Pairwise_Certification_Study_2026-11-13.zip` from the repository root to stage allowed study files.
4. Run the imported tests and audits with the pinned data source as documented inside the package.
5. Inspect and commit the generated changes to this branch.

This GitHub branch contains the research record and importer now; until the ZIP's validated implementation is imported and committed, it is **not** a complete executable release.
