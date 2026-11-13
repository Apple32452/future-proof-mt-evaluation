# Frozen v1 results: a valid-inference prototype, not a demonstrated efficiency win

These results concern only `shared_prefix_pairwise_racing_v1`, its frozen
`protocol.json`, and `validation_v1/`. They do not describe any later scheduler
revision. The protocol was fixed before this validation run, but this is a
development study, not a preregistered independent replication.

**Decision: the predefined efficiency gate was not passed.** No tested scenario
achieved at least 10% lower cost to 90% pair resolution than both `random_pair`
and `common_active`. Small gains on the previously exposed MT matrix do not
establish a general advantage, novelty, or an ACL-ready contribution. The
near-tie setting exposes a substantial mismatch between v1's acquisition
heuristic and the partial-resolution objective.

## Study and cost definitions

- Six synthetic scenarios used 100 repetitions each. Each repetition generated
  a new fixed matrix of 120 items by 6 systems, with 15 pairwise mean differences
  and a full cost of 720 cells.
- The real-development replay used one previously exposed WMT25 Czech-German
  matrix: 231 items, 128 documents, all 20 machine systems, 190 pairs, and 4,620
  cells. Twenty repetitions changed the random item order; they were not 20
  independent language datasets.
- All four policies used the same familywise alpha of 0.05, mixture-betting
  tests, deterministic completion bounds, exact-census tie rule, four-row counted
  warmup, and shared-cache accounting. Policies were paired on the same matrix
  and randomized item order. There were 2,480 policy-run records in total.
- A cost unit is one previously unrevealed system-item **mean-score cell**.
  Reusing a cached value across pairs incurs no additional cell cost. All warmup
  cells count. There is no charge for synthetic cells in actual human time.
- Each real-data cell averages exactly **two historical human ratings**.
  Therefore the replay rating-count equivalent is twice the reported cell cost;
  full observation corresponds to 9,240 historical ratings. This is not measured
  time, dollars, newly collected annotations, or independent pairwise votes.
- `cost90` is the first acquisition cost with at least 90% of pair decisions
  resolved: 14 of 15 synthetic pairs or 171 of 190 real pairs. `cost_all` is the
  cost to decide every pair. These are decision counts, not a claim that every
  decision is correct; wrong certificates are assessed separately below.
- Every recorded run reached full resolution within its full-matrix cost cap.
  Exact ties can be declared only after the relevant two system means are fully
  observed. Statistical certificates can still be wrong at the controlled error
  rate. Unresolved comparisons are never promoted to a certified total ranking.

The target is the signs of finite released-score mean differences, conditional
on the fixed matrix. It is not the latent expected preferences of new raters or
the ranking on new documents, domains, language pairs, or future systems.

## Primary outcome: mean cells to 90% pair resolution

Lower is better. All original scenarios and all four comparison policies are
shown; the real replay is marked development-only.

| Scenario | Repetitions per policy | `shared_racing` | `random_pair` | `common_all` | `common_active` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Well separated | 100 | 270.57 | 269.27 | 272.53 | 271.37 |
| One near tie | 100 | 418.58 | 295.22 | 318.17 | 289.10 |
| All tied | 100 | 720.00 | 720.00 | 720.00 | 720.00 |
| Heterogeneous noise | 100 | 311.67 | 334.77 | 356.35 | 336.93 |
| Strong item effect | 100 | 463.28 | 461.07 | 465.26 | 463.93 |
| Weak separation | 100 | 707.81 | 709.47 | 716.55 | 711.96 |
| Czech-German development replay | 20 | 4,377.90 | 4,438.80 | 4,554.30 | 4,550.40 |

Relative savings are `100 * (1 - mean_cost_v1 / mean_cost_baseline)`.
Positive numbers favor v1; negative numbers mean it used more labels. Brackets
are 95% paired percentile-bootstrap intervals, using 5,000 resamples of matched
repetitions. They describe variability under these simulation/replay repetitions,
not uncertainty across independent language populations. The two columns below
are the baselines named in the frozen efficiency gate.

| Scenario | Savings vs `random_pair`, % [95% interval] | Savings vs `common_active`, % [95% interval] |
| --- | ---: | ---: |
| Well separated | -0.48 [-0.73, -0.24] | 0.29 [0.16, 0.43] |
| One near tie | -41.79 [-42.58, -40.96] | -44.79 [-45.28, -44.28] |
| All tied | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| Heterogeneous noise | 6.90 [6.09, 7.72] | 7.50 [6.75, 8.27] |
| Strong item effect | -0.48 [-0.63, -0.33] | 0.14 [0.08, 0.20] |
| Weak separation | 0.23 [0.11, 0.36] | 0.58 [0.46, 0.72] |
| Czech-German development replay | 1.37 [0.88, 1.84] | 3.79 [3.51, 4.05] |

The largest favorable synthetic result against both stronger simple baselines
was heterogeneous noise, but 6.90% and 7.50% do not pass the chosen 10% gate.
The 12.54% gain in that scenario against `common_all` is insufficient because
`common_active` and `random_pair` are stronger controls. The gate is a research
project choice, not an ACL submission rule.

For the real replay, v1's mean cost90 of 4,377.90 cells corresponds to 8,755.80
historical-rating equivalents, compared with 8,877.60 for `random_pair` and
9,100.80 for `common_active`. Ratios are unchanged by multiplying all cell
costs by two. These averages do not imply fractional annotations were purchased
in an individual run.

## Complete pairwise resolution is a different objective

Mean cells to decide all pairs:

| Scenario | `shared_racing` | `random_pair` | `common_all` | `common_active` |
| --- | ---: | ---: | ---: | ---: |
| Well separated | 273.44 | 273.44 | 276.90 | 273.44 |
| One near tie | 422.61 | 422.14 | 716.63 | 422.24 |
| All tied | 720.00 | 720.00 | 720.00 | 720.00 |
| Heterogeneous noise | 351.11 | 351.11 | 396.91 | 351.11 |
| Strong item effect | 466.06 | 466.06 | 469.98 | 466.06 |
| Weak separation | 712.49 | 712.97 | 718.63 | 713.12 |
| Czech-German development replay | 4,588.45 | 4,590.70 | 4,617.60 | 4,592.30 |

In the near-tie scenario, v1 is much worse at reaching 90% resolution while its
full-resolution cost is almost unchanged relative to the stronger baselines.
This suggests acquisition ordering, rather than a large increase in the total
evidence needed for full resolution, is the immediate failure mode. The
post-hoc traces below directly investigate that mechanism.

In the real replay, even complete-resolution savings are small: v1 still
reveals 4,588.45 of the available 4,620 cells on average. At half of the full
budget, v1 resolves 32.08% of pairs, compared with 37.03% for `random_pair` and
37.32% for both common-row policies. Thus the small cost90 gain should not be
described as uniformly better performance throughout the annotation process.
Forced sample-mean ordering accuracy is a separate, uncertified secondary
metric and must not substitute for certified resolution.

## Safety: observed false certificates and their uncertainty

The event is **at least one wrong certificate anywhere in a run**, not error
rate per individual pair. The following counts and two-sided 95%
Clopper-Pearson intervals apply separately to each of the four policies.

| Setting | Runs with any wrong certificate, per policy | Observed frequency | 95% interval |
| --- | ---: | ---: | ---: |
| All tied | 2 / 100 | 2.00% | [0.24%, 7.04%] |
| Each of the other five synthetic scenarios | 0 / 100 | 0.00% | [0.00%, 3.62%] |
| Czech-German development replay | 0 / 20 | 0.00% | [0.00%, 16.84%] |

The all-tied false certificates occur in the same paired replicates, 60 and 61,
under each policy. Each affected run has one wrong strict-order certificate.
They must not be presented as eight independent failures, hidden by pooling
with easier scenarios, or described as a zero-error result. A 2% estimate with
an interval extending to 7.04% neither establishes a violation of a 5% bound nor
empirically verifies that the error probability is below 5%. Likewise, zero
errors in only 20 real replays is weak empirical evidence; its upper interval
endpoint is 16.84%.

The mathematical error-control argument in `METHOD.md` is conditional on its
bounded fixed-matrix and sampling assumptions. Tests and finite simulations
check implementation and illustrate behavior; they are not substitutes for
that argument or evidence about new-rater/population uncertainty.

## Post-hoc diagnosis: why the near-tie partial-resolution result is worse

This diagnosis was conducted **after examining the frozen v1 results**. It uses
the first ten already-used `one_near_tie` repetitions, 0 through 9, with run
seeds 20271113 through 20271122. It is a mechanism analysis, not independent
validation, a new success claim, or parameter tuning. The three traced policies
were fixed to `shared_racing`, `common_active`, and `random_pair`; all 30
replays exactly matched the frozen decisions, decision costs, total costs,
resolution milestones, and area-under-curve values.

There are 15 pairs. The difficult pair is systems 4 and 5 (zero-based indices,
pair index 14), which are also the two highest-mean systems in all ten finite
matrices. Numbers below are averages over these ten cases, not the original
100-case estimates above.

| Trace measure | `shared_racing` | `common_active` | `random_pair` |
| --- | ---: | ---: | ---: |
| Cells for systems 4/5 by total cost 180 | 141.3 | 60.0 | 61.4 |
| Cells for the other four systems by cost 180 | 38.7 | 120.0 | 118.6 |
| Share of the first 180 cells spent on systems 4/5 | 78.50% | 33.33% | 34.11% |
| Pairs resolved at total cost 180, out of 15 | 0.0 | 9.0 | 8.9 |
| Fraction of pairs resolved at total cost 360 | 56.67% | 93.33% | 93.33% |
| Cost90 | 419.3 | 290.8 | 297.1 |
| Cost to resolve the near-tied pair itself | 298.9 | 422.4 | 422.5 |
| Cases resolving the near-tied pair by cost90 | 10 / 10 | 0 / 10 | 0 / 10 |
| Cost to resolve all 15 pairs | 422.8 | 422.4 | 422.5 |

Directly measured allocation behavior:

- V1 targets the near-tied pair on the first action after warmup in all ten
  cases. Its computed utility for that action is a median 40.59 times the best
  other action (range 2.07 to 265.44).
- Up to cost 180, 669 of 863 v1 pair-target actions (77.52%, pooled across cases)
  target the near-tied pair. The corresponding `random_pair` count is 105 of
  1,156 actions (9.08%). Warmup is excluded from these action counts.
  `common_active` does not choose a target pair, so that frequency is not
  defined for it. An action can reveal one or two cells, so action frequencies
  and cell-cost shares are different quantities.
- By total cost 360, v1 has purchased a mean 119.8 of 120 cells for each
  near-tied system, while the other four system counts average only 23.2, 25.9,
  27.8, and 43.5. Cached scores cannot complete paired-prefix observations
  against systems whose own prefixes have not caught up.

The mechanism is consistent with the exact frozen heuristic: a
variance-to-squared-gap score makes a difficult near-zero-gap pair attractive,
even when resolving many easier pairs would advance the cost90 objective more
quickly. In these ten cases, v1 uses 44.19% more cells than `common_active` and
41.13% more than `random_pair` to reach cost90, while all-pairs costs differ by
only 0.4 and 0.3 cells respectively.

**There is an important top-pair tradeoff.** Reaching 14 of 15 decisions allows
the two strongest systems' relative order to remain unresolved. The simpler
policies do exactly that in these ten cases. V1 resolves that particular
comparison earlier, but this does not retroactively make it a cost90 win, nor
does the pair's resolution alone establish that a globally best system has
been certified. Partial pair coverage, full ordering, and best-system
identification are distinct objectives. A follow-up must choose its objective
before assessing a new scheduler, rather than selecting whichever metric
looks favorable afterward.

A scheduler aligned more directly with certificate progress is a reasonable
follow-up **hypothesis**, not a result established here. No later-version
outcome is included in this frozen-v1 report.

## Verification, provenance, and reproducibility

- The v1 core and loader have 37 passing unit tests in `test_racing.py` and
  `test_loader.py`, including callback-only acquisition, shared-prefix
  consistency, non-duplicate costs, zero-budget behavior, exact ties, and
  reproducibility. The historical-input test skips when that CSV is absent.
- `math_audit.json` records an independent exact-rational enumeration of 2,160
  labeled permutations over three small null cases, plus boundary and
  shared-prefix checks. It passed; its finite-case scope is not a general proof.
- `output_audit.json` records 156,221 assertions across all 2,480 trials,
  independent truth calculations for 600 synthetic populations and all 190 real
  pairs, and eight traced real replays with 36,751 query events. It passed with
  no issues. Bootstrap percentile intervals were not independently recomputed
  by that audit. Trace replays reuse the frozen controller; truth and arithmetic
  checks are independent.
- The post-hoc diagnostic verified that the core, validation driver, protocol,
  original trial and summary files, and earlier output audit retained their
  checksums. Diagnostic outputs are separate from `validation_v1/`.

From this package directory, reproduce v1 in a **new** output directory:

```bash
python3 -m unittest test_racing test_loader -v
python3 validate.py --synthetic-reps 100 --real-reps 20 \
  --real-csv inputs/cs-de_DE_human_scores.csv --out reproduced_v1
python3 audit_math.py
python3 audit_outputs.py
python3 diagnostics/diagnose_near_tie.py --out diagnostics/reproduced_near_tie_v1
```

The two audit commands check the frozen inputs/outputs described by their
scripts; they do not automatically replace their targets with `reproduced_v1`.
See `README.md` and `inputs/provenance.json` for the pinned public input URL.
The CSV is not included in the distributable archive because its text
redistribution license was not confirmed.

Frozen references:

- Protocol SHA-256:
  `062e4a8d062adf4c05b7fae68dca76691bcccc03ffa11302848eeaec260e04ae`.
- Trial records SHA-256:
  `cea91c6ddfafa1edf6b640973a18847c2321e0939cfd1331b8041d57ba146873`.
- Historical input SHA-256:
  `f6de1e8da36fb67c7d99ec354e8807a016045c2d5b420f3ab8aeb72816bc279d`.

## Claim boundary

This is a fully specified, tested development prototype with an informative
negative efficiency result and a measured scheduling failure mode. It has not
demonstrated a general reproducible advantage over strong random/common-active
sampling baselines, prospective savings in actual annotation time, or improved
rankings on untouched MT data. No new human annotations or independent holdout
were used. Published adaptive-allocation methods have not yet been reproduced
as baselines, so this is not evidence of methodological novelty. These
limitations remain even where a development-only bootstrap interval excludes
zero.
