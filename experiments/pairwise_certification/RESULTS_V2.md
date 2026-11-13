# Certificate-focused allocation v2: development results

**Conclusion:** the single allocation revision removes v1's pronounced
near-tie scheduling weakness, but its advantage over strong simple baselines
on the exposed MT matrix remains modest. It does not pass the preselected
development gate, establish generalization, or demonstrate an ACL contribution.

## What changed and what did not

The new policy, `certificate_racing_v2`, forecasts how many further paired
observations a fictional constant-mean continuation would need to reach the
existing certificate boundary. It assigns inverse-forecast priorities to
cache-aware acquisitions. The forecast is a planning heuristic, not expected
annotation cost, statistical evidence, or a confidence bound.

All actual inference is inherited unchanged from `pairwise_racing.py`: known
score bounds, shared uniform item order, prefix observations, mixture tests,
familywise alpha 0.05, deterministic completion, and exact-census ties. Warmup,
cost accounting, and seeded tie-breaking remain unchanged. `METHOD_V2.md`
specifies the rule; `protocol_v2.json` was fixed before v2 performance results.
This is a post-v1 development revision, not independent confirmation or a
preregistration.

The run added **620 policy trials**: six synthetic scenarios with 100 repetitions
each, plus 20 item-order repetitions on the previously exposed Czech-German
matrix. It reused the **2,480 audited v1 baseline records**, paired on the same
matrices and seeds. The 3,100 comparison records are not 3,100 new experiments.

## Cost to certify at least 90% of pairs

Numbers are mean newly revealed score cells; lower is better. Each synthetic
matrix has 120 items, 6 systems, 15 pairs, and 720 possible cells. The real matrix
has 231 items, 20 systems, 190 pairs, and 4,620 cells. The 90% target therefore
requires 14 or 171 pair decisions, respectively. Incorrect certificates are
reported separately, not removed from the decision-count denominator.

| Scenario | v2 certificate-focused | v1 shared racing | Random pair | Common all | Common active |
| --- | ---: | ---: | ---: | ---: | ---: |
| Well separated | 267.44 | 270.57 | 269.27 | 272.53 | 271.37 |
| One near tie | 287.69 | 418.58 | 295.22 | 318.17 | 289.10 |
| All tied | 720.00 | 720.00 | 720.00 | 720.00 | 720.00 |
| Heterogeneous noise | 304.11 | 311.67 | 334.77 | 356.35 | 336.93 |
| Strong item effect | 459.78 | 463.28 | 461.07 | 465.26 | 463.93 |
| Weak separation | 708.25 | 707.81 | 709.47 | 716.55 | 711.96 |
| Czech-German development | 4,336.05 | 4,377.90 | 4,438.80 | 4,554.30 | 4,550.40 |

In the near-tie scenario, v2 reduces cost90 by **31.27% relative to v1**, but
only **2.55% relative to random pair** and **0.49% relative to common active**.
The large improvement primarily repairs v1's weakness; it is not a comparably
large advance over the stronger controls.

On the real development matrix, v2 cost90 savings are:

| Comparator | Relative savings | 95% paired bootstrap interval |
| --- | ---: | ---: |
| v1 shared racing | 0.96% | [0.53%, 1.45%] |
| Random pair | 2.31% | [2.02%, 2.62%] |
| Common all | 4.79% | [4.51%, 5.10%] |
| Common active | 4.71% | [4.45%, 5.00%] |

Savings are `1 - mean(v2 cost90) / mean(comparator cost90)`. Intervals use
5,000 resamples of matched runs. These 20 repetitions assess randomization
variability on one fixed matrix, not variation across independent languages,
raters, domains, or test sets.

The frozen gate required at least 10% savings against **both random pair and
common active** on this real development matrix. It was not passed. The largest
favorable synthetic gains against those two controls were 9.16% and 9.74% in
heterogeneous noise. Weak separation was slightly worse than v1. The 10% gate
is an internal project criterion, not an ACL rule.

## Complete ranking and limited-budget accuracy

Mean cost to decide every pair:

| Scenario | v2 certificate-focused | v1 shared racing | Random pair | Common all | Common active |
| --- | ---: | ---: | ---: | ---: | ---: |
| Well separated | 273.44 | 273.44 | 273.44 | 276.90 | 273.44 |
| One near tie | 422.22 | 422.61 | 422.14 | 716.63 | 422.24 |
| All tied | 720.00 | 720.00 | 720.00 | 720.00 | 720.00 |
| Heterogeneous noise | 351.11 | 351.11 | 351.11 | 396.91 | 351.11 |
| Strong item effect | 466.06 | 466.06 | 466.06 | 469.98 | 466.06 |
| Weak separation | 713.19 | 712.49 | 712.97 | 718.63 | 713.12 |
| Czech-German development | 4,592.25 | 4,588.45 | 4,590.70 | 4,617.60 | 4,592.30 |

Full-resolution costs barely improve against common active and can be worse
than v1 or random pair. On the real matrix, v2 still uses about 99.4% of all
available cells to complete the ranking. Its partial-resolution gain must not
be presented as a substantial reduction in complete-ranking cost.

At half the real-data budget, 2,310 cells:

| Method | Certified pair fraction | Forced sample-mean ordering accuracy |
| --- | ---: | ---: |
| v2 certificate-focused | 39.50% | 96.05% |
| v1 shared racing | 32.08% | 96.50% |
| Random pair | 37.03% | 96.53% |
| Common all | 37.32% | 96.34% |
| Common active | 37.32% | 96.34% |

V2 certifies more pairs at this budget, while its forced ordering accuracy is
slightly lower. Forced accuracy guesses orders for unresolved pairs and is not
a certified ranking. These are distinct outcomes, not interchangeable measures.

## The top-two tradeoff

In the 100 near-tie synthetic cases, v1 resolves the two highest-mean systems'
comparison at mean cost **298.55**, while v2 does so at **422.22**. V1 resolves
that pair by half budget in 100/100 cases; v2 does so in 0/100. V2 instead
reaches 14 of 15 pair decisions sooner and leaves the difficult leading pair
until later. Thus broader partial coverage does not imply faster winner
identification.

The direction of this tradeoff is not universal. On the real development
matrix, the top-two comparison is resolved at mean cost **4,439.65** for v2,
versus **4,574.95** for v1, **4,523.05** for random pair, and **4,522.65** for
common active. None resolves it by half budget. These are evaluator-only
diagnostics: full-data top-two identities never enter the acquisition rule.
Certifying that pair alone does not certify a globally best system. In exactly
tied synthetic populations, index-based top-two tie-breaking is merely a
reporting convention, not evidence of a uniquely best pair.

## Safety, audits, and scope

V2 has at least one incorrect certificate in **2/100 all-tied runs**; the
two-sided 95% binomial interval is **[0.24%, 7.04%]**. Each other synthetic
scenario has 0/100 errors, and the real replay has **0/20**, whose interval is
**[0%, 16.84%]**. Do not claim zero errors overall or empirical verification of
a 5% upper bound. The null failures occur on the same paired populations across
policies, so they are not independent policy-specific evidence.

The mathematical error-control statement concerns a fixed complete bounded
score matrix and the stated randomization assumptions. It does not establish
latent human quality, fresh-rater reliability, or performance on future data.

- The complete suite has **54 passing tests**, including **17 new independent
  v2 tests** for scalar/vectorized forecast agreement, inference immutability,
  cache invalidation, reproducibility, prefix constraints, and score costs.
- `output_audit_v2.json` reports **PASS**, with **217,041 assertions** across
  620 new and 2,480 reused records, 35 policy/scenario aggregates, all 28 paired
  bootstrap comparisons, and 9,170 queried cells from two traced real replays.
  Frozen hashes remained unchanged. The audit recomputes arithmetic and truths;
  it does not replace the statistical proof or independent data validation.
- Each real cell averages two historical human ratings. Rating-count equivalents
  are twice these cell costs; no new annotations, dollars, or annotation-time
  measurements were collected.
- No untouched data, private collaborative implementation, or SAFE-EVAL code
  was used. The published confusion-minimization baseline has not been
  implemented here; related allocation ideas are acknowledged in `METHOD_V2.md`.

## Reproduce and interpret

Obtain the exact previously exposed input described in `inputs/provenance.json`
and `README.md`; the archive intentionally excludes the raw source-text CSV.
From the project directory, use a new output directory:

```bash
python3 -m unittest discover -v
python3 validate_v2.py --out reproduced_v2
```

The driver checks frozen v1 source/protocol hashes and pairs the new runs with
the bundled audited reference records. `validation_v2/` contains all results,
not only favorable subsets. No policy was substituted after seeing v2 results.

The supported conclusion is a well-defined, reproducible development algorithm
with better partial-certificate scheduling than v1, modest real-data savings
over random controls, and explicit top-pair/full-ranking tradeoffs. Further
methodological or novelty claims require separate evidence. This report does
not include or anticipate any later hindsight cost-bound diagnostic.
