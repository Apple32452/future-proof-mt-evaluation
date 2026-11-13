# Hindsight allocation diagnostic: results

**Conclusion:** on the five prespecified exposed development item orders,
allocation-only changes have limited remaining room under the current
certification rules. The numerical cost bound permits less than **4.86% further
mean savings over v2**. Even the full-data hindsight optimum cannot reach the
project's 10% mean-saving target against both main random baselines on these
five runs. This is a conditional diagnostic, not a general impossibility result
for efficient MT evaluation.

## Question and fixed scope

The question was: if a scheduler could see all existing scores in advance, how
cheaply could it reach the same target with the same inference?

The diagnostic used the same complete Czech-German development matrix of 231
items and 20 systems, normalized from the known 0–100 scale to [0,1]. It retained:

- The target of 171 issued decisions among 190 pairs (90% pair coverage).
- Familywise alpha 0.05 and all original betting and completion rules.
- One fixed shared item order per run and a counted four-row-per-system warmup.
- Unique system-item score-cell costs, caching, and the original exact-tie rule.
- No transitive inference, new annotations, or changed score bounds.

Replicates 0–4, with seeds 2026111300–2026111304, were selected in
`oracle_protocol.json` before any oracle outcome. They are the first five of
the previously reported 20 real-development runs, not a favorable subset chosen
after solving. All numbers below compare the same five runs; they must not be
mixed with the 20-run means in `RESULTS_V2.md`.

The optimization permits arbitrary per-system prefix acquisitions **after the
same warmup**. This is more permissive than the actual pair-bundle schedulers,
but does not include policies that skip warmup, change item order, or select
arbitrary non-prefix items. Every all-counts-four state had zero resolved pairs,
so the original procedures must finish warmup before reaching the target.

## Per-order optimization outcomes

Costs are newly revealed historical mean-score cells; lower is better.

| Replicate | Conservative numerical lower bound | Numerically optimal replayed incumbent | Actual v2 cost90 | Solve/replay seconds |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4,063 | 4,064 | 4,324 | 8.257 |
| 1 | 4,085 | 4,086 | 4,251 | 9.322 |
| 2 | 4,137 | 4,138 | 4,359 | 6.561 |
| 3 | 4,135 | 4,136 | 4,316 | 8.184 |
| 4 | 4,146 | 4,147 | 4,365 | 7.033 |
| Mean | **4,113.2** | **4,114.2** | **4,323.0** | — |

All five mixed-integer solves reported numerical optimality with zero solver
gap, within the 30-second-per-run limit. The one-cell difference between the
reported lower bound and feasible optimum comes from the deliberately
conservative `floor(raw_dual_bound - 1e-7)` rule. It is **not** a confidence
interval or an unfinished optimization gap.

Each returned allocation was replayed through the original `RaceState`
inference. Each replay issued 171 decisions and reached that target at exactly
the proposed cost; none of its issued certificates was wrong relative to the
fixed full-matrix truth. Five error-free hindsight replays do not empirically
establish a 5% error guarantee. Statistical validity belongs to the original
inference argument, not this optimization exercise.

## Remaining allocation headroom

For a comparator's mean cost C, the feasible hindsight allocation gives
`1 - 4114.2/C`, and the conservative numerical lower bound gives a maximum
possible reduction of `1 - 4113.2/C` in the stated expanded-prefix class.

| Comparator | Observed mean cost90, same five runs | Hindsight saving range |
| --- | ---: | ---: |
| v2 certificate-focused | 4,323.0 | 4.83%–4.85% |
| v1 shared racing | 4,388.4 | 6.25%–6.27% |
| Random pair | 4,424.0 | 7.00%–7.03% |
| Common active | 4,551.8 | 9.61%–9.64% |
| Common all | 4,555.2 | 9.68%–9.70% |

These ranges bracket optimization headroom conditional on the fixed five
orders, with the numerical-solver qualification below. They are not statistical
uncertainty intervals. The lower endpoint is feasible for the hindsight
expanded-prefix class; it does not promise a deployable pair-bundle scheduler
can achieve it without access to unobserved scores.

In particular, a 10% reduction against random pair would require mean cost at
most 3,981.6 cells, and against common active at most 4,096.62 cells. Both are
below the 4,113.2 numerical lower bound. Individual orders can differ: the
conclusion concerns the aggregate of these five prespecified orders, not every
single-order comparison.

Each observed score cell is the mean of two historical ratings, so rating-count
equivalents are twice the cell costs. This is not an estimate of annotation
minutes or money, and no human annotation budget was spent in this diagnostic.

## What this establishes, and what it does not

The result gives a concrete reason to deprioritize further scheduler-only
tuning in this exact configuration. Even an unrealistically informed allocator
has limited headroom over v2. It does not show that v2 is generally optimal.

It also does **not** prove that the current stopping rule is unnecessarily
conservative. Close system means may genuinely require substantial evidence.
The diagnostic holds inference fixed and cannot distinguish that difficulty
from avoidable conservatism within the inference procedure. It does not bound
methods with different valid tests, pairwise evidence sharing, different item
orders, model-assisted estimation, or a different scientifically justified
target.

The lower bounds are standard numerical HiGHS solver reports, rounded down;
they are not independently verified exact-rational optimization certificates.
`METHOD_ORACLE.md` proves the discrete formulation's equivalence, and the tests
and output audit check its implementation and replays. Those checks do not
convert a floating-point solver result into a formal exact-arithmetic proof.

The full project suite passes **63 tests**, including nine diagnostic-specific
tests and exhaustive optimization comparisons on tiny count grids. The separate
output audit passes **22,122 checks** with zero issues and no new real-data
optimizations. It independently recomputes 950 pair-specific betting thresholds,
replays all five incumbents in two acquisition orders each, and reproduces
21,615 v2 cell acquisitions across the five paired runs. The check count includes
individual query-accounting assertions; it is not a count of independent
experiments. Frozen input, source, protocol, and result hashes remained intact.

All data were already exposed in earlier development. The five orderings are
not independent datasets or languages. No claim of new-human validation,
publication readiness, or novelty follows. The published strong allocation
baseline remains unimplemented.

## Recommended next research question

Keep v2 unchanged as a reference, along with the 171-pair target and alpha 0.05.
Investigate whether an alternative, mathematically justified paired
finite-population sequential procedure can certify the same claims with less
evidence. Derive its assumptions and simultaneous/anytime error control before
testing efficiency. First compare inference procedures on the identical fixed
common-row acquisition stream, with identical score bounds and costs, to isolate
the inference change. Merely freezing v2's policy code would not guarantee
identical acquisitions: its priorities depend on certificate forecasts and
which pairs remain unresolved. Adaptive integration should be a separate later
experiment. Validate on exact small cases and difficult null/near-tie simulations
before reserving untouched human data for final confirmation.

A practical-equivalence tolerance is a separate possible research objective,
not a post hoc performance fix. It would need a domain-grounded definition,
advisor agreement, and a separately frozen protocol. The present study does
not change the target or implement either follow-up.

## Reproduce

After obtaining the exact input described in `README.md`, run from the project
directory with a fresh output directory:

```bash
python3 -m unittest discover -v
python3 run_oracle_diagnostic.py --out reproduced_oracle
python3 audit_oracle_outputs.py --run-dir reproduced_oracle
```

The bundled `oracle_v1/` contains the frozen protocol manifest, five complete
run records, aggregate summary, and completion checksum. `oracle_audit.json`
records the separate output audit. The raw source-text score CSV is not
redistributed in this archive.
