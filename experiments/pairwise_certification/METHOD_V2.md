# Certificate-aware shared-prefix racing: acquisition revision v2

Status: one prospective, acquisition-only development revision. It is not a new
statistical test, an independent replication, or an established efficient MT
evaluation method. No performance result is implied by this specification.

## What stays frozen

The finite-matrix target, bounded-score assumption, single shared random item
permutation, per-system prefixes, annotation cache, four-row warmup, annotation
cost definition, betting fractions, mixture weights, `alpha = 0.05` familywise
error allocation, deterministic completion bounds, and actual stopping logic
are unchanged from `METHOD.md` and `pairwise_racing.py`.

The implementation inherits the actual inference methods from `RaceState`.
Only acquisition priorities change. Actual certificates depend solely on
observed labels and the frozen inference rule, never on projected labels.
The pair-local supermartingale and simultaneous all-prefix argument in
`METHOD.md` therefore apply unchanged. This remains a guarantee about a fixed
matrix of existing scores, not fresh-annotator noise or future-document quality.

## Motivation and objective

The v1 variance-to-gap heuristic can devote substantial effort to very close
pairs. The v2 heuristic instead forecasts distance to the actual certificate
boundary. Its development objective is fewer newly scored system-item cells to
certify 90% of all pairs, at the same statistical reliability and with unresolved
pairs reported honestly.

The forecast is a plug-in planning approximation. It is **not** expected
remaining annotation cost, a confidence bound, a certificate probability,
information gain, or an optimal allocation rule.

## 1. Constant-mean continuation for each unresolved pair

For pair `p=(a,b)`, let its current paired prefix have length `n_p`, difference
sum `S_p`, and mean `dbar_p = S_p/n_p`. Use zero for an empty-prefix mean. Let
`M_p` and `peak_p` denote its current directional log-capital vectors and
historical mixture peaks. These are exactly the observed-data states of v1.

For each possible future paired step, use the fixed forecast

\[
\widetilde D_{p,n_p+j}=\bar d_p,\qquad
j=1,\ldots,N-n_p.
\]

Run this continuation through a **copy** of the existing betting process. For
the positive-direction forecast at projected step `j`, for example,

\[
\widetilde X_j=(1+\bar d_p)/2,\qquad
\widetilde q_j=
\frac{N/2-[(n_p+S_p)/2+(j-1)\widetilde X_j]}
     {N-n_p-j+1}.
\]

Use the same seven fixed betting fractions and neutral boundary guard as v1.
Update the current capital products, then their historical peaks. Do not
multiply the historical peak instead of the current product. The reverse
direction uses `(1-dbar_p)/2` and its own prefix sum.

This already uses finite-population conditioning. No additional normal
approximation or finite-population variance correction is introduced.

## 2. Forecast the first usable certificate

At projected paired step `h`, form the paired-prefix-only completion interval

\[
\widetilde C_p(h)=\left[
\frac{S_p+h\bar d_p-(N-n_p-h)}{N},
\frac{S_p+h\bar d_p+(N-n_p-h)}{N}
\right].
\]

Let `h_p` be the smallest integer from 1 through `N-n_p` for which:

1. Exactly one projected directional peak exceeds the frozen betting threshold;
2. The projected completion interval excludes zero; or
3. The projection reaches census, `h = N-n_p`.

Use the same strict numerical comparisons as v1. Simultaneous crossing of both
directional thresholds is not a betting certificate. Deterministic completion
or projected census can still resolve that case. At census a pair is resolved
even when its projected mean difference is exactly zero.

Thus every unresolved non-census pair has a defined finite forecast

\[
1\le h_p\le N-n_p.
\]

The actual algorithm does not stop because a forecast predicts a certificate.
It acquires the next real score, updates v1 inference, and forecasts again when
the pair's actual prefix changes.

## 3. Cache-aware acquisition utility

Replace the v1 pair gain with

\[
g_p=1/h_p.
\]

The remainder of the action utility is unchanged. For each system `s`,

\[
B_s=\sum_{q=(s,r)\text{ unresolved},\ n_r>n_s}g_q.
\]

An unresolved candidate pair `(a,b)` has utility

\[
U_{ab}=\begin{cases}
B_a,&n_a<n_b,\\
B_b,&n_b<n_a,\\
(B_a+B_b+g_{ab})/2,&n_a=n_b.
\end{cases}
\]

Choose a maximum-utility action with the same seeded uniform numerical-tie
rule as v1. Charge every newly acquired cell, including initialization and a
final partially affordable two-cell bundle. Reuse already scored cells across
all applicable comparisons.

This is a resolution-rate **proxy** per current action cost. `h_p` counts
predicted paired observations, not total future new cells. The utility's
one-/two-cell normalization accounts for the present acquisition only.

## 4. Caching, ties, and termination

A pair forecast depends only on its current paired prefix and fixed constants,
so it can be cached until that prefix advances. An unpaired observation can
change action costs or exact completion bounds without changing this forecast.
Resolved pairs are masked out even when their cached forecast is unchanged.

A predicted zero difference normally gives `h_p=N-n_p`; it is not mistaken
for an observed exact tie. The census endpoint gives every remaining pair a
positive gain, so no extra exploration parameter is added. Every action buys
at least one previously unseen cell; acquisition ends by full census or the
declared budget. A pair may still be deprioritized within a limited budget, and
unresolved comparisons must remain visible in the results.

## 5. Explicit limitations of the projection

- Constant future differences ignore sampling noise. An unusually favorable
  warmup can make projected completion overly optimistic.
- Past volatility affects the current betting products, but future volatility
  is not modeled.
- If one system is ahead, the forecast deliberately ignores individually known
  ahead-system scores. Its hypothetical differences may be incompatible with
  those cells. It is therefore not a guaranteed lower or upper bound on cost.
- Reducing one-step projected certificate distance does not guarantee a globally
  efficient acquisition schedule or success at the 90% target.
- Repeated seeds on an already examined human-score matrix assess conditional
  randomization behavior, not independent generalization.

## 6. Attribution

The broad idea of forecasting the benefit of another evaluation while holding
current estimated means fixed overlaps with the confusion-minimization policy
in Zouhar et al. (2026), *Dynamically Allocating Evaluation Effort for Model
Ranking*, Findings of EMNLP, Section 4.2, Equations 14-19. That policy uses an
approximate ranking-certainty objective. This prototype instead forecasts
distance to its unchanged finite-population certificate boundary, with
cache-aware paired actions. This distinction is not, by itself, a novelty claim.

https://aclanthology.org/2026.findings-emnlp.360.pdf

The underlying betting inference remains attributed to the references in
`METHOD.md`. No changes to that inference are proposed here.

## 7. Correctness checks before performance evaluation

Run:

```
python3 -m unittest -v test_certificate_v2
```

The independent scalar reference checks vectorized projection, mean-zero census
behavior, current versus peak capital, contradictory crossings, positive finite
horizons, and input immutability. State and runner tests check unchanged
inference for identical observations, forecast-cache invalidation, absence of
forecast-based actual certificates, unique prefix queries, budget accounting,
reproducibility, and exact ties at census.

These tests are implementation checks, not performance results. Freeze this
one acquisition policy and its comparisons before running the development
experiment. Do not replace an unfavorable result with a post-hoc policy sweep.
