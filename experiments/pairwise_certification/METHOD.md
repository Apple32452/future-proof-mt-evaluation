# Shared-prefix pairwise racing, version 1

Status: a mathematically specified experimental baseline, not a claim of novelty,
optimality, or superiority to random sampling. This specification separates a
provably controlled stopping rule from an unproved acquisition heuristic.

## 1. Target and assumptions

There are `K >= 2` systems and `N >= 1` fixed evaluation items. Let
`Y[i, s] in [0, 1]` be a fixed human score, with larger values meaning better
quality. A known bounded annotation scale can be mapped affinely to this range.
The scale must be established independently of hidden target outcomes; observed
minimum/maximum values do not establish valid population bounds. A clipping
transform, if used, must be declared and changes the target ranking.

Define

\[
\mu_s=N^{-1}\sum_{i=1}^N Y_{i,s},\qquad
\Delta_{ab}=\mu_a-\mu_b,\qquad
P=K(K-1)/2.
\]

We seek correct statements about the signs of these finite-matrix differences.
An evaluation cost is one newly revealed `(item, system)` score, not one source
item and not one pairwise comparison. Existing score aggregates are treated as
fixed observations. If an aggregate originally required several human ratings,
the aggregate-cell count and original rating count must be distinguished.

The guarantee below is conditional on the fixed matrix. It does **not** account
for uncertainty from new annotators, establish latent translation quality, or
guarantee ranking generalization to future documents. Missing cells require a
separately specified protocol; this version assumes a rectangular matrix.

## 2. Shared random prefixes and annotation reuse

Generate one uniformly random permutation `pi` of the `N` item indices,
independently of the target scores. System `s` has an observed prefix length
`n_s`; its known cells are exactly `Y[pi[1:n_s], s]`. For pair `(a, b)`, the
available paired prefix length is `n_ab = min(n_a, n_b)`.

For an unresolved pair, its next paired observation has position
`j = min(n_a, n_b) + 1`. If prefix lengths differ, reveal the next cell of the
lagging system (cost one). If they agree, reveal the two missing cells at that
position (cost two). Each physical score is acquired at most once, retained in
a shared cache, and reused by every applicable pair. Advancing a system can
therefore advance several pairwise prefixes without additional annotation.

Only a pair's own ordered paired prefix enters its statistical test. Do not
skip items based on scores, select arbitrary cached differences as if they were
a random sample, or inspect unqueried scores when making acquisition choices.
This version adapts **which systems/pairs receive scores**, not the content-based
ordering of the item stream.

Start with up to four common rows across all systems, charging every revealed
cell to the budget. Infer throughout acquisition; initialization does not create
free labels. Stop on a complete set of pairwise conclusions or exhaustion of
the declared budget. Every budget comparison must use the actual spent count.

## 3. Pairwise finite-population betting tests

For the ordered pair `(a, b)`, let

\[
X_j=\frac{Y_{\pi(j),a}-Y_{\pi(j),b}+1}{2}\in[0,1],
\qquad \bar X_N=(\Delta_{ab}+1)/2.
\]

The positive-direction null is `H0+: Delta_ab <= 0`, equivalently
`mean(X) <= m`, with `m = 1/2`. At paired prefix step `j`, define

\[
S_{j-1}=\sum_{\ell<j}X_\ell,\qquad
q_j=\frac{Nm-S_{j-1}}{N-j+1}.
\]

Fix the fractions and mixture weights before validation:

```
c = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95]
w = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
```

For each fixed fraction `c_l`, initialize capital `M_0,l = 1` and, when
`q_j > 0`, update

\[
M_{j,\ell}=M_{j-1,\ell}
 \left(1-c_\ell+c_\ell\frac{X_j}{q_j}\right),
\qquad E_j^+=\sum_\ell w_\ell M_{j,\ell}.
\]

Use log-domain arithmetic in the implementation. The implementation uses a
neutral factor when `q_j <= 1e-12`, a predictable choice to skip a bet near a
numerically delicate boundary. A value `q_j > 1` is allowed:
the factors remain nonnegative and are evidence against this direction, not an
error. If `q_j = 0` and `X_j = 0`, use a neutral factor of one. If the observed
sum exceeds `Nm`, the positive null is deterministically impossible; this can
be handled by an exact completion certificate. Cases impossible under a true
null need not be treated as ordinary division operations.

For the negative-direction null `H0-: Delta_ab >= 0`, apply the identical test
to `1 - X_j`. Reject a direction when its mixture reaches

\[
T=\frac{2P}{\alpha};\qquad \alpha=0.05\text{ in the main validation.}
\]

A rejection of `H0+` certifies `a > b`; a rejection of `H0-` certifies `b > a`.
Capital processes use only their own pair-prefix observations and the frozen
mixture. Globally cached future-prefix labels must not be used to choose bets.
The heuristic in Section 5 may use all already observed pair-prefix summaries.

These are anytime-valid **tests**, not posterior probabilities or numerical
confidence intervals for `Delta_ab`. A heuristic uncertainty score used for
acquisition must not be reported as a calibrated confidence interval.

## 4. Deterministic completion certificates and guarantee

Let `S_s` be the sum of the `n_s` observed scores for system `s`. Whatever the
unseen scores are,

\[
\mu_s\in\left[\frac{S_s}{N},
                    \frac{S_s+N-n_s}{N}\right]=[L_s,U_s].
\]

Thus `L_a > U_b` certifies `a > b` with no error probability, and conversely for
`b > a`. At census of both systems, their finite-matrix means and any exact tie
are known. Before census, a failed significance test is **not** evidence of a
tie. Return unresolved comparisons rather than forcing a complete ordering.
No practical-equivalence threshold is used in this version.

### Theorem: familywise error control at all annotation budgets

For any fixed bounded matrix satisfying the assumptions, and any history-based
scheduler preserving the shared-prefix observation rule,

\[
\Pr_\pi\{\text{at least one incorrect certified strict pairwise ordering
at any time}\}\le\alpha.
\]

The probability is over the random item permutation (and, if needed, the
scheduler's independent tie-breaking randomness). It is not a posterior
probability attached to one realized ranking.

**Proof.** For each pair, regard its entire potential random-prefix sequence as
defined before scheduling. In that pair's own prefix filtration, under a true
positive null,

\[
\mathbb E[X_j\mid X_1,\ldots,X_{j-1}]
=\frac{N\bar X_N-S_{j-1}}{N-j+1}\le q_j.
\]

For `q_j > 0`, each nonnegative multiplier therefore has conditional expectation
at most one. The zero-boundary convention is valid because under the null a
zero remaining allowed total forces the next observation to be zero. Each
capital process, and its fixed convex mixture, is a nonnegative supermartingale
starting at one. Ville's inequality bounds the probability of ever crossing
`2P/alpha` by `alpha/(2P)` for each true directional null.

Take the union bound over the at most `2P` directional tests. Their dependence
from the common permutation and reused judgments does not invalidate this
bound. On the resulting simultaneous all-prefix event, every pairwise prefix
that an adaptive scheduler reaches is safe. Deterministic completion
certificates are always valid and add no error probability. This proves the
claim. In particular, **no global-filtration martingale property is asserted**:
other systems' observations may reveal information about a pair's future, but
the already-established all-prefix event covers every adaptively chosen cursor.

Finite-precision implementations must handle boundaries conservatively and
record contradictions or numerical failures, rather than silently repairing
them by choosing a preferred ordering. Error-rate validation must count an
incorrect certificate anywhere on the trajectory, not only in the final output.

## 5. Frozen variance-to-gap acquisition heuristic

For unresolved pair `p`, let `n_p` be its paired prefix length, `dbar_p` its
sample mean difference, and `s2_p` its usual sample variance (`s2_p = 1` if
`n_p < 2`). For an empty prefix use `dbar_p = 0`. Set

\[
A_p=\frac{s_p^2+10^{-4}}{\bar d_p^2+0.01^2},\qquad
u_p(n)=\frac{A_p}{n+1},\qquad
g_p=A_p\left(\frac1{n_p+1}-\frac1{n_p+2}\right).
\]

For each system `s`, define the gain from advancing it by one prefix position:

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

Choose a maximum-utility action, breaking numerical ties uniformly with a
separately seeded generator. Acquire its missing cells in random order, one at
a time, never exceeding the remaining budget; a final one-cell remainder may
therefore pay for only part of a two-cell action. Advance all applicable paired
prefixes and update inference after every acquired cell. The equal-prefix expression includes the
new `(a,b)` difference and the effects on pairs whose other system is already
ahead; division by two charges both newly annotated outputs.

This is an explicit variance-to-squared-gap **heuristic**, using a frozen
one-step approximation that holds `A_p` fixed during action scoring. It is not
the actual expected decrease in ranking error, exact information gain, or an
optimal experimental design. The statistical theorem does not imply that this
scheduler beats random allocation or minimizes annotation cost.

## 6. Required validation and limits of the claim

Compare allocation policies using the same score scale, initialization cost,
inferential tests, exact completion certificates, and annotation accounting.
Include a common-row random baseline, since sharing labels alone is not a new
advantage if the baseline shares them too.

Report both any-time false-certificate frequency and resolution versus actual
annotation cost. Compare costs at a fixed resolution target and reliability
level; leaving difficult pairs unresolved is not equivalent to obtaining the
same complete ranking more cheaply. Count and report budget-exhausted runs.

Stress tests must include ties, near ties, heterogeneous variance, strong shared
item effects, and score-bound violations. Replay of previously seen human
ratings is development validation, not a fresh holdout or a new annotation
study. Simulations and unit tests check implementation behavior; they do not
replace the theorem or establish empirical superiority on new MT data.

## References and attribution

- Waudby-Smith, I. and Ramdas, A. (2020). *Confidence sequences for sampling
  without replacement*. NeurIPS. Establishes finite-population sequential
  confidence methods and the relevant local-filtration setup.
  https://proceedings.neurips.cc/paper/2020/file/e96c7de8f6390b1e6c71556e4e0a4959-Paper.pdf
- Waudby-Smith, I. and Ramdas, A. (2024). *Estimating means of bounded random
  variables by betting*. JRSS-B, 86(1), 1-27. Provides the established bounded-mean
  betting framework, including without-replacement extensions.
  https://academic.oup.com/jrsssb/article/86/1/1/7043257

The proof above specializes standard nonnegative-supermartingale, Ville, and
Bonferroni arguments to this shared-prefix construction. These ingredients and
their combination are not asserted to be a new statistical theorem.
