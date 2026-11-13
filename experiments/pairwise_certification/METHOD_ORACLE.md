# Hindsight final-prefix cost bounds: method and interpretation

This is an **offline diagnostic on already exposed development data**, not a
deployable annotation policy, an independent validation set, or a new stopping
test. Its purpose is to distinguish inefficient allocation from the annotation
cost required by the current certification machinery.

The algorithm, validation, and oracle diagnostic answer different questions:

- v1/v2 choose scores using only previously purchased observations.
- The oracle is allowed to inspect the complete existing score matrix when
  choosing a final allocation.
- Both use the same fixed score table, item order, score bounds, statistical
  threshold, and definition of an issued pairwise decision.

`oracle_protocol.json` fixes the five development repetitions and computation
budget before their oracle results. This document specifies the mathematical
formulation independently of whether a numerical solve reaches optimality.
The real-data oracle's expanded-prefix acquisition class retains the original
four-row-per-system warmup; it expands only acquisition choices after warmup.

## 1. Fixed inference and the count-state representation

Let the fixed bounded score matrix, already arranged in the shared randomized
item order, be

\[
Y\in[0,1]^{N\times K}.
\]

There are `P = K(K-1)/2` system pairs. System `s` may reveal only a prefix of
length `c_s` in this order. Define

\[
S_s(c)=\sum_{i=1}^{c}Y_{i,s},\qquad S_s(0)=0.
\]

Every revealed system-item cell costs one unit, even if it is reused for several
pairs. Thus an allocation has total cost

\[
C(c)=\sum_{s=1}^{K}c_s.
\]

The real-data replay counts existing mean-score cells. Each such cell represents
two historical human ratings; this is not measured annotation time or a new
human experiment.

The oracle may optimize the final prefix counts, but **not the shared item
permutation, the stopping threshold, or the score transformation**.

### Paired-prefix betting decisions

For pair `(a,b)`, let `tau_ab` be the first paired-prefix length at which exactly
one historical directional mixture peak crosses the frozen threshold, including
the controller's numerical guards. Use `N+1` if no such crossing occurs.

The threshold is the original global value

\[
\log(2P/\alpha),
\]

not the value obtained by treating each pair as a two-system experiment. The
fractions, conditional-null formula, mixture, and peak updates are those in
`pairwise_racing.py` and `METHOD.md`.

The first usable crossing must persist in the oracle's geometry, even if both
directions would cross later in a hypothetical continued calculation. In the
actual controller, an issued decision is frozen. Its betting-resolution
condition is therefore

\[
\min(c_a,c_b)\ge\tau_{ab}.
\]

The offline calculation advances each pair's statistical path independently of
global scheduling. If a particular schedule would already have resolved a pair
by deterministic completion, the pair is nevertheless resolved; this does not
invalidate the resolution-count characterization below.

### Unequal-prefix deterministic completion

Known score bounds imply the system-mean interval

\[
\mu_s\in\left[\frac{S_s(c_s)}{N},
\frac{S_s(c_s)+N-c_s}{N}\right].
\]

Consequently, the pair's completion interval is

\[
L_{ab}(c_a,c_b)
=\frac{S_a(c_a)-S_b(c_b)-(N-c_b)}{N},
\]

\[
U_{ab}(c_a,c_b)
=\frac{S_a(c_a)+(N-c_a)-S_b(c_b)}{N}.
\]

The deterministic rule resolves the pair if the lower endpoint is above the
frozen positive guard or the upper endpoint is below its negative counterpart.
These bounds use **both actual prefix lengths**, not only their minimum.
Omitting unequal-prefix completion can overstate an optimum and invalidate a
purported lower-bound conclusion.

### Census and the resolved predicate

Both `c_a=N` and `c_b=N` resolve the pair, including an exact tie, by the existing
census convention. Define the Boolean predicate

\[
R_{ab}(c_a,c_b)
=\mathbf 1\!\left\{
\begin{array}{l}
\min(c_a,c_b)\ge\tau_{ab},\ \text{or}\\
L_{ab}(c_a,c_b)>\text{tol},\ \text{or}\\
U_{ab}(c_a,c_b)<-\text{tol},\ \text{or}\\
c_a=c_b=N
\end{array}\right\}.
\]

In exact arithmetic, this resolved region is upward closed: additional prefix
scores cannot undo a prior resolution. `build_pair_geometry` verifies numerical
monotonicity and fails if the computed grid violates it. It does not silently
repair the grid and continue claiming an exact representation.

The predicate concerns whether a decision has been issued. It does not assert
that every such decision is correct: the original cost90 metric also counts
issued decisions and assesses wrong certificates separately.

## 2. Binary optimization formulation

Introduce binary prefix indicators

\[
x_{s,r}=\mathbf 1[c_s\ge r],\qquad r=1,\ldots,N.
\]

Enforce prefix consistency:

\[
x_{s,r+1}\le x_{s,r}.
\]

Their sum recovers the cost exactly:

\[
C=\sum_{s=1}^{K}\sum_{r=1}^{N}x_{s,r}.
\]

For each pair, define its downward-closed unresolved set

\[
\mathcal U_{ab}=\{(u,v):R_{ab}(u,v)=0\},
\]

and retain only its coordinatewise maximal points
`U_ab_max`. Introduce a binary pair-inclusion variable `z_ab`. For every maximal
unresolved point `(u,v)`, impose

\[
z_{ab}\le x_{a,u+1}+x_{b,v+1},
\]

where an indicator at `N+1` is interpreted as zero. Finally, require

\[
\sum_{a<b}z_{ab}\ge Q,
\qquad Q=\lceil0.9P\rceil,
\]

and minimize `C`. In the specified real experiment, `P=190` and `Q=171`.

It is harmless for a feasible `z_ab` to omit an additional pair that happens to
be resolved: only a minimum number of decisions is required.

### Equivalence proof

For any unresolved count state `(c_a,c_b)`, finiteness of the count grid implies
that some maximal unresolved point `(u,v)` dominates it. Both indicators on the
right-hand side of that point's constraint are zero. The model must therefore
set `z_ab=0`.

For a resolved count state, no unresolved frontier point can dominate it,
because the unresolved region is downward closed. At least one of
`c_a>u` or `c_b>v` holds for every frontier point. Every corresponding right-hand
side is therefore at least one, so `z_ab=1` is permitted.

Thus the constraints permit selecting a pair **if and only if** that pair is
resolved at the chosen prefix counts. Together with the prefix indicators and
target constraint, the formulation exactly minimizes final-prefix cost in the
stated broad acquisition class.

For implementation, let `v(u)` be the largest unresolved second count at first
count `u`, or `-1` if no unresolved count remains in that row. This boundary is
nonincreasing. Its maximal unresolved points are exactly those `(u,v(u))` with
`v(u)>=0` and either `u=N` or `v(u+1)<v(u)`.

## 3. Warmup and action-class qualifications

The deployable procedures count a four-row common warmup. Before enforcing
`c_s>=4`, the real-data diagnostic checks that the all-counts-four state resolves
fewer than `Q` pairs. By monotonicity, no earlier warmup state can already reach
the target. The original policies, which follow this common warmup, must then
complete it before reaching cost90. This justifies the count floor for the
comparison with those policies. It does not show that an arbitrary prefix
policy allowed to skip warmup must reach four observations on every system.

If this guard fails, imposing the floor could overstate the minimum cost. A
different study must explicitly handle the target's earlier fixed-warmup cost,
or relax the warmup restriction. The present real-data protocol does not silently
substitute another treatment.

The oracle class **requires the same four-row-per-system common warmup, then
allows arbitrary per-system prefix increments**. It is a relaxation of the
particular pair-bundle actions in v1/v2 while retaining that warmup requirement.
It is not the class of all prefix policies without a warmup constraint. Let
`C_broad_star` be the optimum in this expanded, warmup-retaining class and
`C_bundle_star` the optimum under the narrower action restrictions with the same
warmup. Then

\[
C^*_{\rm broad}\le C^*_{\rm bundle}\le C_{\rm v2}.
\]

A lower bound for the broad class is consequently also a lower bound for the
narrower class. However, a cheap broad-class incumbent does not prove that the
same cost can be achieved by a v2-style bundle scheduler, much less by a
deployable policy without access to unobserved scores.

## 4. Numerical solve, replay, and bound interpretation

`solve_oracle` uses SciPy's mixed-integer optimizer with HiGHS, a fixed time
limit, and a requested zero relative optimality gap. A time-limited run is not
necessarily optimal.

For each fixed development item order, report separately:

- The solver status, message, and numerical optimality gap.
- Its reported dual lower bound `L`.
- A feasible incumbent allocation and cost `U`, when available.
- The paired, actually observed v2 cost90 `C_v2`.

The implementation conservatively rounds the solver's lower bound down, with
a small numerical subtraction, and retains the trivial counted-warmup bound.
This is a **standard numerical solver-reported bound**, not an independently
verified exact-rational branch-and-bound proof. The combinatorial equivalence
above is a mathematical argument; it does not itself certify every floating-
point optimization step.

Every returned incumbent must satisfy integer and prefix constraints, match its
reported objective, and resolve at least `Q` pairs according to the geometry.
`replay_allocation` then purchases its actual cells through the original
`RaceState` under a common-row schedule with the selected terminal counts.
This checks the achieved decision count and cost using actual inference rather
than using the oracle predicate as the sole validator. Earlier target crossing
during that replay should be recorded, not charged as though all terminal cells
were necessary.

Subject to the model and numerical qualifications, the reported bounds give

\[
L\le C^*_{\rm broad}\le U.
\]

In particular, the maximum possible further cost reduction relative to v2 is
at most

\[
1-\frac{L}{C_{\rm v2}},
\]

within this fixed-inference, shared-prefix setting with the retained common
warmup. A tight lower bound can
therefore rule out a large scheduling-only gain **on the tested item order**.
A weak lower bound is inconclusive: it does not demonstrate that a cheap
allocation exists. A replayed low-cost incumbent supplies that separate
evidence of opportunity in the broad oracle class.

Per-order bounds can be averaged across the five fixed runs, but such an
average remains conditional on these development runs. It is not a bound for
every possible random order, future language pair, or new human population.

## 5. What is deliberately excluded

- No transitive pair certificates are added. The actual frozen controller does
  not infer `A>C` solely from certificates for `A>B` and `B>C`.
- Incorrect certificates are not removed inside the optimization. Doing so
  would change the original resolution-count objective. Their correctness is
  audited separately against the fixed full-matrix truth.
- No threshold, error allocation, score bound, or exact-tie convention changes.
- No optimization of item order, acquisition from a non-prefix position, or
  treatment of a reused score as a new independent observation.
- No hidden-label access is added to either deployable algorithm. Full-matrix
  access is restricted to the explicitly labeled offline diagnostic.
- No claim about new raters, latent quality, future documents, publication
  readiness, or general superiority over established methods.

The diagnostic identifies possible development directions. A tight bound near
the current cost suggests examining the inference requirements or research
objective. A much cheaper feasible allocation suggests further acquisition
work may be worthwhile. Neither outcome automatically supplies a new method.

## 6. Correctness checks

The accompanying tests and audits should cover:

1. Exhaustive count-vector enumeration on small matrices versus the MILP.
2. Frontier-clause equivalence at every small-grid count state.
3. Unequal-prefix completion, census-only ties, first-crossing persistence, and
   contradictory directional crossings.
4. Use of the original global pair threshold.
5. The counted-warmup guard and any explicitly supported exceptional case.
6. Numerical monotonicity, integer/prefix feasibility, and cost accounting.
7. Actual-inference replays of returned incumbents, including wrong-decision
   counts and any earlier target crossing.

These checks validate the diagnostic's implementation and stated optimization
scope. They do not replace the stopping-rule argument in `METHOD.md` or turn
development data into an independent test set.
