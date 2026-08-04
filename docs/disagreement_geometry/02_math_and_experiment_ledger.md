# Mathematical Toolkit and Experiment Ledger

## 1. Notation

Let:

- \(i,j\in\{1,\ldots,N\}\) index items;
- \(c\in\{1,\ldots,C\}\) index labels;
- \(v_i=(v_{i1},\ldots,v_{iC})\) be human vote counts;
- \(M_i=\sum_c v_{ic}\);
- \(p_i=v_i/M_i\) be the empirical human distribution;
- \(q_i^{(m)}\) be model \(m\)'s predicted distribution.

For ChaosNLI:

\[
C=3,\qquad M_i=100.
\]

## 2. Posterior human uncertainty

Use a Dirichlet posterior:

\[
\theta_i\mid v_i\sim
\operatorname{Dirichlet}(v_i+\alpha).
\]

This distinguishes uncertainty about a finite observed vote vector from the underlying collective distribution.

### Suggested reporting

For each item:

- posterior mean;
- posterior entropy distribution;
- credible interval for each label;
- posterior probability of majority-label change;
- posterior stability of neighbors and prototypes.

## 3. Pointwise geometry

### 3.1 Hellinger distance

\[
H(p,q)
=
\frac{1}{\sqrt2}
\left\|
\sqrt p-\sqrt q
\right\|_2.
\]

Advantages:

- bounded;
- symmetric;
- metric;
- naturally connected to the square-root simplex;
- handles zero components.

### 3.2 Jensen–Shannon divergence

\[
\operatorname{JSD}(p,q)
=
\frac12\operatorname{KL}(p\|m)
+
\frac12\operatorname{KL}(q\|m),
\quad
m=\frac{p+q}{2}.
\]

Specify whether it is in bits or nats and whether the square root is taken.

### 3.3 Fisher–Rao distance

For categorical distributions, the square-root map places probabilities on a sphere. A common geodesic form is

\[
d_{\mathrm{FR}}(p,q)
=
2\arccos\left(\sum_c\sqrt{p_cq_c}\right).
\]

This supplies a differential-geometric interpretation of the Hellinger/Bhattacharyya relationship.

### 3.4 Aitchison distance

For strictly positive compositions,

\[
d_A(p,q)
=
\left[
\frac{1}{2C}
\sum_{c=1}^C\sum_{d=1}^C
\left(
\log\frac{p_c}{p_d}
-
\log\frac{q_c}{q_d}
\right)^2
\right]^{1/2}.
\]

Aitchison geometry emphasizes relative ratios. It requires careful zero handling.

### 3.5 Optimal transport

If labels possess a meaningful ground distance \(D_{cd}\), use Wasserstein distance:

\[
W(p,q)
=
\min_{\gamma\in\Pi(p,q)}
\sum_{c,d}\gamma_{cd}D_{cd}.
\]

For NLI labels, the ground geometry is not automatically obvious. Any label cost must be justified rather than assumed.

## 4. Human support graph

### 4.1 Posterior expected fuzzy support

For each posterior draw \(b\), form a tie-aware \(k\)-nearest-neighbor graph \(W^{(b)}\). Define

\[
S_{ij}
=
\mathbb{E}_b[W_{ij}^{(b)}].
\]

Interpretation:

\[
S_{ij}
=
\text{posterior probability or expected weight that \(j\) is a human-supported neighbor of \(i\)}.
\]

### 4.2 Model graph

Construct a tie-aware model graph \(W^{(m)}\) from \(q_i^{(m)}\).

Each row has total mass \(k\):

\[
\sum_j W_{ij}^{(m)}=k.
\]

### 4.3 Relational support

\[
Q_{\mathrm{support}}^{(m)}
=
\frac{1}{Nk}
\sum_{i,j}
W_{ij}^{(m)}S_{ij}.
\]

This measures whether model-selected neighbors are supported by the human posterior.

## 5. Nulls and normalized recovery

### 5.1 Dataset-stratified identity null

Permute item identities within declared strata while preserving the model graph.

The null must use the same:

- focal rows;
- folds;
- \(k\);
- graph condition;
- strata;
- item order.

### 5.2 Analytic blockwise null

For block labels \(g(i)\), the expected null can be computed from model edge-mass density and target support mass within each block pair.

This is efficient for enumerating many coalitions.

### 5.3 Human-normalized recovery

\[
R_m
=
\frac{
Q_m-Q_{\mathrm{null},m}
}{
Q_{HH}-Q_{\mathrm{null},m}
}.
\]

Interpretation:

> Fraction of split-half human-reference excess support recovered by the model.

It is not mathematically bounded by 100% because the split-half statistic is a reliability reference, not an absolute maximum.

## 6. Conditional-null ladder

Let \(N_\ell\) denote increasingly restrictive identity permutations.

A strictly nested ladder may be:

\[
\begin{aligned}
N_0 &: \text{global},\\
N_1 &: \text{dataset},\\
N_2 &: N_1+\text{majority label},\\
N_3 &: N_2+\text{entropy quintile},\\
N_4 &: N_3+\text{runner-up label}+\text{margin bin},\\
N_5 &: N_4+\text{exact vote profile}.
\end{aligned}
\]

Define residual excess:

\[
E_{m,\ell}
=
Q_m-\mathbb{E}_{N_\ell}[Q_m].
\]

Define residual fraction:

\[
F_{m,\ell}
=
\frac{E_{m,\ell}}{E_{m,0}}.
\]

### Interpretation

- Large \(F_{m,4}\): model retains item-specific relational structure beyond coarse ambiguity summaries.
- Near-zero \(F_{m,4}\): model alignment is almost entirely explained by label and ambiguity type.
- Non-informative \(N_5\): exact-profile groups contain too few movable items.

Always report:

- number of groups;
- non-singleton groups;
- movable items;
- largest group;
- null interval;
- raw and adjusted \(p\)-values.

## 7. Graph overlap and turnover

### 7.1 Fuzzy min-overlap

\[
\operatorname{Overlap}_{\min}(W^A,W^B)
=
\frac{1}{Nk}
\sum_{ij}
\min(W_{ij}^A,W_{ij}^B).
\]

Turnover:

\[
\operatorname{Turnover}_{\min}
=
1-\operatorname{Overlap}_{\min}.
\]

This guarantees zero turnover for an identical fuzzy graph.

### 7.2 Edge persistence

For a family of conditions \(t\),

\[
P_{ij}
=
\frac{1}{|T|}
\sum_{t\in T}
\mathbf{1}[W_{ij}(t)>0].
\]

Useful for identifying stable human or model analogies.

### 7.3 Core recall

Define a high-support core with threshold \(\tau\):

\[
\mathcal{C}
=
\{(i,j):S_{ij}\ge\tau\}.
\]

Measure the fraction of model mass assigned to this core, using a target constructed at the same graph scale.

## 8. Prototype-equivalent complexity

Fit \(K\) prototypes to training human distributions in square-root space.

For held-out item \(i\):

1. assign it to the nearest training prototype;
2. reconstruct \(\hat p_i^{(K)}\);
3. build a coherent prototype graph;
4. score held-out focal rows.

This gives a rate–distortion curve:

\[
K\mapsto R_{\mathrm{proto}}(K).
\]

### 8.1 Discrete effective prototype equivalent

\[
K_{\mathrm{eff}}(m)
=
\min\{K:R_{\mathrm{proto}}(K)\ge R_m\}.
\]

### 8.2 Interpolated effective bits

Let \(K_L<K_U\) bracket \(R_m\). Interpolate in bit space:

\[
\lambda
=
\frac{R_m-R(K_L)}{R(K_U)-R(K_L)},
\]

\[
b_{\mathrm{eff}}(m)
=
\log_2K_L
+
\lambda
\left(
\log_2K_U-\log_2K_L
\right).
\]

Effective states:

\[
\widetilde K_{\mathrm{eff}}=2^{b_{\mathrm{eff}}}.
\]

### Claim boundary

This is an **external relational-performance equivalence**. It is not:

- internal model entropy;
- mutual information;
- number of literal concepts;
- storage capacity.

## 9. Empirical-vote support retained

Let \(Q_{\mathrm{emp}}\) be support achieved by the empirical-vote graph.

\[
C_K
=
\frac{
Q_{\mathrm{proto}}(K)-Q_{\mathrm{null}}(K)
}{
Q_{\mathrm{emp}}-Q_{\mathrm{null,emp}}
}.
\]

Interpretation:

> Fraction of the empirical-vote graph’s null-adjusted support retained by a \(K\)-prototype representation.

## 10. Ensemble attribution

For model set \(M\) and coalition \(A\subseteq M\), define value \(v(A)\).

The exact Shapley contribution of model \(m\) is

\[
\phi_m
=
\sum_{A\subseteq M\setminus\{m\}}
\frac{|A|!(|M|-|A|-1)!}{|M|!}
\left[
v(A\cup\{m\})-v(A)
\right].
\]

Required efficiency check:

\[
\sum_m\phi_m=v(M)-v(\varnothing).
\]

Useful value functions:

- relational recovery;
- raw excess support;
- NLL reduction from a declared prior;
- empirical support retained;
- effective bits, with caution because interpolation adds nonlinearity.

## 11. Collective-twin overlap

Assume groups \(A\) and \(B\) answer the same items.

### 11.1 Pointwise distribution overlap

\[
O_{\mathrm{point}}
=
1-\frac{1}{N}\sum_i H(p_i^A,p_i^B).
\]

Scale or transform as needed for interpretability.

### 11.2 Identity-aligned relational overlap

\[
O_{\mathrm{rel}}
=
\frac{1}{Nk}
\sum_{ij}
\min(W_{ij}^A,W_{ij}^B).
\]

### 11.3 Cross-support

\[
Q_{A\rightarrow B}
=
\frac{1}{Nk}
\sum_{ij}
W_{ij}^A S_{ij}^B.
\]

This need not be symmetric.

### 11.4 Missing-region recall

For a high-support region or prototype set \(\mathcal R_A\) in group \(A\):

\[
\operatorname{Recall}_{B\leftarrow A}
=
\frac{
\text{mass in \(A\)'s region represented by \(B\)}
}{
\text{mass of \(A\)'s region}
}.
\]

This is a candidate measure of underrepresentation.

### 11.5 Gromov–Wasserstein structural overlap

When item identity is unavailable or spaces differ, compare internal distance matrices:

\[
\operatorname{GW}(A,B)
=
\min_{\gamma\in\Pi(\mu_A,\mu_B)}
\sum_{i,j,k,l}
|D^A_{ij}-D^B_{kl}|^2
\gamma_{ik}\gamma_{jl}.
\]

Fused GW can combine structure with observable features.

### Claim boundary

Low GW distance means similar internal shape under some coupling. It does not prove shared semantics.

## 12. Polarization and fragmentation measures

Entropy alone is insufficient.

Potential decomposition:

\[
\text{polarization}
=
\text{within-cluster identification}
\times
\text{between-cluster alienation}.
\]

Candidate operational quantities:

- cluster silhouette;
- between/within distance ratio;
- Esteban–Ray-style mass-and-distance index;
- modularity;
- spectral gap;
- number and persistence of modes;
- posterior probability of multiple clusters.

Use annotator-level or group-level observations where possible.

## 13. Uncertainty and inference

### 13.1 Stratified focal-item bootstrap

Resample focal items within:

- dataset;
- majority label;
- entropy quintile.

Use common resamples across conditions and models.

### 13.2 Human posterior uncertainty

Repeat the graph construction across posterior draws or batches.

### 13.3 Model-selection uncertainty

For ensemble search:

- training-only coalition selection;
- held-out focal-row scoring;
- bootstrap selection frequency;
- near-optimal coalition sets.

### 13.4 Prototype uncertainty

Report:

- fold variability;
- restart variability;
- sample-bootstrap variability;
- occupancy;
- empty clusters;
- tie fraction;
- distortion;
- monotonicity.

## 14. Experiment ledger

| Concept | Mathematical object | Established lineage | Immediate experiment | Claim boundary |
|---|---|---|---|---|
| Amount vs type of disagreement | Entropy plus simplex location | Information theory; compositional geometry | Compare entropy-matched ambiguity types | Does not identify semantic reason |
| Reliability | Variance components; split-half graph | Generalizability theory | Vote-budget saturation | Reliability is not validity |
| Shared culture | Agreement matrix eigenstructure | Cultural consensus | Multiple-consensus mixture on group data | One-consensus assumptions may fail |
| Viewpoint types | Person-by-item factors | Q methodology | Factor repeated annotators | Requires repeated identities |
| Latent ideology | Person/item coordinates | IRT; ideal points | Joint annotator–item embedding | Axes require substantive interpretation |
| Aggregation loss | Majority map and logical constraints | Judgment aggregation | Compare full distributions vs majority graph | Loss may be structural, not model error |
| Polarization | Cohesion × separation | Esteban–Ray; opinion dynamics | Group-aware cluster metrics | Aggregate 50/50 is not proof of camps |
| Relational recovery | \(Q_{\mathrm{support}},R\) | Graph comparison | E001–E004 | Depends on metric, \(k\), and target |
| Conditional resolution | Nested null ladder | Conditional randomization | E005 full | Sequential, not unique causal variance |
| Complementarity | Coalition value and Shapley | Cooperative game theory | E007 full and held-out selection | Shapley depends on value function |
| Compressibility | \(R(K),C_K,b_{\mathrm{eff}}\) | Rate–distortion; vector quantization | E008 full | External equivalence, not internal bits |
| Calibration risk | \(\Delta\)NLL, turnover, \(\Delta R\) | Calibration; proper scoring | E002/E004/E009 | Improving one score is not universal calibration |
| Group overlap | Cross-support and missing-region recall | Cultural models; OT | OpinionQA/PRISM geometry | Group labels are correlational |
| Shape without correspondence | GW/FGW | Optimal transport | Cross-dataset structure comparison | Similar shape need not mean same meaning |
| Minority-view preservation | Region recall | Pluralistic alignment | Demographic datasets | Do not infer demographics from ChaosNLI |
| Annotation allocation | Expected graph stabilization | Active learning; reliability | E006 | Stopping rules depend on target use |
| Geometry-preserving distillation | Distance/edge loss | Metric learning; distillation | New method | Must preserve held-out geometry |

## 15. Priority mathematical additions

1. Analytic and Monte Carlo validation of all nulls.
2. Multi-scale \(k\) analysis with scale-matched human targets.
3. Group-aware cross-support and missing-region metrics.
4. Effective-bit uncertainty intervals.
5. Geometry-preserving distillation loss.
6. GW/FGW pilot on coarsened graphs.
7. Variance-component decomposition of graph uncertainty.
8. Longitudinal geometry only after repeated-time data exist.
