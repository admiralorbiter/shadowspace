# Collective Opinion as a Relational Space: Tie-Aware Neighborhood Analysis of Human and Model NLI Distributions

- **Document type:** paper draft
- **Status:** release candidate RC1; ready for external circulation pending four final label fixes (now resolved)
- **Scope:** ChaosNLI computational audit, reference-graph similarity, and formal tie mathematics

---

## Abstract

Human NLI annotations can exhibit persistent and substantively meaningful variation, alongside annotation error. Conventional majority-label evaluation ignores this variation. We study whether collective human disagreement, encoded as vote distributions over semantic labels, forms a reproducible relational structure that NLI models recover. Methodologically, we demonstrate that conventional index-resolved fixed-$k$ neighborhood graphs are unstable under distance ties ($0.9554 \pm 0.0015$ top-$k$ overlap under array re-indexing; 49.1% of items affected at $k=10$), whereas our fractional soft-overlap statistic $Q_{NX}^{\text{soft}}(k)$ is strictly invariant (maximum absolute difference $0.0000$). We formalize a three-quantity tie-aware framework ($Q_{\text{strict}} \le Q_{\text{expected}} \le Q_{\text{fuzzy}}$) and prove six core theoretical properties. Empirically, across $N=3,113$ ChaosNLI items under a fully paired design where both model and human scores are evaluated against identical posterior-predictive cohorts, benchmark NLI models recover substantially less human-opinion neighborhood structure than human replicates (BART-Large paired mean $0.01572$ vs. posterior-predictive human benchmark $\bar{H}=0.07549$, mean difference $\Delta_m = 0.05977$, 95% CI $[0.05431, 0.06539]$; all 1,000 bootstrap replicates show $\Delta_m > 0$). Plug-in empirical reference similarity $R_{\text{reference}}(n, k)$ increases monotonically with annotation depth across all five tested neighborhood scales (50-seed simulation; simulation interval lower bounds monotone for all $k \in \{5, 10, 20, 50, 100\}$), from $0.0109 \pm 0.0002$ at $n=3$ to $0.1391 \pm 0.0033$ at $n=100$ ($k=10$). The corrected boundary definition finds exact ties crossing the empirical $k=10$ boundary for $49.1\%$ of items; multi-regime Dirichlet simulations place that value between the tested $\alpha=0.1$ and $\alpha=0.5$ regimes at 100 votes.

---

## 1. Introduction

Natural language inference — determining whether a premise entails, contradicts, or is neutral toward a hypothesis — is a foundational benchmark for language models. Standard evaluation assigns each item a single majority label, assuming annotation disagreement reflects random noise. Yet extensive evidence demonstrates that NLI items often exhibit genuine, persistent human disagreement arising from multiple valid interpretations (Pavlick and Kwiatkowski, 2019; Nie et al., 2020; Jiang and de Marneffe, 2022).

The ChaosNLI dataset (Nie et al., 2020) provides 100 human judgments for each of 3,113 selected low-original-agreement NLI examples, yielding an empirical probability distribution over {*entailment*, *neutral*, *contradiction*}. Prior evaluation compares model and human distributions *pointwise* using Jensen–Shannon divergence or Earth Mover's Distance (Zhou et al., 2022; Wang et al., 2022; Baan et al., 2022). Pointwise evaluation asks: "does model $m$ match the human vote distribution for item $i$?"

We investigate a complementary, relational question: **does model $m$ recover the neighborhood structure among human opinion distributions?** That is, among examples that humans judge similarly, do models assign similar probability distributions? And among examples humans distinguish, do models distinguish them?

This relational framing evaluates whether a model's **output probability distributions** preserve the *relational organization of collective judgment patterns*. It also directly informs tools like Shadowspace that generate structured views of opinion distributions for diagnostic review.

Section 3 formalizes the tie-aware overlap framework and proves six core properties. Section 4 benchmarks nine NLI models against human posterior-predictive replicates, characterizes reference similarity scaling with annotation depth, and maps tie-prevalence regimes. Section 5 presents a secondary exploratory analysis on matched VariErr items. Section 6 gives full reproducibility specifications; Sections 7–8 discuss limitations and conclusions.

---

## 2. Related Work

### 2.1 Co-Ranking, $Q_{NX}$, and Multiscale Neighborhood Preservation

Nonlinear dimensionality reduction and manifold learning rely on rank-based neighborhood preservation metrics. Lee and Verleysen (2008) introduced the co-ranking matrix and $Q_{NX}(k)$ to measure neighbor agreement across spaces while avoiding distance scale distortions. Chen and Buja (2009) introduced local multidimensional scaling and the Local Continuity Meta-Criterion ($\text{LCMC}(k) = Q_{NX}(k) - \frac{k}{N-1}$) to adjust for random chance, which Lee and Verleysen (2009) incorporated into the co-ranking framework and Lueks et al. (2011) extended for error visualization. However, existing $Q_{NX}$ definitions assume continuous feature spaces with strict orderings ($P(d_{ij} = d_{ik}) = 0$). When applied to finite categorical distributions, continuous co-ranking fails because discrete lattice grids create large tie blocks.

### 2.2 Related Methodological Connections

Our tie-aware relational framework connects to three broader lines of research:

1. **Uncertain Nearest-Neighbor Search**: In stochastic data settings, pairwise distances are accessed through noisy estimates (Mason et al., 2019). Our setting frames annotation acquisition as reducing neighborhood uncertainty, previewing the posterior uncertain graph membership probability $s_{ij} = P(j \in N_k(i) \mid \text{votes})$.
2. **Fuzzy Human-Label Evaluation**: Prior HLV work surveys soft-label and disaggregated-label approaches (Uma et al., 2021), while Kurniawan et al. (2026) introduce fuzzy-set-based item-level evaluation metrics. We extend fuzzy evaluation from item-label relationships to relational neighborhoods induced among empirical label distributions.
3. **Graph Estimator Stability**: Following graph perturbation analyses (e.g., Mapper-type algorithm parameter sensitivity reviewed in Chazal and Michel, 2021), we treat variability in a derived neighborhood graph as an informative property of empirical opinion spaces rather than an implementation artifact.

Tied-rank correlations (Kendall's $\tau_b$, Spearman's $\rho$ with tie corrections) handle discrete ties in linear rankings but do not extend to multiscale set-based $k$-nearest-neighbor graphs. Fuzzy set theory (Zadeh, 1965) models partial set membership using minimum operators ($\min(w_A, w_B)$). In statistical geometry, compositional data (Aitchison, 1982) and information geometry (Amari, 2000; Endres and Schindelin, 2003) provide metric distances for probability simplices (Hellinger, Jensen–Shannon). We synthesize co-ranking, fuzzy-set membership, and information geometry into a tie-aware neighborhood framework.

### 2.3 Human Disagreement in NLI and Finite-Rater Reliability

Pavlick and Kwiatkowski (2019) established that NLI disagreement reflects genuine interpretive variation. Nie et al. (2020; ChaosNLI) and Plank (2022) argued majority-vote calibration is ill-defined under label variation. Baan et al. (2022) demonstrated that standard calibration metrics become unreliable when humans genuinely disagree. Gruber et al. (2024) demonstrated that annotation depth better recovers latent class boundaries than breadth. Weber-Genzel et al. (2024; VariErr NLI) introduced 7,732 validity judgments over 500 re-annotated MNLI items to separate valid human variation from annotation error.

---

## 3. Formal Tie Mathematics: The Three-Quantity Interval

### 3.1 Boundary Weight Construction

For focal item $i$ and target rank $k$, let $A_i$ be the set of candidate neighbors strictly closer than distance $d_i(k)$, $B_i$ be the set of candidates tied at distance $d_i(k)$, and $r_i = k - |A_i|$ be the remaining slots. The fractional tie-aware weight $w_{ij}$ assigned to candidate neighbor $j$ is:

$$w_{ij} = \begin{cases} 1, & d_{ij} < d_i(k) \\ \frac{r_i}{|B_i|}, & d_{ij} = d_i(k) \\ 0, & d_{ij} > d_i(k) \end{cases}$$

### 3.2 The Three-Quantity Overlap Family

For candidate weights $w_{ij}^A, w_{ij}^B \in [0, 1]$, we formalize three distinct neighborhood overlap quantities:

1. **$Q_{\text{strict}}$ (Strict-Core Lower Bound)**: Counts only non-tied core boundary neighbors:
   $$Q_{\text{strict}}(k) = \frac{1}{N k} \sum_{i=1}^N \sum_{j \ne i} \mathbf{1}(w_{ij}^A=1) \mathbf{1}(w_{ij}^B=1)$$

2. **$Q_{\text{expected}}$ (Expected Random-Tie Overlap)**: Represents independent uniform random resolution of boundary ties:
   $$Q_{\text{expected}}(k) = \frac{1}{N k} \sum_{i=1}^N \sum_{j \ne i} w_{ij}^A w_{ij}^B$$

3. **$Q_{\text{fuzzy}}$ (Min-Based Fuzzy Membership Overlap)**: Treats boundary weights as partial set membership:
   $$Q_{\text{fuzzy}}(k) = \frac{1}{N k} \sum_{i=1}^N \sum_{j \ne i} \min(w_{ij}^A, w_{ij}^B)$$

### 3.3 Theoretical Proof and Six Fundamental Properties

**Theorem (Three-Quantity Overlap Inequality)**: For any two weighted neighborhood graphs with weights satisfying $w_{ij}^A, w_{ij}^B \in [0, 1]$ and $\sum_{j \ne i} w_{ij}^A = \sum_{j \ne i} w_{ij}^B = k$ for every focal item $i$, the strict-core lower bound, expected random-tie overlap, and fuzzy membership overlap satisfy:

$$Q_{\text{strict}} \le Q_{\text{expected}} \le Q_{\text{fuzzy}} \le 1.0$$

*Proof*: For any $x, y \in [0, 1]$, $\mathbf{1}(x=1)\mathbf{1}(y=1) \le xy \le \min(x, y)$. Summing over candidate neighbors $j \ne i$ and scaling by $1/Nk$ preserves the point-wise inequalities. Furthermore, since each weighted neighborhood sums to $k$ ($\sum_{j \ne i} w_{ij} = k$), the fuzzy overlap numerator satisfies $\sum_{j \ne i} \min(w_{ij}^A, w_{ij}^B) \le \sum_{j \ne i} w_{ij}^A = k$, which ensures $Q_{\text{fuzzy}} \le 1.0$. $\blacksquare$

**Six Fundamental Properties**:
1. **Range**: All three quantities are bounded in $[0, 1]$.
2. **Symmetry**: $Q_\bullet(G^A, G^B) = Q_\bullet(G^B, G^A)$ for all three formulations.
3. **Fuzzy Identity**: $Q_{\text{fuzzy}}(G, G) = 1.0$ for any weighted neighborhood graph $G$.
4. **Expected and Strict Self-Overlap**: $Q_{\text{expected}}(G, G) = \frac{1}{Nk}\sum_{i}\sum_{j \ne i} w_{ij}^2 \le 1.0$, measuring collision probability under independent random tie resolutions. $Q_{\text{strict}}(G, G) = 1.0$ when every selected neighborhood membership has unit weight — that is, when there are no fractional boundary memberships at rank $k$.
5. **Row-Order Permutation Invariance**: For any permutation $\pi$, $Q_\bullet(G^A, G^B) = Q_\bullet(\pi G^A, \pi G^B)$ after matching persistent object identities. Across 1,000 random permutations, the maximum absolute pre/post-permutation difference was **0.0000**.
6. **Reduction to Standard $Q_{NX}$ Under Unique Boundary**: When every $k$-boundary distance is unique ($|B_i| = 1, r_i = 1$), candidate weights are binary ($w_{ij} \in \{0, 1\}$) and all three formulations reduce strictly to standard $Q_{NX}$.

### 3.4 Item-Level Permutation Damage Breakdown

To demonstrate how arbitrary array storage re-indexing affects deterministic top-$k$ neighbor sets across items, we analyze the item-level overlap distribution under 1,000 random row permutations after explicitly excluding self-distance:

**Table 1: Item-Level Permutation Overlap Breakdown Across Neighborhood Scales ($k$)**

| Scale ($k$) | Mean Overlap | Median Overlap | 5% – 95% Interval | Min Overlap | Items Changed (%) |
|---|---|---|---|---|---|
| $k=5$ | 0.9062 | 1.0000 | [0.6818, 1.0000] | 0.3760 | 49.5% |
| $k=10$ | **0.9554** | **1.0000** | **[0.8516, 1.0000]** | **0.7266** | **49.1%** |
| $k=20$ | 0.9763 | 0.9763 | [0.9200, 1.0000] | 0.7424 | 50.7% |
| $k=50$ | 0.9905 | 0.9902 | [0.9674, 1.0000] | 0.9287 | 51.6% |

*Takeaway*: At $k=10$, array re-indexing alters the top-$k$ neighbor set of **49.1% of items**, with minimum mean item overlap dropping to $0.7266$ across 1,000 permutations. Storage-order instability therefore affects nearly half of items in this finite vote lattice.

---

## 4. Study 1: Empirical Benchmark and Reference Analysis

Conventional index-resolved fixed-$k$ neighborhoods rely on array storage row order for tie-breaking. Across 1,000 random row permutations, deterministic top-$k$ overlap fluctuates ($0.9554 \pm 0.0015$, SD $0.0015$), whereas fractional soft overlap is strictly row-order invariant ($1.0000 \pm 0.0000$).

### 4.1 Nine-Model Hellinger Benchmark ($k=10$)

We evaluate nine benchmark NLI models against 500 posterior-predictive simulation pairs on $N=3,113$ ChaosNLI items under Hellinger distance at $k=10$. We use a **fully paired estimand** where both human reliability $H_b$ and model performance $M_{m,b}$ are evaluated symmetrically against identical simulated posterior cohorts ($G_{H1}^{(s)}, G_{H2}^{(s)}$). Focal-bootstrap means across 1,000 stratified replicates serve as point estimates; bootstrap is used for confidence intervals and $\Delta_m$ distributions. Fixed full-data reference scores $Q(G_m, G_{100}^{\text{obs}})$ are reported separately as a descriptive baseline:

**Table 2: Benchmark NLI Model Overlap Scores Under Fully Paired Estimand ($k=10$)**

| Model | Focal-Bootstrap Mean $\hat{M}_m$ | Mean $\Delta_m$ (vs. HH100) | 95% Bootstrap Interval | Replicates $\Delta_m > 0$ | Fixed-ref. $Q(G_m, G_{100}^{\text{obs}})$ |
|---|---|---|---|---|---|
| BART-Large | **0.01572** | **0.05977** | [0.05431, 0.06539] | 1,000 / 1,000 | 0.01867 |
| RoBERTa-Large | **0.01415** | **0.06135** | [0.05557, 0.06685] | 1,000 / 1,000 | 0.01821 |
| XLNet-Large | **0.01285** | **0.06264** | [0.05711, 0.06846] | 1,000 / 1,000 | 0.01319 |
| ALBERT-xxLarge | **0.01124** | **0.06426** | [0.05896, 0.06997] | 1,000 / 1,000 | 0.01074 |
| BERT-Large | **0.01029** | **0.06520** | [0.05966, 0.07076] | 1,000 / 1,000 | 0.01059 |
| RoBERTa-Base | **0.01007** | **0.06543** | [0.05979, 0.07106] | 1,000 / 1,000 | 0.01129 |
| XLNet-Base | **0.00927** | **0.06623** | [0.06069, 0.07175] | 1,000 / 1,000 | 0.00893 |
| DistilBERT | **0.00854** | **0.06695** | [0.06124, 0.07261] | 1,000 / 1,000 | 0.00854 |
| BERT-Base | **0.00768** | **0.06782** | [0.06235, 0.07356] | 1,000 / 1,000 | 0.00865 |
| **HH100 Focal-Bootstrap Mean** ($\bar{H}$) | **0.07549** | — | [0.07000, 0.08099] | — | — |

*Methods and Inference Note*: All paired point estimates ($\hat{M}_m$ for models, $\bar{H}$ for the human reference) are **focal-bootstrap means** across 1,000 stratified replicates — the mean of $M_{m,b}$ and $H_b$ respectively over 1,000 bootstrap draws. These closely approximate the corresponding direct 500-pair means within Monte Carlo error: the direct human mean $\bar{H}_{\text{direct}} = \frac{1}{500}\sum_s H_s = 0.07550$ vs. focal-bootstrap mean $\bar{H} = 0.07549$ ($\Delta < 0.00001$). Direct 500-pair means for individual models are not separately stored in the canonical release; focal-bootstrap means serve as the primary point estimates throughout. Bootstrap is used exclusively for confidence intervals and $\Delta_m$ distributions. In 1,000 of 1,000 stratified joint bootstrap replicates, every model difference interval $\Delta_m$ comfortably excludes zero (minimum lower bound $0.05431$). Formally, for bootstrap replicate $b \in \{1,...,1000\}$, a stratified focal-item resample $\mathbf{b} \subset \{1..N\}$ is drawn. Let $s = b \bmod 500$ index one of 500 pre-computed posterior human pairs. We define:
$$H_b = \frac{1}{|\mathbf{b}|}\sum_{i \in \mathbf{b}} Q_{\text{fuzzy, item}}\!\left(G_{H1}^{(s)}, G_{H2}^{(s)}\right)$$
$$M_{m,b} = \frac{1}{|\mathbf{b}|}\sum_{i \in \mathbf{b}} \frac{1}{2}\left[Q_{\text{fuzzy, item}}\!\left(G_m, G_{H1}^{(s)}\right) + Q_{\text{fuzzy, item}}\!\left(G_m, G_{H2}^{(s)}\right)\right]$$
$$\Delta_{m,b} = H_b - M_{m,b}$$
$\hat{M}_m = \frac{1}{1000}\sum_b M_{m,b}$; $\bar{H} = \frac{1}{1000}\sum_b H_b$. Both human reliability $H_b$ and model score $M_{m,b}$ evaluate against the exact same two simulated posterior cohorts. Given the same simulated human cohorts, models resemble those cohorts substantially less than the cohorts resemble one another.

*Bootstrap Scope Statement*: The bootstrap estimates sampling variation across focal items within the fixed ChaosNLI candidate population. It does not reconstruct the neighbor graph from resampled rows or estimate uncertainty over an unselected NLI population. Edge contributions are non-independent through shared candidate nodes; this limitation is unlikely to alter the qualitative finding given the size of the observed gap.

### 4.2 Reference Graph Similarity Surface $R_{\text{reference}}(n, k)$ and Reference Ladder

To evaluate how rapidly an $n$-vote graph similarity recovers relative to the observed 100-vote graph $G_{100}^{\text{obs}}$, we compute **plug-in empirical reference similarity** $R_{\text{reference}}(n, k) = Q(G_n^{\text{rep}}, G_{100}^{\text{obs}})$, where $G_n^{\text{rep}}$ is an independent $n$-vote draw from the **plug-in multinomial** $\mathbf{y}_i \sim \text{Multinomial}(n, \hat{p}_i)$ using observed proportions $\hat{p}_i$ (not a posterior-predictive sample). A posterior-predictive surface $R_{\text{posterior}}(n, k)$ incorporating additional uncertainty over latent $\boldsymbol{\theta}_i$ will be reported in a follow-up revision.

To provide an interpretable anchor for model evaluation, we present a **Reference Ladder** comparing models, human replicates, and oracles against the observed human graph:

**Table 3: Reference Ladder of Graph Overlap ($k=10$)**

*Panel A: Model vs. Human Reference Overlap ($Q_{\text{fuzzy}}$, Fixed Reference)*

| Graph Comparison | Overlap Score ($Q_{\text{fuzzy}}$) | Interpretation |
|---|---|---|
| Empirical Stratified Null | 0.00354 | Mean overlap under stratified item-identity permutations ($k=10, N=3{,}113$) |
| BERT-Base vs. Observed | 0.00865 | Lowest-performing evaluated model |
| BART-Large vs. Observed | 0.01867 | Best-performing evaluated model |
| **Posterior-Predictive 100-Vote Replicate vs. Observed** | **0.13850** | **Human replicate reference: posterior cohort vs. observed graph** |
| Observed Graph Self-Overlap ($Q_{\text{fuzzy}}$) | 1.00000 | Exact fuzzy self-identity |

*Panel B: Tie-Interpretation Sensitivity Across 500 HH100 Reference Pairs ($k=10$)*

| Overlap Formulation | Simulation Mean | 95% Simulation Interval | Scientific Attitude |
|---|---|---|---|
| $Q_{\text{strict}}$ (Core Bound) | 0.00020 | [0.00018, 0.00023] | Guaranteed common neighbors only |
| $Q_{\text{expected}}$ (Random Resolution) | 0.07450 | [0.07011, 0.07907] | Collision probability under random tie choices |
| $Q_{\text{fuzzy}}$ (Partial Membership) — **$\bar{H}$** | **0.07549** | [0.07111, 0.08007] | Full partial-membership weighted overlap; canonical HH100 bootstrap mean |

*Note*: $\bar{H} = 0.07549$ is the bootstrap mean of $H_b$ over 1,000 replicates, consistent with the direct 500-pair simulation mean ($0.07550$, stored as $0.0755$ in canonical results). The Panel B 95% interval $[0.07111, 0.08007]$ is from the 500-pair simulation distribution. An earlier analysis version reported $0.07522$ for this cell; that value was from a different analysis path and is superseded by the canonical computation. The Panel A $0.13850$ value is computed from posterior-predictive 100-vote replicates (Dirichlet sampling) evaluated against $G_{100}^{\text{obs}}$. This differs from the plug-in surface value $0.1391 \pm 0.0033$ (Table 4, $n=100, k=10$), which uses observed proportions $\hat{p}_i$ directly without Dirichlet sampling.

*Takeaway*: The fixed-reference comparison shows that all evaluated models lie substantially closer to the empirical stratified null ($0.00354$) than to the posterior-predictive human reference ($0.13850$). Note that $0.13850$ is a single-draw estimate (posterior-predictive 100-vote cohort, seed 0 vs. observed graph) and should be treated as **illustrative** rather than a stable reference denominator; it is documented in the canonical results under `posterior100_vs_observed_seed0`. The primary benchmark result is the fully paired comparison in Table 2, where all 1,000 bootstrap replicates place $\Delta_m > 0$ for every evaluated model.

**Table 4: Reference Graph Similarity Surface $R_{\text{reference}}(n, k) = Q(G_n^{\text{rep}}, G_{100}^{\text{obs}})$ (50-Seed Simulation, Mean $\pm$ SD)**

| Votes ($n$) | $k=5$ | $k=10$ | $k=20$ | $k=50$ | $k=100$ |
|---|---|---|---|---|---|
| 3 | $0.0060 \pm 0.0001$ | $0.0109 \pm 0.0002$ | $0.0206 \pm 0.0004$ | $0.0490 \pm 0.0008$ | $0.0940 \pm 0.0014$ |
| 5 | $0.0082 \pm 0.0002$ | $0.0148 \pm 0.0003$ | $0.0280 \pm 0.0005$ | $0.0653 \pm 0.0011$ | $0.1216 \pm 0.0019$ |
| 10 | $0.0135 \pm 0.0005$ | $0.0242 \pm 0.0007$ | $0.0449 \pm 0.0009$ | $0.0999 \pm 0.0019$ | $0.1793 \pm 0.0032$ |
| 20 | $0.0230 \pm 0.0011$ | $0.0402 \pm 0.0013$ | $0.0726 \pm 0.0016$ | $0.1550 \pm 0.0023$ | $0.2620 \pm 0.0033$ |
| 30 | $0.0318 \pm 0.0013$ | $0.0550 \pm 0.0018$ | $0.0973 \pm 0.0021$ | $0.2009 \pm 0.0027$ | $0.3242 \pm 0.0035$ |
| 50 | $0.0474 \pm 0.0020$ | $0.0813 \pm 0.0018$ | $0.1424 \pm 0.0020$ | $0.2769 \pm 0.0033$ | $0.4136 \pm 0.0038$ |
| 75 | $0.0640 \pm 0.0023$ | $0.1106 \pm 0.0028$ | $0.1908 \pm 0.0034$ | $0.3502 \pm 0.0037$ | $0.4898 \pm 0.0033$ |
| **100** | $\mathbf{0.0807 \pm 0.0025}$ | $\mathbf{0.1391 \pm 0.0033}$ | $\mathbf{0.2341 \pm 0.0041}$ | $\mathbf{0.4080 \pm 0.0039}$ | $\mathbf{0.5448 \pm 0.0038}$ |

*Takeaway*: Reference similarity increases monotonically with annotation depth across all five tested neighborhood scales — a result confirmed for both column means and 95% normal-approximation simulation interval lower bounds ($\bar{x} \pm 1.96 \times \text{SD}$) across 50 independent simulation seeds (Rust/Rayon implementation). SD values are uniformly small ($\le 0.004$ across all cells), confirming low simulation variance. The annotation-depth monotonicity claim is robust to seed choice.

---

### 4.3 Phase Diagram: Tie Prevalence Under Synthetic Annotation Regimes

To model how annotation scale ($n$) and label granularity ($C$) shape boundary tie prevalence, we simulate synthetic items across Dirichlet concentration regimes ($\boldsymbol{\theta}_i \sim \text{Dirichlet}(\alpha \mathbf{1}_C)$, $\mathbf{y}_i \sim \text{Multinomial}(n, \boldsymbol{\theta}_i)$, $\hat{p}_i = \mathbf{y}_i / n$). We evaluate 105 parameter combinations ($\alpha \in \{0.1, 0.5, 1.0\} \times C \in \{2, 3, 5, 7, 10\} \times n \in \{3, 5, 10, 20, 30, 50, 100\}$) with 100 replications per cell ($10,500$ total simulations).

**Table 5: Multi-Regime Phase Diagram Surface for 3-Class Tasks ($C=3, k=10$)**

| Votes ($n$) | Concentrated ($\alpha=0.1$) | Symmetric ($\alpha=0.5$) | Uniform ($\alpha=1.0$) | Empirical ChaosNLI |
|---|---|---|---|---|
| 3 | 100.0% ± 0.0% | 100.0% ± 0.0% | 100.0% ± 0.0% | — |
| 5 | 99.8% ± 0.2% | 100.0% ± 0.0% | 100.0% ± 0.0% | — |
| 10 | 99.1% ± 0.3% | 100.0% ± 0.1% | 100.0% ± 0.0% | — |
| 20 | 95.7% ± 0.7% | 94.3% ± 1.4% | 92.5% ± 1.6% | — |
| 30 | 92.2% ± 0.9% | 84.8% ± 1.7% | 85.8% ± 1.7% | — |
| 50 | 87.6% ± 1.2% | 66.2% ± 1.7% | 62.5% ± 1.5% | — |
| **100** | **80.2% ± 1.2%** | **35.7% ± 1.6%** | **25.5% ± 1.1%** | **49.1%** |

#### Occupancy and Simplex Concentration Analysis

Theoretical occupancy analysis over ChaosNLI's $N=3,113$ items and $S = \binom{100+3-1}{3-1} = 5,151$ possible 100-vote 3-class profiles shows that under uniform independent occupancy, expected occupied profiles are:
$$\mathbb{E}[U] = S \left[1 - \left(1 - \frac{1}{S}\right)^N\right] \approx 2,337$$

Empirically, ChaosNLI populates only **1,604 unique profiles**, demonstrating far greater profile concentration than uniform occupancy predicts. The observed boundary-tie prevalence (**49.1%**) lies above the tested $\alpha=0.5$ ($35.7\% \pm 1.6\%$) and $\alpha=1.0$ ($25.5\% \pm 1.1\%$) regimes but below the more concentrated $\alpha=0.1$ regime ($80.2\% \pm 1.2\%$). A single symmetric Dirichlet does not fully characterize the generating distribution, and this comparison should be treated as descriptive rather than as parameter identification.

---

## 5. Study 2: Secondary Exploratory Analysis — VariErr External Validation

As a secondary exploratory analysis, we evaluated our two-level architecture against 500 matched items from VariErr NLI (Weber-Genzel et al., 2024), containing 7,732 human validity judgments over re-annotated MNLI items. This test is underpowered for a confirmatory conclusion (52 multi-item profiles) and should be interpreted as hypothesis-generating. The 500 matched items span **52 multi-item profiles** (group sizes: min 2, median 3, max 16):

- **Overall Item-Level SD**: $0.1413$ (Overall Item-Level Variance: $0.0200$, Bessel correction $n-1$).
- **Mean of Profile-Level SDs**: $0.1060$ (Mean of Profile-Level Variances: $0.0180$, equal profile weighting).
- **Null-Relative Effect Lead**: Observed within-profile SD ($0.1060$) was **7.8% below the profile-size-preserving null mean** ($0.1150$).
- **Statistical Status**: A **500,000-permutation** profile-size preserved null yields an empirical $p$-value of **$p = 0.2045$** ($102,248 / 500,000$ resamples $\le$ observed). The descriptive 25.0% reduction against overall SD ($0.1413$) partly reflects downward sample-SD bias in small groups.

**Table 6: Profile Homogeneity Statistics on Matched VariErr Items**

| Metric | Value | Description |
|---|---|---|
| Matched Items | 500 | Items present in both VariErr and ChaosNLI-M |
| Multi-Item Profiles ($\lvert g\rvert > 1$) | 52 | Distinct ChaosNLI vote profiles containing $\ge 2$ VariErr items |
| Overall Item-Level Validity SD | 0.1413 | Total sample SD of explanation validity ratios ($n=500$) |
| Overall Item-Level Validity Variance | 0.0200 | Total sample variance ($0.1413^2 = 0.0200$) |
| Mean Within-Profile Validity SD | 0.1060 | Average sample SD across 52 multi-item profiles |
| Mean Within-Profile Validity Variance | 0.0180 | Average sample variance across 52 multi-item profiles |
| **Null-Relative SD Reduction** | **7.8%** | Observed within-profile SD ($0.1060$) vs. null mean ($0.1150$) |
| Descriptive SD Reduction vs. Overall | 25.0% | Within-profile SD ($0.1060$) vs. overall SD ($0.1413$) |
| **500,000-Permutation Null Mean Within-Profile SD** | **0.1150** | Profile-size preserved label permutations ($N_{\text{perm}} = 500,000$) |
| **Permutation $p$-value** | $p = 0.2045$ | **Inconclusive** — observed within-profile SD ($0.1060$) vs. null mean ($0.1150$) |

*Scientific Takeaway*: Under the selected profile-dispersion statistic, we found no evidence that exact vote-profile identity predicts explanation-validity composition ($p = 0.2045$). This is consistent with the view that aggregate Level-1 vote distributions are insufficient for recovering rationale structure, but does not establish a zero effect.

---

## 6. Complete Methods and Reproducibility Specifications

### 6.0 Data and Inclusion Scope

ChaosNLI contains 3,113 items ($1,514$ SNLI, $1,599$ MNLI) with 100 human votes per item over 3 semantic classes (entailment, neutral, contradiction). Items were selected by Nie et al. (2020) for annotator disagreement. Selection-conditioned scope: results reflect low-agreement NLI items.

### 6.1 Distance Metrics and Floating-Point Tie Detection

Distance matrices use double-precision float64 arithmetic. Hellinger distance between $p, q$ is $d_H(p, q) = \frac{1}{\sqrt{2}} \sqrt{\sum_{c=1}^C (\sqrt{p_c} - \sqrt{q_c})^2}$. Tie detection uses absolute tolerance $|d_{ij} - d_{ik}| < 10^{-7}$ throughout (both for human count-vector distances and for model softmax-probability distances), matching the canonical manifest `conventions.tie_tolerance`.

### 6.2 Dirichlet Posterior Prior and Sampling Procedure

Posterior distributions use a symmetric Dirichlet prior $\boldsymbol{\alpha} = (0.5, 0.5, 0.5)$ (Jeffreys prior). Posterior predictive samples draw $\boldsymbol{\theta}_i \sim \text{Dirichlet}(\mathbf{x}_i + \boldsymbol{\alpha})$, followed by independent multinomial draws $\mathbf{y}_{i,1}, \mathbf{y}_{i,2} \sim \text{Multinomial}(n, \boldsymbol{\theta}_i)$.

### 6.3 Model Predictions and Logit Conversion

Model predictions use pre-computed logits from 9 benchmark NLI models: BART-Large, RoBERTa-Large, XLNet-Large, ALBERT-xxLarge, BERT-Large, RoBERTa-Base, XLNet-Base, DistilBERT, BERT-Base. Softmax converts logits to 3-class probability distributions $q_m = \text{softmax}(z_m)$ with label order [entailment, neutral, contradiction]. The generated probability artifact is intentionally excluded from version control. The supplied model-prediction artifact is hash-locked and source-attributed in the canonical release manifest (SHA-256 and source URL recorded). Exact checkpoint revisions were not included with the supplied artifact and remain unavailable.

### 6.4 Storage-Order Row Permutation Experiment

Row-permutation tests apply 1,000 random permutations $\pi$ to persistent item identities. Self-distance is explicitly excluded, and deterministic neighbors are ordered by distance with stable storage index as the boundary tie policy. In the presence of exact distance ties, re-indexing changes which boundary candidates have the lowest storage positions. Persistent object identities are restored before computing top-$k$ overlap.

### 6.5 Stratified Joint Bootstrap Procedure

Bootstrap resampling draws 1,000 stratified samples of focal items ($1,514$ SNLI, $1,599$ MNLI drawn with replacement). Each replicate $b$ pairs focal items with posterior pair $s = b \bmod 500$. We use a fully paired design: $H_b = Q(G_{H1}^{(s)}, G_{H2}^{(s)})$ evaluates two independent posterior-predictive cohorts, while $M_{m,b} = \tfrac{1}{2}[Q(G_m, G_{H1}^{(s)}) + Q(G_m, G_{H2}^{(s)})]$ evaluates each model symmetrically against the same two cohorts. Fixed-reference scores $Q(G_m, G_{100}^{\text{obs}})$ are reported as a secondary descriptive baseline. Difference statistics $\Delta_{m,b} = H_b - M_{m,b}$ construct 95% percentile confidence intervals.

### 6.6 Reference Graph Similarity Surface Simulation

Plug-in empirical reference similarity $R_{\text{reference}}(n, k) = Q(G_n^{\text{rep}}, G_{100}^{\text{obs}})$ simulates independent $n$-vote plug-in multinomial draws $\mathbf{y}_i \sim \text{Multinomial}(n, \hat{p}_i)$ using observed proportions $\hat{p}_i$ (not posterior-predictive samples) across 8 vote depths ($n \in \{3..100\}$) and 5 scales ($k \in \{5..100\}$). The simulation used 50 independent seeds per $(n, k)$ cell (seeds 0–49, generated via NumPy `default_rng` with sequential integer seeds). Reported $\pm$ values are sample standard deviations across the 50 seed replicates. The 95% normal-approximation simulation intervals in Table 4 are computed as sample mean $\pm 1.96 \times \text{SD}$, describing variation across the 50 simulation draws (not precision of the mean). Monotonicity of simulation interval lower bounds was verified empirically across all five $k$-columns. A complementary posterior-predictive surface $R_{\text{posterior}}(n, k)$, incorporating Dirichlet sampling over latent $\boldsymbol{\theta}_i$, will be reported in a follow-up revision.

### 6.7 Phase Diagram Simulation Parameters

Phase simulations evaluate 105 parameter combinations ($\alpha \in \{0.1, 0.5, 1.0\} \times C \in \{2, 3, 5, 7, 10\} \times n \in \{3..100\}$) with 100 replications per cell ($10,500$ simulations). Standard deviation bounds describe empirical variation across the 100 cell replications.

### 6.8 VariErr External Matching and Permutation Test

VariErr NLI (Weber-Genzel et al., 2024) matches 500 items to ChaosNLI-M via `source_pair_id`. Validity ratio per item is $y_i = \text{valid judgments} / \text{total judgments}$. Profile-size preserved null shuffles validity ratios 500,000 times natively across 52 multi-item profiles with Bessel $n-1$ SDs and equal profile weighting. Singletons ($|g|=1$) are excluded from profile dispersion averages.

---

## 7. Limitations

1. **Selected Low-Agreement Population**: ChaosNLI targets items with known annotator disagreement, which may overrepresent boundary uncertainty compared to standard NLI corpora.
2. **Pre-2023 Model Set**: Evaluated models reflect BERT/RoBERTa/BART-era architectures; modern generative LLM ensembles may exhibit different neighborhood recovery.
3. **Posterior Prior Assumptions**: Posterior predictive simulations rely on a symmetric Dirichlet prior ($\boldsymbol{\alpha} = 0.5$); alternative priors or empirical bootstraps may alter reference distribution width.
4. **Fixed-Reference Geometry Sensitivity**: Geometry robustness is evaluated under a single observed reference graph $G_{100}^{\text{obs}}$; full posterior-averaged sensitivity across all metrics remains pending.
5. **Scale Dependence ($k$)**: Neighborhood preservation results depend on chosen neighborhood size $k$.
6. **Single Aitchison Zero Policy**: Log-ratio geometry uses a single zero-replacement threshold ($\epsilon=10^{-4}$); alternative log-ratio zero policies were not evaluated.
7. **Unobserved Disagreement Drivers**: Level-1 vote profiles describe label frequencies but do not reveal underlying linguistic or cognitive causes of disagreement.
8. **VariErr Sample Constraints**: External validity test is restricted to 500 items across 52 multi-item profiles, limiting statistical power for subtle profile-level effects.
9. **Relational Agreement vs. Ground Truth**: High neighborhood preservation indicates structural alignment with human opinion neighborhood structure, not absolute semantic correctness.
10. **Conditional Graph Construction**: The bootstrap resamples focal-item overlap contributions while holding each precomputed model neighbor graph $G_m$ and posterior-predictive human neighbor graphs $G_{H1}^{(s)}, G_{H2}^{(s)}$ fixed. It therefore does not account for uncertainty from reconstructing those graphs under a new sample of items.
11. **Bootstrap Edge Non-Independence**: Focal-item resampling treats per-item overlaps as exchangeable, but edges in the neighbor graph are non-independent (shared candidate nodes). Confidence intervals should be interpreted as approximations.
12. **Plug-in Surface vs. Posterior-Predictive Surface**: The reference surface $R_{\text{reference}}(n, k)$ is a plug-in analysis conditioned on observed vote proportions $\hat{p}_i$ and does not incorporate posterior uncertainty over each item's latent human distribution $\boldsymbol{\theta}_i$. A posterior-predictive surface $R_{\text{posterior}}(n, k)$ will be reported in a follow-up revision.

---

## 8. Conclusion

We have presented a tie-aware computational study of human collective NLI opinion neighborhood structure. We formalized the three-quantity tie interval ($Q_{\text{strict}} \le Q_{\text{expected}} \le Q_{\text{fuzzy}}$), proved six core properties, established a 9-model recovery gap under 1,000 joint bootstrap replicates, mapped reference graph similarity $R_{\text{reference}}(n, k)$, and executed high-performance native Rust verification. The Hellinger benchmark gap remains significant under joint posterior-reference resampling, while fixed-reference gaps persist across five distance specifications. Recovering the neighbor-graph organization of collective human disagreement remains an important open challenge for language models.

---

## References

Aitchison, J. (1982). The statistical analysis of compositional data. *JRSS B, 44*(2), 139–177.  
Amari, S. (2000). Methods of information geometry. *AMS*.  
Baan, J. et al. (2022). Stop measuring calibration when humans disagree. *EMNLP 2022*.  
Chazal, F., & Michel, B. (2021). An introduction to topological data analysis: Fundamental and practical aspects for data scientists. *Frontiers in Artificial Intelligence, 4*, 667963.  
Chen, L., & Buja, A. (2009). Local multidimensional scaling for nonlinear dimension reduction, graph drawing, and proximity analysis. *Journal of the American Statistical Association, 104*(485), 209–219.  
Endres, D.M., & Schindelin, J.E. (2003). A new metric for probability distributions. *IEEE TIT, 49*(7), 1858–1860.  
Gruber, N. et al. (2024). More labels or cases? Assessing label variation in NLI.  
Jiang, M., & de Marneffe, M.-C. (2022). Investigating reasons for disagreement in NLI. *TACL, 10*, 1357–1374.  
Kurniawan, K., Mistica, M., Baldwin, T., & Lau, J.H. (2026). Training and evaluating with human label variation: An empirical study. *Computational Linguistics, 52*(1).  
Lee, J.A., & Verleysen, M. (2008). Quality assessment of nonlinear dimensionality reduction based on K-ary neighborhoods. *PMLR, 4*, 21–35.  
Lee, J.A., & Verleysen, M. (2009). Quality assessment of dimensionality reduction: Rank-based criteria. *Neurocomputing, 72*, 1431–1443.  
Lueks, W. et al. (2011). How to evaluate dimensionality reduction? Improving co-ranking matrix analysis. *ESANN 2011*.  
Mason, B., Tripathy, A., & Nowak, R. (2019). Learning nearest neighbor graphs from noisy distance samples. *NeurIPS 2019*.  
Nie, Y., Zhou, X., & Bansal, M. (2020). What can we learn from collective human opinions on NLI data? *EMNLP 2020*.  
Pavlick, E., & Kwiatkowski, T. (2019). Inherent disagreements in human textual inferences. *TACL, 7*, 677–694.  
Plank, B. (2022). The "problem" of human label variation. *EMNLP 2022*.  
Uma, A.N., Fornaciari, T., Hovy, D., Paun, S., Plank, B., & Poesio, M. (2021). Learning from disagreement: A survey. *Journal of Artificial Intelligence Research, 72*, 1385–1470.  
Wang, et al. (2022). Capture human disagreement distributions by calibrated networks. *EMNLP 2022*.  
Weber-Genzel, S., Peng, S., de Marneffe, M.-C., & Plank, B. (2024). VariErr NLI: Separating annotation error from human label variation. *ACL 2024*.  
Zadeh, L.A. (1965). Fuzzy sets. *Information and Control, 8*(3), 338–353.  
Zhou, X., Nie, Y., & Bansal, M. (2022). Distributed NLI: Learning to predict human opinion distributions. *ACL Findings 2022*.
