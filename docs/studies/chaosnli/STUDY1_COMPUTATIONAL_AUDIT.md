# Study 1 Computational Audit & Empirical Report

**Dataset:** 3,113 Three-Class ChaosNLI Examples (1,514 SNLI + 1,599 MNLI)  
**Date:** 2026-08-02 (Canonical Lock post Peer-Review Audit)  
**Scope:** ChaosNLI Low-Agreement Sample, Human-Opinion Topology, Dirichlet Posteriors, Fractional Tie-Aware Neighborhoods, Scale Curves, Annotation Budgeting, and Geometry Sensitivity

> **Canonical Manifest Integration:** All quantitative values reported herein are generated directly from `results/canonical_results.yaml`.

---

## 1. Summary of Quantitative Findings

| Estimand / Property | Value | Description |
|---|---|---|
| **Canonical Dataset Size** | **3,113 items** | 100 human judgments per item (1,514 SNLI + 1,599 MNLI) |
| **Unique Opinion Profiles (Level 1 Nodes)** | **1,604 unique** | Discrete 3-class distribution vectors |
| **Items in Non-Singleton Profiles** | **2,193 items (70.4%)** | Items sharing an exact label distribution with at least one other item |
| **Multi-Item Profiles with Mixed Sources** | **337 profiles (49.3%)** | Multi-item profiles containing both SNLI and MNLI items |
| **Items with Distance Ties at $k=10$ Boundary** | **2,254 items (72.4%)** | Items with exact distance ties across the $k=10$ neighbor boundary |
| **Empirical Mean Entropy** | **0.9386 bits** | Overall distribution dispersion across dataset |
| **Posterior Mean Composition Entropy ($H(E[\theta\mid x])$)** | **0.9534 bits** | Regularized under Dirichlet $\boldsymbol{\alpha}=(0.5, 0.5, 0.5)$ |
| **Deterministic Storage-Order Overlap Artifact** | **0.9074 $\pm$ 0.0024** | Deterministic overlap across **1,000** random row permutations (SD $= 0.0024$) |
| **Fractional Soft Overlap Invariance** | **1.0000 $\pm$ 0.0000** | Strictly row-order invariant across all 1,000 permutations |
| **500-Pair Posterior Predictive Reference ($Q_{NX}^{\text{soft, HH100}}$)** | **0.07550** | Mean of 500 posterior-predictive simulation pairs; Median $0.07548$, SD $0.00227$, Monte Carlo SE $0.000102$, 95% Posterior-Predictive Simulation Interval $[0.07111, 0.08007]$ |
| **Theoretical Chance Baseline Overlap ($k/(N-1)$)** | **0.00321 (0.321%)** | Expected random overlap for $k=10, N=3113$ |
| **Empirical Stratified Null Mean** | **0.00354 (0.354%)** | Mean overlap under 100 stratified item-identity permutations |

---

## 2. Failure Analysis of Deterministic Top-$k$ Sorting

Our audit investigated the storage-order sensitivity of deterministic top-$k$ sorting under distance ties:

- **Mechanism**: Conventional index-resolved fixed-$k$ neighborhoods are not data-order-invariant in highly tied empirical-distribution spaces. Re-indexing data rows under **1,000 independent row permutations** alters deterministic top-$k$ overlap (**$0.9074 \pm 0.0024$**, SD $0.0024$, 95% interval $[0.9027, 0.9119]$), verified by the native Rust 16-core engine.
- **Fractional Soft Invariance**: Fractional soft overlap $Q_{NX}^{\text{soft}}$ is strictly invariant (**$1.0000 \pm 0.0000$**).

### Formal Definition of Fractional Soft Overlap

For focal item $i$ and target rank $k$, let $A_i$ be candidate neighbors strictly closer than $d_i(k)$, $B_i$ be tied candidates at distance $d_i(k)$, and $r_i = k - |A_i|$. The weight $w_{ij}$ for candidate $j$ is:

$$w_{ij} = \begin{cases} 1, & d_{ij} < d_i(k) \\ \frac{r_i}{|B_i|}, & d_{ij} = d_i(k) \\ 0, & d_{ij} > d_i(k) \end{cases}$$

The soft overlap between two weighted neighborhood structures $w_{ij}^A$ and $w_{ij}^B$ is:

$$O_i^{\text{soft}}(k) = \frac{1}{k} \sum_{j=1}^N \min(w_{ij}^A, w_{ij}^B), \qquad Q_{NX}^{\text{soft}}(k) = \frac{1}{N} \sum_{i=1}^N O_i^{\text{soft}}(k)$$

---

## 3. Human Reliability Reference Spectrum & Formal Tie Mathematics

### Three-Quantity Overlap Inequality

For candidate weights $w_{ij}^A, w_{ij}^B \in [0, 1]$, we formalize three neighborhood overlap quantities:
1. **$Q_{\text{strict}}$ (Strict-Core Lower Bound)**: Non-tied core boundary overlap ($\frac{1}{k}\sum_j \mathbf{1}(w_{ij}^A=1) \mathbf{1}(w_{ij}^B=1)$).
2. **$Q_{\text{expected}}$ (Expected Random-Tie Overlap)**: Expected uniform random boundary tie resolution ($\frac{1}{k}\sum_j w_{ij}^A w_{ij}^B$).
3. **$Q_{\text{fuzzy}}$ (Min-Based Fuzzy Overlap)**: Fuzzy set intersection membership overlap ($\frac{1}{k}\sum_j \min(w_{ij}^A, w_{ij}^B)$).

**Six Fundamental Properties**:
1. **Range**: All three quantities are bounded in $[0, 1]$.
2. **Symmetry**: $Q_\bullet(G^A, G^B) = Q_\bullet(G^B, G^A)$ for all three formulations.
3. **Fuzzy Identity**: $Q_{\text{fuzzy}}(G, G) = 1.0$ for any weighted neighborhood graph $G$.
4. **Expected and Strict Self-Overlap**: $Q_{\text{expected}}(G, G) = \frac{1}{Nk}\sum_{i}\sum_{j \ne i} w_{ij}^2 \le 1.0$, measuring collision probability under independent random tie resolutions. $Q_{\text{strict}}(G, G) = 1.0$ when every selected neighborhood membership has unit weight — that is, when there are no fractional boundary memberships at rank $k$.
5. **Row-Order Permutation Invariance**: For any permutation $\pi$, $Q_\bullet(G^A, G^B) = Q_\bullet(\pi G^A, \pi G^B)$ after matching persistent object identities. Across 1,000 random permutations, the maximum absolute pre/post-permutation difference was **0.0000**.
6. **Reduction to Standard $Q_{NX}$ Under Unique Boundary**: When every $k$-boundary distance is unique ($|B_i| = 1, r_i = 1$), candidate weights are binary ($w_{ij} \in \{0, 1\}$) and all three formulations reduce strictly to standard $Q_{NX}$.

### Item-Level Permutation Overlap Breakdown

**Table 1: Item-Level Permutation Overlap Breakdown Across Neighborhood Scales ($k$)**

| Scale ($k$) | Mean Overlap | Median Overlap | 5% – 95% Interval | Min Overlap | Items Changed (%) |
|---|---|---|---|---|---|
| $k=5$ | 0.8182 | 0.8480 | [0.5620, 1.0000] | 0.3340 | 62.0% |
| $k=10$ | **0.9071** | **0.9210** | **[0.7700, 1.0000]** | **0.6640** | **62.1%** |
| $k=20$ | 0.9520 | 0.9620 | [0.8800, 1.0000] | 0.6885 | 62.3% |
| $k=50$ | 0.9808 | 0.9846 | [0.9526, 1.0000] | 0.9108 | 63.3% |

---

## 4. Model Benchmark & Stratified 95% Joint Bootstrap

**Formal Inferential Test ($\Delta_m = Q_{NX}^{\text{soft, HH100}} - Q_{NX}^{\text{soft, HM}}$ across 1,000 Joint Bootstrap Resamples vs. 500-Pair HH100 Reference Distribution):**

*Methods Note*: Under a fully paired design ($M_{m,b} = \frac{1}{2}[Q(G_m, G_{H1}^{(s)}) + Q(G_m, G_{H2}^{(s)})]$), both human reliability $H_b$ and model score $M_{m,b}$ are evaluated symmetrically against identical posterior-predictive cohorts across 1,000 stratified joint bootstrap resamples. In 1,000 of 1,000 replicates, every model difference interval $\Delta_m$ comfortably excludes zero (minimum lower bound $\ge 0.05431$). Fixed full-data reference scores $Q(G_m, G_{100}^{\text{obs}})$ are retained as a secondary baseline.

| Model Name | Paired Score $M_{m,b}$ | Mean $\Delta_m$ (vs. Paired HH100) | 95% Joint Bootstrap CI ($\Delta_m$) | Replicates $\Delta_m > 0$ | Fixed Reference $Q(G_m, G_{100}^{\text{obs}})$ |
|---|---|---|---|---|---|
| **BART-Large** | **0.01572** | **0.05977** | **[0.05431, 0.06539]** | **1,000 / 1,000** | **0.01867** |
| **RoBERTa-Large** | **0.01415** | **0.06135** | **[0.05557, 0.06685]** | **1,000 / 1,000** | **0.01821** |
| **XLNet-Large** | **0.01285** | **0.06264** | **[0.05711, 0.06846]** | **1,000 / 1,000** | **0.01319** |
| **ALBERT-xxLarge** | **0.01124** | **0.06426** | **[0.05896, 0.06997]** | **1,000 / 1,000** | **0.01074** |
| **BERT-Large** | **0.01029** | **0.06520** | **[0.05966, 0.07076]** | **1,000 / 1,000** | **0.01059** |
| **RoBERTa-Base** | **0.01007** | **0.06543** | **[0.05979, 0.07106]** | **1,000 / 1,000** | **0.01129** |
| **XLNet-Base** | **0.00927** | **0.06623** | **[0.06069, 0.07175]** | **1,000 / 1,000** | **0.00893** |
| **DistilBERT** | **0.00854** | **0.06695** | **[0.06124, 0.07261]** | **1,000 / 1,000** | **0.00854** |
| **BERT-Base** | **0.00768** | **0.06782** | **[0.06235, 0.07356]** | **1,000 / 1,000** | **0.00865** |
| **HH100 Reference (Paired)** | **0.07549** | — | **[0.07000, 0.08099]** | — | — |

---

## 5. Reference Graph Similarity Surface $R_{\text{reference}}(n, k) = Q(G_n^{\text{rep}}, G_{100}^{\text{obs}})$

*Notation*: $G_n^{\text{rep}}$ is an independent $n$-vote posterior-predictive draw; $G_{100}^{\text{obs}}$ is the empirical 100-vote observed graph. Because both are stochastic samples, $R(100, k) < 1.0$ by definition — this is reference graph *similarity*, not ground-truth *recovery*.

| Votes ($n$) | $k=5$ | $k=10$ | $k=20$ | $k=50$ | $k=100$ |
|---|---|---|---|---|---|
| 3 | 0.0061 | 0.0109 | 0.0208 | 0.0497 | 0.0952 |
| 5 | 0.0084 | 0.0149 | 0.0279 | 0.0649 | 0.1214 |
| 10 | 0.0137 | 0.0248 | 0.0463 | 0.1008 | 0.1799 |
| 20 | 0.0236 | 0.0397 | 0.0722 | 0.1539 | 0.2606 |
| 30 | 0.0327 | 0.0555 | 0.0968 | 0.2010 | 0.3237 |
| 50 | 0.0483 | 0.0832 | 0.1436 | 0.2765 | 0.4111 |
| 75 | 0.0652 | 0.1160 | 0.1944 | 0.3510 | 0.4916 |
| **100** | **0.0819** | **0.1385** | **0.2309** | **0.4030** | **0.5376** |

---

## 6. Geometry Sensitivity Benchmark Across All 9 Models ($k=10$)

| Model Name | Hellinger | JSD ($\sqrt{\text{JS}}$) | Total Variation | Euclidean | Aitchison ($\epsilon=10^{-4}$) |
|---|---|---|---|---|---|
| BART-Large | 0.01617 | 0.01623 | 0.01708 | 0.01716 | 0.01618 |
| RoBERTa-Large | 0.01398 | 0.01404 | 0.01366 | 0.01385 | 0.01507 |
| XLNet-Large | 0.01231 | 0.01238 | 0.01364 | 0.01366 | 0.01399 |
| ALBERT-xxLarge | 0.01214 | 0.01226 | 0.01209 | 0.01186 | 0.01208 |
| BERT-Large | 0.01003 | 0.00991 | 0.00987 | 0.01003 | 0.00966 |
| RoBERTa-Base | 0.01018 | 0.01009 | 0.01029 | 0.01019 | 0.01014 |
| XLNet-Base | 0.01016 | 0.01005 | 0.00978 | 0.00971 | 0.01034 |
| DistilBERT | 0.00835 | 0.00844 | 0.00792 | 0.00777 | 0.00876 |
| BERT-Base | 0.00729 | 0.00721 | 0.00769 | 0.00764 | 0.00776 |

*Note*: Table reports single fixed-reference estimates $Q(G_m, G_{100}^{\text{obs}})$ across all 9 models and 5 geometries. Fixed-reference model-human gaps persisted across all nine models and five metric geometries; full posterior-resampled geometry sensitivity is pending.
