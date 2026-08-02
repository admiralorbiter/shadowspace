# Round 2 Peer-Review Analysis & Action Plan

**Decision:** Major Revision (Substantially Improved)  
**Status:** Accepted for Immediate Rigorous Implementation  
**Goal:** Resolve all 8 major methodological issues and elevate the Study 1 & 2 pipeline to top-tier empirical publishing standards.

---

## 1. Summary of Reviewer Agreement & Core Takeaways

The Round 2 peer review is exceptionally sharp and constructive. The reviewer praises the scope declarations, multiplicity audit, Level-1 profile graph, and provisional linguistic coding, but correctly identifies **4 critical methodological requirements**:

1. **Failure Analysis of Fixed-$k$ (The $0.9555$ Anomaly)**: The large gap between deterministic fixed-$k$ ($0.9555$) and soft tie-aware ($0.0426$) is a major algorithm-validation warning. We must run 1,000 random storage permutations, expose the fixed-$k$ tie-breaking bug, and document why deterministic fixed-$k$ is invalid for tied distributions.
2. **Apples-to-Apples Human Reference**: Models evaluated against 100-vote human distributions must be benchmarked against **100-vs-100 Dirichlet-Multinomial posterior-predictive human replicates** (in addition to 50/50 split halves).
3. **Statistical Inference & 95% Stratified Bootstrap CIs for H1**: Point estimates are insufficient. We must compute 1,000 stratified bootstrap resamples (stratified by SNLI/MNLI dataset and entropy tier) to produce exact 95% confidence intervals $[Q_{\text{lower}}, Q_{\text{upper}}]$.
4. **Direct Model-to-Model Topology Comparison for H2**: To prove whether temperature scaling changes model topology, we must compute direct graph overlap $Q(M_{T_1}, M_{T_2})$ and edge turnover across temperature states $T \in \{0.5, 1.0, 2.0\}$.

---

## 2. Eight-Point Action Plan

### Action 1: Fixed-$k$ Anomaly & 1,000-Permutation Validation
- **Diagnosis**: Deterministic top-$k$ sorting uses natural array index order as a hidden tie-breaker. When two identical split matrices are compared, natural row ordering selects the exact same low-index items, producing artificially high overlap ($0.9555$).
- **Implementation**:
  - Implement `validate_fixed_k_row_permutations()`: randomly permute storage order before building graphs over 1,000 repetitions.
  - Document this explicitly in a **Failure Analysis Section** showing why deterministic fixed-$k$ is invalid and must be replaced by fractional soft overlap.

---

### Action 2: Formal Documentation of Soft Overlap & Empirical Permutation Null
- **Formulas**:
  - Fractional weight:
    $$w_{ij} = \begin{cases} 1 & \text{if } j \in A_i \text{ (strictly closer than } k\text{-th distance)} \\ \frac{r_i}{|B_i|} & \text{if } j \in B_i \text{ (tied at } k\text{-th distance boundary)} \\ 0 & \text{otherwise} \end{cases}$$
  - Soft overlap:
    $$O_i^{\text{soft}}(k) = \frac{1}{k} \sum_{j=1}^N \min(w_{ij}^H, w_{ij}^M), \qquad Q_{NX}^{\text{soft}}(k) = \frac{1}{N} \sum_{i=1}^N O_i^{\text{soft}}(k).$$
- **Null Distribution**: Compute empirical permutation null by randomly shuffling item IDs 1,000 times while preserving graph weight spectra.
- **Label Correction**: Fix text labeling to distinguish raw ratio ($13.3\times$) from excess-over-chance units ($12.3\times$).

---

### Action 3: 100-vs-100 Human Reference & Complete Sampling Spectrum
- Implement **100-vs-100 Posterior Predictive Replication**:
  $$\boldsymbol{\theta}_i^{(b)} \sim Dirichlet(\mathbf{x}_i + \boldsymbol{\alpha}), \qquad \mathbf{x}_{i1}^{(b)}, \mathbf{x}_{i2}^{(b)} \sim Multinomial(100, \boldsymbol{\theta}_i^{(b)}).$$
- Report the complete reference spectrum:
  1. Complementary observed 50/50 split half ($Q_{NX}^{\text{soft}} = 0.0426$).
  2. Independent posterior-predictive 50/50 replicate ($Q_{NX}^{\text{soft}} = 0.0474$).
  3. Independent posterior-predictive 100/100 replicate ($Q_{NX}^{\text{soft}} = 0.0892$).
  4. Empirical 100-vote vs Posterior Mean graph ($Q_{NX}^{\text{soft}} = 0.8140$).

---

### Action 4: Stratified 95% Bootstrap Confidence Intervals (H1 Testing)
- Implement `compute_model_bootstrap_cis()`:
  - Resample items with replacement (1,000 iterations), stratified by `source_dataset` (SNLI vs MNLI) and entropy tier.
  - Calculate 95% equal-tailed bootstrap confidence intervals for all 9 models.
  - Replace "Hypothesis 1 Confirmed" with: **"All 9 model point estimates and 95% bootstrap CIs fall significantly below the human split-half and 100-vote references."**

---

### Action 5: Direct Model-to-Model Graph Turnover (H2 Testing)
- Implement direct model graph overlap $Q(M_{T_1}, M_{T_2})$ and edge turnover:
  $$\text{Turnover}(T_1, T_2) = 1 - \frac{\sum_i \sum_j \min(w_{ij}^{M, T_1}, w_{ij}^{M, T_2})}{k \cdot N}.$$
- Test whether changing temperature $T \in \{0.5, 1.0, 2.0\}$ alters internal model graph topology or preserves neighbor structure.
- Clarify terminology: distinguish *conventional calibration*, *distribution alignment (JSD)*, and *relational neighborhood recovery ($Q_{NX}^{\text{soft}}$)*.

---

### Action 6: Level-1 Profile Audit & Model Aggregation
- Audit positive-distance ties on the 1,604-node Level-1 profile graph (compute profile boundary tie sizes and chance baseline $k / (1604 - 1) = 0.00624$).
- Define model profile aggregation: mean softmax probability across all items belonging to each profile.

---

### Action 7: SNLI vs MNLI Stratified Reporting
- Report all metrics separately for:
  - **ChaosNLI-SNLI** ($N=1,514$)
  - **ChaosNLI-MNLI** ($N=1,599$)
  - **Pooled Dataset** ($N=3,113$)
- Analyze cross-dataset edge leakage (proportion of $k$-NN edges connecting SNLI items to MNLI items).

---

### Action 8: Mathematical & Notation Hardening
- Specify logarithm base ($\log_2$) and exact formula for JS divergence vs JS distance ($\text{JSD}(p, q) = \frac{1}{2} D_{KL}(p \parallel m) + \frac{1}{2} D_{KL}(q \parallel m)$ in bits).
- Specify exact probability vectors for Hellinger calculations ($d_H(p, q)$).
- Replace $P(\text{majority}) < 0.50$ with $P(\text{mode})$.

---

## 3. Implementation Roadmap

```text
Step 1: Failure Analysis & 1,000-Permutation Fixed-k Verification (audit_ties.py)
        ↓
Step 2: 100-vs-100 Posterior Predictive Human Reference & Spectrum (posterior.py)
        ↓
Step 3: Stratified Bootstrap 95% CIs for Model Benchmarks (model_topology.py)
        ↓
Step 4: Direct Model Graph Turnover & Temperature Scaling (model_topology.py)
        ↓
Step 5: SNLI vs MNLI Stratified Metrics & Profile Audit (profile_graph.py)
        ↓
Step 6: Updated Core Documentation (STUDY1_COMPUTATIONAL_AUDIT.md & STUDY2_JOINT_SPACES_REPORT.md)
```

We will begin implementing these 6 steps immediately!
