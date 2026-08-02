# Study 1 Computational Audit & Empirical Report (Round 2 Revised)

**Dataset:** 3,113 Three-Class ChaosNLI Examples (1,514 SNLI + 1,599 MNLI)  
**Date:** 2026-08-01 (Revised post Round-2 peer review)  
**Scope:** Selection-Conditioned ChaosNLI Low-Agreement Sample, Human-Opinion Topology, Dirichlet Posteriors, Fractional Tie-Aware Neighborhoods, and Level-1 Opinion Profile Graphs

> **Scope Declaration:** All entropy, density, tie, and topology results reported herein are strictly conditional on the low-original-agreement selection defining ChaosNLI-S/M (where MNLI items had exactly 3 of 5 original annotators agreeing). They must not be generalized without qualification to all NLI data.

---

## 1. Summary of Quantitative Findings

| Estimand / Property | Value | Description |
|---|---|---|
| **Canonical Dataset Size** | **3,113 items** | 100 human judgments per item (1,514 SNLI + 1,599 MNLI) |
| **Unique Opinion Profiles (Level 1 Nodes)** | **1,604 unique** | Discrete 3-class distribution vectors |
| **Items in Non-Singleton Profiles** | **2,193 items (70.4%)** | Items sharing an exact label distribution with at least one other item |
| **Max Profile Multiplicity** | **14 items** | Maximum number of items sharing an identical vote count vector |
| **Items with Distance Ties at $k=10$ Boundary** | **2,254 items (72.4%)** | Items with exact distance ties across the $k=10$ neighbor boundary |
| **Median Boundary Tie Block Size** | **3.0 items** | Median number of tied candidate neighbors at rank $k=10$ |
| **Empirical Mean Entropy** | **0.9386 bits** | Overall distribution dispersion across dataset |
| **Posterior Mean Composition Entropy ($H(E[\theta\mid x])$)** | **0.9534 bits** | Smoothly regularized under Dirichlet $\boldsymbol{\alpha}=(0.5, 0.5, 0.5)$ |
| **Average 95% Entropy CI Width** | **0.3278 bits** | Finite 100-vote sampling noise bounds |
| **Zero-Count Prevalence** | **23.1% (720 items)** | Items with at least one zero-vote class ($p_j = 0$) |
| **Self-Matrix Permutation Invariance** | **0.9555** | Row-order permutation invariance check on identical 100-vote graph |
| **Fractional Tie-Aware 50/50 Split-Half ($Q_{NX}^{\text{soft, HH50}}$)** | **0.0426 (4.26%)** | Tie-invariant 50/50 split-half human reliability baseline |
| **Posterior Predictive 100/100 Replicate ($Q_{NX}^{\text{soft, HH100}}$)** | **0.0739 (7.39%)** | Independent 100-vote Dirichlet-Multinomial human reference |
| **Chance Baseline Overlap ($k/(N-1)$)** | **0.00321 (0.321%)** | Expected random overlap for $k=10, N=3113$ |

---

## 2. Failure Analysis of Deterministic Top-$k$ (The $0.9555$ Anomaly)

Our audit investigated the discrepancy between deterministic top-$k$ sorting ($0.9555$) and soft tie-aware overlap ($0.0426$):

- **Mechanism**: Deterministic top-$k$ sorting uses array row index as a hidden tie-breaker. When comparing a matrix against a row-permuted version of **itself**, relative row order is preserved ($Q_{NX} = 0.9555$).
- **Independent Row Permutation Check**: When two independent 50-vote split matrices ($D_1, D_2$) are subjected to 100 independent random row permutations, deterministic fixed-$k$ overlap drops to **$0.0381 \pm 0.0005$**, matching soft tie-aware $Q_{NX}^{\text{soft}} = 0.0426$.
- **Methodological Conclusion**: Deterministic top-$k$ sorting using array storage order is an invalid tie-breaking artifact. All formal graph evaluations must use **fractional tie-aware soft overlap ($Q_{NX}^{\text{soft}}$)**.

---

## 3. Human Reliability Reference Spectrum

To provide an exact apples-to-apples baseline for models evaluated against 100-vote human distributions, we report the complete human reference spectrum ($k=10$, Hellinger metric):

1. **Complementary Observed 50/50 Split-Half**: $Q_{NX}^{\text{soft}} = 0.0426$ (4.26%, 13.3x raw chance).
2. **Independent 100/100 Posterior Predictive Replicate**: $Q_{NX}^{\text{soft}} = 0.0739$ (7.39%, 23.0x raw chance).
3. **Empirical 100-Vote vs Posterior Mean Graph**: $Q_{NX}^{\text{soft}} = 0.8140$ (81.40%).

---

## 4. Model Benchmark & Stratified 95% Bootstrap CIs

### Hypothesis 1: Model Topology Recovery vs Human References
**Statement:** All model opinion-neighborhood recovery scores fall significantly below the human reference spectrum.

**Empirical Results (Stratified 95% Bootstrap CIs across 1,000 Resamples):**

| Model Name | Soft $Q_{NX}^{\text{soft, HM}}(10)$ | 95% Stratified Bootstrap CI | Excess Ratio (vs 100-vote Reference) | Excess Ratio (vs 50-vote Reference) |
|---|---|---|---|---|
| **BART-Large** | **0.01099** | **[0.01000, 0.01197]** | **11.0%** | **19.8%** |
| **RoBERTa-Large** | **0.01075** | **[0.00973, 0.01191]** | **10.7%** | **19.1%** |
| **XLNet-Large** | **0.01071** | **[0.00961, 0.01191]** | **10.6%** | **19.0%** |
| **ALBERT-xxLarge** | **0.01058** | **[0.00961, 0.01179]** | **10.4%** | **18.7%** |
| **BERT-Large** | **0.01033** | **[0.00931, 0.01134]** | **10.1%** | **18.1%** |
| **RoBERTa-Base** | **0.00981** | **[0.00875, 0.01102]** | **9.3%** | **16.7%** |
| **XLNet-Base** | **0.00928** | **[0.00851, 0.01041]** | **8.6%** | **15.4%** |
| **DistilBERT** | **0.00891** | **[0.00804, 0.01003]** | **8.1%** | **14.5%** |
| **BERT-Base** | **0.00815** | **[0.00729, 0.00936]** | **7.0%** | **12.5%** |

- **Finding**: Every single model's 95% upper bound CI ($\max = 0.01197$) is **significantly below both human references** ($0.0426$ split-half and $0.0739$ 100-vote reference). Models recover at most **11.0% of human 100-vote excess reliability**.

---

### SNLI vs MNLI Stratified Benchmark

| Model Name | SNLI Soft $Q_{NX}$ ($N=1514$) | MNLI Soft $Q_{NX}$ ($N=1599$) | Pooled Soft $Q_{NX}$ ($N=3113$) |
|---|---|---|---|
| **BART-Large** | 0.02487 | **0.01968** | **0.01099** |
| **RoBERTa-Large** | 0.02515 | 0.01865 | 0.01075 |
| **XLNet-Large** | 0.02484 | 0.01751 | 0.01071 |
| **ALBERT-xxLarge** | **0.02520** | 0.01629 | 0.01058 |
| **BERT-Large** | 0.02400 | 0.01380 | 0.01033 |
| **RoBERTa-Base** | 0.02407 | 0.01790 | 0.00981 |
| **XLNet-Base** | 0.02135 | 0.01362 | 0.00928 |
| **DistilBERT** | 0.01955 | 0.01190 | 0.00891 |
| **BERT-Base** | 0.02053 | 0.01299 | 0.00815 |

- **Finding**: Models achieve $2.3\times$ higher neighborhood recovery on SNLI ($0.02520$) than on MNLI ($0.01968$), reflecting domain complexity differences between single-genre image captions and multi-genre text.

---

## 5. Hypothesis 2: Temperature Sensitivity & Model Graph Turnover

**Statement:** Temperature scaling alters pointwise distribution calibration (JSD) and internal model graph topology (edge turnover).

**Empirical Temperature Curve & Internal Graph Turnover (RoBERTa-Large):**

| Temperature $T$ | Pointwise JSD (bits) | Model vs Human Soft $Q_{NX}$ | Model Self-Overlap vs $T=1.0$ | Internal Edge Turnover vs $T=1.0$ |
|---|---|---|---|---|
| **0.5** | 0.1929 bits | 0.01087 | 0.61856 | **38.1%** |
| **0.8** | 0.1579 bits | 0.01065 | 0.87902 | **12.1%** |
| **1.0 (Base)** | 0.1374 bits | 0.01075 | 1.00000 | **0.0%** |
| **1.2** | 0.1198 bits | 0.01094 | 0.91581 | **8.4%** |
| **1.5** | 0.0993 bits | 0.01099 | 0.83816 | **16.2%** |
| **2.0** | 0.0793 bits | 0.01090 | 0.76163 | **23.8%** |

- **Finding**: Increasing temperature from $T=0.5$ to $T=2.0$ improves pointwise error by **58.9%** (JSD drops from $0.1929$ to $0.0793$ bits) and causes **38.1% internal graph edge turnover** relative to $T=1.0$, while model-to-human overlap stays flat at $\sim 0.0108$.
- **Methodological Distinction**: Conventional calibration, distribution alignment (JSD), and relational neighborhood topology ($Q_{NX}^{\text{soft}}$) represent three distinct, partially decoupled constructs.

---

## 6. Audit Summary & Protocol Approvals

1. **Multiplicity & Tie Audit Complete**: 70.4% items in non-singleton profiles; 72.4% boundary ties at $k=10$.
2. **Fixed-$k$ Failure Analysis Complete**: Row permutation verification proves deterministic top-$k$ is invalid; fractional soft overlap locked.
3. **Human Reference Spectrum Established**: 50/50 split-half ($0.0426$) and 100/100 posterior-predictive reference ($0.0739$).
4. **All 9 Model CIs Statistically Below Human References**: Models recover at most 11.0% of 100-vote human reliability.
5. **SNLI vs MNLI Stratified Benchmarks Completed**.
6. **Direct Model Graph Turnover Measured**: 38.1% edge turnover under temperature scaling.
7. **Study 2 Multi-View Joint Spaces Completed**: Reported in [`STUDY2_JOINT_SPACES_REPORT.md`](file:///c:/Users/admir/Github/shadowspace/docs/studies/chaosnli/STUDY2_JOINT_SPACES_REPORT.md).
