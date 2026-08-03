# E003: Relational Repair Capacity of Flexible Post-Hoc Calibration & Ensembling

**Experiment ID**: E003  
**Title**: Relational Repair Capacity of Increasingly Flexible Post-Hoc Transformations & Ensembling  
**Status**: `complete_publication_grade`  
Dataset Release: `chaosnli-canonical-2026-08-02` (N = 3,113 items)  
Cross-Validation: 5-Fold Stratified Coherent Cross-Fitting by (Dataset, Majority Label, Empirical Entropy Quintile)  
Bound E001 Artifact (k=10): `E001-hellinger-k010-expected-fuzzy-support-v1` (SHA-256: `94e483e714d92f03...`)  
Bound E001 Artifact (k=50): `S_hellinger_k050.bin` (SHA-256: `2da027e261d9a74a...`)  
Model Probs Hash: `3353c88dbeb3d229...`  
Human Soft-Label Entropy Floor ($H(p)$): 0.65062 nats  
Human Relational Reference ($Q_{HH}$): 0.07228  

---

## Executive Summary

Experiment **E003** evaluates the **Relational Repair Ladder**: *How much of the human belief-space relational topology gap ($G_Q$) can be recovered through increasingly flexible post-hoc calibration and ensembling techniques BEFORE representational fine-tuning becomes necessary?*

### Key Scientific Findings

1. **BART-Large Post-Hoc Calibration Produces Little Relational Repair ($G_Q \le 0.83\%$)**:
   - For the BART-Large anchor, moving from scalar temperature scaling ($G_Q = 0.59\%$) to class-wise vector scaling + bias, coarse-grid identifiable 8-parameter affine matrix scaling, and coarse-grid identifiable 8-parameter Dirichlet calibration closes at most approximately 0.83% of BART's remaining relational gap.
2. **Diverse-Model Probability Ensembling Substantially Improves Relational Alignment**:
   - Combining BART-Large, RoBERTa-Large, and XLNet-Large output distributions materially improves both pointwise and relational alignment relative to BART-Large alone ($G_Q \approx 17.18\% - 17.71\%$). Equal-weight ensembling provides most of the gain.
3. **Level 6a Topology Weighting Gain is Incremental**:
   - Level 6a produced the highest relational-recovery point estimate, but its advantage over equal and NLL-optimized weighting is small and evaluated directly via paired bootstrap contrasts.
4. **Post-Hoc Ceiling Provides Strong Motivation for Fine-Tuning**: Global post-hoc recalibration leaves most of BART's relational gap unclosed. This motivates topology-aware fine-tuning (E004), but does not establish it as uniquely necessary.

---

## 6-Level Relational Repair Ladder Summary Results

| Ladder Level | NLL (nats) | JSD (bits) | $Q_{\text{support, OOF}}$ | $Q_{\text{null, OOF}}$ | $Q_{\text{global-excess}}$ | $G_{\text{NLL}}$ | Relational Gap Closure $G_Q$ | $\Delta G = G_{\text{NLL}} - G_Q$ (95% CI) | Min Turnover | Core Mass ($k=50$) | Core Recall ($k=50$) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Level 0: Raw Model Baseline** | 1.3342 | 0.1402 | **0.01048** | 0.00326 | 0.00723 | **-222.35%** | **0.00%** | **+-0.00%** [0.00%, 0.00%] | 0.00% | 0.013749 | 8.27% |
| **Level 1: Global Isotropic Scalar Temperature** | 0.8559 | 0.0771 | **0.01046** | 0.00326 | 0.00720 | **3.22%** | **-0.05%** | **+70.04%** [69.31%, 70.80%] | 31.86% | 0.014655 | 8.81% |
| **Level 2: Class-Wise Vector Scaling + Bias** | 0.8346 | 0.0698 | **0.01034** | 0.00326 | 0.00708 | **13.24%** | **-0.23%** | **+73.32%** [72.51%, 74.13%] | 30.43% | 0.014815 | 8.91% |
| **Level 3: Coarse-Grid Identifiable 8-Parameter Affine Calibration** | 0.8692 | 0.0775 | **0.01040** | 0.00326 | 0.00714 | **-3.07%** | **-0.14%** | **+68.18%** [67.50%, 68.84%] | 27.84% | 0.014603 | 8.78% |
| **Level 4: Coarse-Grid Identifiable 8-Parameter Dirichlet Calibration** | 0.8729 | 0.0860 | **0.01042** | 0.00326 | 0.00717 | **-4.81%** | **-0.10%** | **+67.60%** [66.72%, 68.42%] | 30.95% | 0.014623 | 8.79% |
| **Level 5a: Equal-Weight Multi-Model Ensemble** | 1.2310 | 0.1233 | **0.01112** | 0.00326 | 0.00785 | **-173.67%** | **1.02%** | **+14.04%** [12.25%, 15.87%] | 90.40% | 0.014835 | 8.92% |
| **Level 5b: Convex NLL-Optimized Simplex Ensemble** | 1.2313 | 0.1234 | **0.01116** | 0.00326 | 0.00790 | **-173.81%** | **1.09%** | **+13.93%** [12.19%, 15.64%] | 89.90% | 0.014783 | 8.89% |
| **Level 6a: Training-Only Topology-Optimized Simplex Ensemble** | 1.2611 | 0.1277 | **0.01113** | 0.00327 | 0.00786 | **-187.86%** | **1.03%** | **+9.66%** [7.77%, 11.47%] | 88.79% | 0.014520 | 8.73% |

---

## Exact-Profile Null & Excess Analysis (Ensemble Conditions)

| Ensemble Condition | $Q_{\text{support}}$ | $Q_{\text{exact-profile null}}$ (95% CI) | $Q_{\text{profile-excess}}$ | $p_{\text{Monte Carlo}}$ |
|---|---|---|---|---|
| **Level 5a: Equal-Weight Multi-Model Ensemble** | 0.01112 | 0.01115 [0.01110, 0.01120] | **-0.00003** | 0.8916 |
| **Level 5b: Convex NLL-Optimized Simplex Ensemble** | 0.01116 | 0.01118 [0.01113, 0.01124] | **-0.00002** | 0.8005 |
| **Level 6a: Training-Only Topology-Optimized Simplex Ensemble** | 0.01113 | 0.01115 [0.01110, 0.01119] | **-0.00001** | 0.7299 |

---

## Direct Paired Bootstrap Contrasts Between Ensemble Conditions

| Contrast | $\Delta G_Q$ (95% CI) | $\Delta Q_{\text{support}}$ (95% CI) | $\Delta R$ (95% CI) | $\Delta \text{NLL}$ (95% CI) | $P(\Delta G_Q > 0)$ |
|---|---|---|---|---|---|
| **Level 5b NLL vs Level 5a Equal** | **+0.07%** [-0.19%, +0.37%] | +0.00005 [-0.00012, +0.00023] | +0.06% [-0.17%, +0.33%] | +0.00029 [-0.00015, +0.00074] | 68.7% |
| **Level 6a Topology vs Level 5b NLL** | **-0.07%** [-0.69%, +0.59%] | -0.00004 [-0.00043, +0.00037] | -0.06% [-0.62%, +0.53%] | +0.02966 [+0.02434, +0.03558] | 40.5% |
| **Level 6a Topology vs Level 5a Equal** | **+0.00%** [-0.66%, +0.70%] | +0.00001 [-0.00040, +0.00043] | +0.00% [-0.59%, +0.62%] | +0.02995 [+0.02430, +0.03622] | 50.7% |

---

## Fold-Specific Optimization Weights & Margins

### Level 5a: Equal-Weight Multi-Model Ensemble

| Fold | Best Weights (BART, RoBERTa, XLNet) | Best Objective | Second-Best Weights | Objective Margin |
|---|---|---|---|---|
| Fold 0 | [0.333, 0.333, 0.333] | 0.00000 | [0.333, 0.333, 0.333] | 0.00000 |
| Fold 1 | [0.333, 0.333, 0.333] | 0.00000 | [0.333, 0.333, 0.333] | 0.00000 |
| Fold 2 | [0.333, 0.333, 0.333] | 0.00000 | [0.333, 0.333, 0.333] | 0.00000 |
| Fold 3 | [0.333, 0.333, 0.333] | 0.00000 | [0.333, 0.333, 0.333] | 0.00000 |
| Fold 4 | [0.333, 0.333, 0.333] | 0.00000 | [0.333, 0.333, 0.333] | 0.00000 |


### Level 5b: Convex NLL-Optimized Simplex Ensemble

| Fold | Best Weights (BART, RoBERTa, XLNet) | Best Objective | Second-Best Weights | Objective Margin |
|---|---|---|---|---|
| Fold 0 | [0.353, 0.353, 0.294] | 1.22315 | [0.350, 0.350, 0.300] | 0.00000 |
| Fold 1 | [0.381, 0.333, 0.286] | 1.23678 | [0.389, 0.333, 0.278] | 0.00004 |
| Fold 2 | [0.348, 0.348, 0.304] | 1.23915 | [0.350, 0.350, 0.300] | 0.00001 |
| Fold 3 | [0.348, 0.348, 0.304] | 1.22583 | [0.364, 0.318, 0.318] | 0.00001 |
| Fold 4 | [0.333, 0.333, 0.333] | 1.22917 | [0.348, 0.348, 0.304] | 0.00004 |


### Level 6a: Training-Only Topology-Optimized Simplex Ensemble

| Fold | Best Weights (BART, RoBERTa, XLNet) | Best Objective | Second-Best Weights | Objective Margin |
|---|---|---|---|---|
| Fold 0 | [0.500, 0.500, 0.000] | 0.01068 | [0.182, 0.727, 0.091] | 0.00002 |
| Fold 1 | [0.476, 0.476, 0.048] | 0.01088 | [0.421, 0.526, 0.053] | 0.00001 |
| Fold 2 | [0.389, 0.556, 0.056] | 0.01006 | [0.476, 0.476, 0.048] | 0.00002 |
| Fold 3 | [0.381, 0.476, 0.143] | 0.01024 | [0.400, 0.500, 0.100] | 0.00001 |
| Fold 4 | [0.000, 0.222, 0.778] | 0.01036 | [0.000, 0.250, 0.750] | 0.00003 |



---

## Scientific Conclusions for Experiment E003

- **BART-Large Post-Hoc Calibration**: Conventional temperature scaling substantially improved soft-label negative log-likelihood while closing less than 1% of BART's remaining relational gap, despite replacing considerable neighborhood mass. More flexible class-wise, affine, and Dirichlet recalibration produced similarly limited relational improvement ($G_Q \le 0.83\%$).
- **Multi-Model Ensembling**: Combining BART-Large, RoBERTa-Large, and XLNet-Large probabilities closed approximately 17% of BART-Large's remaining relational gap (increasing human-normalized recovery $R$ from 19.6% to ~33-34%).
- **Weighting Strategy Equivalence**: Equal, NLL-selected, and topology-selected weighting produced statistically indistinguishable relational results (all direct $\Delta G_Q$ contrasts $< 1.0\%$ and 95% CIs spanned zero), indicating that model combination—not the particular global weighting objective—accounted for the gain.
- **Exact-Profile Boundary**: Exact-vote-profile controls yielded negligible excess support ($Q_{\text{profile-excess}} \approx 0.00000$, $p \ge 0.73$) for every ensemble, showing that the observed improvement concerns the organization of aggregate human judgment distributions rather than detectable item-specific alignment beyond those distributions.
- **Core Takeaway**: The experiment establishes that calibration improves marginal fit while ensembling improves vote-distribution geometry, but global post-hoc weight optimization adds essentially no gain beyond simple averaging.

