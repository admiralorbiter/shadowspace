# E003: Relational Repair Capacity of Flexible Post-Hoc Calibration & Ensembling

**Experiment ID**: E003  
**Title**: Relational Repair Capacity of Increasingly Flexible Post-Hoc Transformations & Ensembling  
**Status**: `complete_publication_grade`  
Dataset Release: `chaosnli-canonical-2026-08-02` (N = 3,113 items)  
Cross-Validation: 5-Fold Stratified Coherent Cross-Fitting by (Dataset, Majority Label, Empirical Entropy Quintile)  
Bound E001 Artifact (k=10): `E001-hellinger-k010-expected-fuzzy-support-v1` (SHA-256: `94e483e714d92f03...`)  
Bound E001 Artifact (k=50): `S_hellinger_k050.bin` (SHA-256: `2da027e261d9a74a...`)  
Model Probs Hash: `218cd1246cb3bf79...`  
Human Soft-Label Entropy Floor ($H(p)$): 0.65062 nats  
Human Relational Reference ($Q_{HH}$): 0.07228  

---

## Executive Summary

Experiment **E003** evaluates the **Relational Repair Ladder**: *How much of the human belief-space relational topology gap ($G_Q$) can be recovered through increasingly flexible post-hoc calibration and ensembling techniques BEFORE representational fine-tuning becomes necessary?*

### Key Scientific Findings

1. **Identifiable Post-Hoc Single-Model Calibration Is Isotropically Bounded ($G_Q \le 0.83\%$)**:
   - Moving from scalar temperature scaling ($G_Q = 0.59\%$) to class-wise vector scaling + bias ($0.30\%$), full 8-parameter reference-class affine matrix scaling ($0.83\%$), and full 8-parameter Multinomial Dirichlet calibration ($-0.05\%$) produces soft-label NLL changes ($G_{\text{NLL}} = 10.3\% - 26.8\%$) but leaves over **99.1\%** of the relational topology gap unclosed.
2. **Topology-Optimized Simplex Ensembling Recovers Maximum Topology ($G_Q = 17.63\%$)**:
   - Level 6a (Topology-Optimized Simplex Ensemble) directly maximizes training-fold excess support $Q_{\text{excess, train}}(\alpha)$, achieving **$G_Q = 17.63\%$** relational recovery and outperforming NLL-optimized ($16.42\%$) and equal-weight ($17.18\%$) ensembling.
3. **Representational Limit Established**: Over **82.3\%** of the relational topology gap remains unclosed under all post-hoc transformations and multi-model probability ensembling.

---

## 6-Level Relational Repair Ladder Summary Results

| Ladder Level | NLL (nats) | JSD (bits) | $Q_{\text{support, OOF}}$ | $Q_{\text{null, OOF}}$ | $Q_{\text{global-excess}}$ | $G_{\text{NLL}}$ | Relational Gap Closure $G_Q$ | $\Delta G = G_{\text{NLL}} - G_Q$ (95% CI) | Min Turnover | Core Mass ($k=50$) | Core Recall ($k=50$) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Level 0: Raw Model Baseline** | 0.8627 | 0.0420 | **0.01681** | 0.00329 | 0.01352 | **-0.00%** | **-0.00%** | **+0.00%** [0.00%, 0.00%] | 0.00% | 0.022923 | 13.78% |
| **Level 1: Global Isotropic Scalar Temperature** | 0.8100 | 0.0578 | **0.01714** | 0.00330 | 0.01384 | **24.86%** | **0.59%** | **+24.22%** [21.43%, 26.68%] | 13.38% | 0.021927 | 13.18% |
| **Level 2: Class-Wise Vector Scaling + Bias** | 0.8059 | 0.0558 | **0.01698** | 0.00330 | 0.01368 | **26.79%** | **0.30%** | **+26.41%** [23.30%, 29.30%] | 16.54% | 0.021850 | 13.14% |
| **Level 3: Full 8-Parameter Affine Matrix Scaling** | 0.8219 | 0.0699 | **0.01727** | 0.00330 | 0.01398 | **19.23%** | **0.83%** | **+18.34%** [13.85%, 22.14%] | 20.41% | 0.021478 | 12.91% |
| **Level 4: Full 8-Parameter Multinomial Dirichlet Calibration** | 0.8409 | 0.0737 | **0.01679** | 0.00330 | 0.01349 | **10.28%** | **-0.05%** | **+10.29%** [5.91%, 14.14%] | 26.90% | 0.020443 | 12.29% |
| **Level 5a: Equal-Weight Multi-Model Ensemble** | 0.7171 | 0.0236 | **0.02636** | 0.00333 | 0.02303 | **68.65%** | **17.18%** | **+51.45%** [49.13%, 53.56%] | 97.08% | 0.035978 | 21.63% |
| **Level 5b: Convex NLL-Optimized Simplex Ensemble** | 0.7891 | 0.0305 | **0.02595** | 0.00334 | 0.02261 | **34.68%** | **16.42%** | **+18.21%** [14.01%, 22.14%] | 96.24% | 0.034211 | 20.57% |
| **Level 6a: Topology-Optimized Simplex Ensemble** | 0.7167 | 0.0233 | **0.02661** | 0.00333 | 0.02328 | **68.84%** | **17.63%** | **+51.21%** [48.86%, 53.45%] | 95.90% | 0.037359 | 22.46% |

---

## Scientific Conclusions for Experiment E003

- **Levels 1 to 4 (Post-Hoc Single-Model Calibration)**: Fails to repair relational belief-space topology ($G_Q \le 0.83\%$).
- **Level 6a (Topology-Optimized Simplex Ensemble)**: Maximizes human relational topology recovery ($G_Q = 17.63\%$, $G_{\text{NLL}} = 68.84\%$).
- **Core Takeaway**: Over 82.3% of the relational belief-space gap remains unclosed under all post-hoc transformations. Topology-aware representation fine-tuning (E004) is required.

