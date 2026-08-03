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

1. **BART-Large Post-Hoc Calibration Produces Little Relational Repair ($G_Q \le 0.83\%$)**:
   - For the BART-Large anchor, moving from scalar temperature scaling ($G_Q = 0.59\%$) to class-wise vector scaling + bias, coarse-grid identifiable 8-parameter affine matrix scaling, and coarse-grid identifiable 8-parameter Dirichlet calibration closes at most approximately 0.83% of BART's remaining relational gap.
2. **Diverse-Model Probability Ensembling Substantially Improves Relational Alignment**:
   - Combining BART-Large, RoBERTa-Large, and XLNet-Large output distributions materially improves both pointwise and relational alignment relative to BART-Large alone. Equal-weight ensembling provides most of the gain.
3. **Post-Hoc Ceiling Provides Strong Motivation for Fine-Tuning**: The tested global post-hoc methods leave most of BART's remaining relational gap unclosed. This provides strong motivation for topology-aware fine-tuning (E004), but does not establish it as uniquely necessary.

---

## 6-Level Relational Repair Ladder Summary Results

| Ladder Level | NLL (nats) | JSD (bits) | $Q_{\text{support, OOF}}$ | $Q_{\text{null, OOF}}$ | $Q_{\text{global-excess}}$ | $G_{\text{NLL}}$ | Relational Gap Closure $G_Q$ | $\Delta G = G_{\text{NLL}} - G_Q$ (95% CI) | Min Turnover | Core Mass ($k=50$) | Core Recall ($k=50$) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Level 0: Raw Model Baseline** | 0.8627 | 0.0420 | **0.01681** | 0.00329 | 0.01352 | **-0.00%** | **-0.00%** | **+0.00%** [0.00%, 0.00%] | 0.00% | 0.022923 | 13.78% |
| **Level 1: Global Isotropic Scalar Temperature** | 0.8099 | 0.0578 | **0.01714** | 0.00330 | 0.01384 | **24.89%** | **0.59%** | **+24.25%** [21.70%, 26.72%] | 13.36% | 0.021921 | 13.18% |
| **Level 2: Class-Wise Vector Scaling + Bias** | 0.8056 | 0.0556 | **0.01698** | 0.00330 | 0.01368 | **26.93%** | **0.30%** | **+26.55%** [23.71%, 29.38%] | 16.48% | 0.021857 | 13.14% |
| **Level 3: Coarse-Grid Identifiable 8-Parameter Affine Calibration** | 0.8219 | 0.0699 | **0.01727** | 0.00330 | 0.01398 | **19.23%** | **0.83%** | **+18.34%** [14.39%, 22.15%] | 20.41% | 0.021478 | 12.91% |
| **Level 4: Coarse-Grid Identifiable 8-Parameter Dirichlet Calibration** | 0.8409 | 0.0737 | **0.01679** | 0.00330 | 0.01349 | **10.28%** | **-0.05%** | **+10.27%** [6.00%, 14.12%] | 26.90% | 0.020443 | 12.29% |
| **Level 5a: Equal-Weight Multi-Model Ensemble** | 0.7171 | 0.0236 | **0.02636** | 0.00333 | 0.02303 | **68.65%** | **17.18%** | **+51.40%** [49.01%, 53.51%] | 97.08% | 0.035978 | 21.63% |
| **Level 5b: Convex NLL-Optimized Simplex Ensemble** | 0.7162 | 0.0232 | **0.02646** | 0.00334 | 0.02311 | **69.06%** | **17.33%** | **+51.66%** [49.37%, 53.75%] | 96.42% | 0.036852 | 22.16% |
| **Level 6a: Training-Only Topology-Optimized Simplex Ensemble** | 0.7167 | 0.0233 | **0.02666** | 0.00334 | 0.02332 | **68.82%** | **17.71%** | **+51.02%** [48.81%, 53.14%] | 96.10% | 0.037199 | 22.37% |

---

## Scientific Conclusions for Experiment E003

- **Levels 1 to 4 (BART-Large Post-Hoc Calibration)**: The tested scalar, vector, coarse-grid affine, and coarse-grid Dirichlet transformations closed at most approximately $0.83\%$ of BART's remaining relational gap.
- **Levels 5a/5b/6a (Multi-Model Ensembling)**: Combining BART-Large, RoBERTa-Large, and XLNet-Large output distributions materially improves both pointwise and relational alignment relative to BART-Large alone.
- **Core Takeaway**: Diverse-model probability ensembling produces a large and reliable improvement in both pointwise and relational alignment, whereas increasingly flexible BART-Large recalibration produces little relational improvement. This provides strong motivation for topology-aware fine-tuning (E004).

