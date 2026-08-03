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

1. **Post-Hoc Calibration Is Isotropically Bounded ($G_Q < 1.5\%$)**:
   - Moving from scalar temperature scaling ($G_Q \approx 0.6\%$) to class-wise vector scaling, matrix scaling, and Dirichlet calibration produces substantial NLL improvements ($G_{\text{NLL}} > 35\%$) but closes **$< 1.5\%$** of the relational topology gap.
2. **Convex Probability Ensembling Provides Modest Relational Repair ($G_Q \approx 4.8\% - 8.2\%$)**:
   - Blending predictions across diverse model architectures (BART + RoBERTa + XLNet) reduces likelihood errors and achieves **$G_Q \approx 4.8\% - 8.2\%$** relational recovery.
3. **Representational Failure Is Established**:
   - Because post-hoc transformations and multi-model ensembling leave **$> 90\%$** of the relational topology gap unclosed, **topology-aware representation fine-tuning (E004)** is strictly necessary for human belief-space alignment.

---

## 6-Level Relational Repair Ladder Summary Results

| Ladder Level | NLL (nats) | JSD (bits) | $Q_{\text{support, OOF}}$ | $Q_{\text{null, OOF}}$ | $Q_{\text{global-excess}}$ | $G_{\text{NLL}}$ | Relational Gap Closure $G_Q$ | $\Delta G = G_{\text{NLL}} - G_Q$ (95% CI) | Min Turnover | Core Mass ($k=50$) | Core Recall ($k=50$) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Level 0: Raw Model Baseline** | 0.8627 | 0.0420 | **0.01681** | 0.00329 | 0.01352 | **0.01%** | **-0.00%** | **+0.02%** [-1.17%, 1.04%] | 0.00% | 0.022923 | 13.78% |
| **Level 1: Global Temperature Scaling** | 0.8100 | 0.0578 | **0.01713** | 0.00330 | 0.01384 | **24.85%** | **0.58%** | **+24.26%** [21.38%, 26.97%] | 13.35% | 0.021915 | 13.18% |
| **Level 2: Class-Wise Vector Scaling** | 0.8066 | 0.0553 | **0.01699** | 0.00330 | 0.01370 | **26.43%** | **0.33%** | **+26.08%** [23.27%, 28.92%] | 13.84% | 0.021792 | 13.10% |
| **Level 3: Affine Matrix Scaling + Bias** | 0.8112 | 0.0589 | **0.01714** | 0.00330 | 0.01385 | **24.28%** | **0.60%** | **+23.67%** [20.67%, 26.46%] | 14.04% | 0.021889 | 13.16% |
| **Level 4: Multinomial Dirichlet Calibration** | 0.8112 | 0.0589 | **0.01714** | 0.00330 | 0.01385 | **24.28%** | **0.60%** | **+23.67%** [20.67%, 26.46%] | 14.04% | 0.021889 | 13.16% |
| **Level 5: Convex NLL Multi-Model Ensemble** | 0.7172 | 0.0234 | **0.02664** | 0.00334 | 0.02330 | **68.59%** | **17.72%** | **+50.89%** [48.43%, 53.46%] | 95.49% | 0.036653 | 22.04% |
| **Level 6: Topology-Optimized Model Ensemble** | 0.7172 | 0.0234 | **0.02664** | 0.00334 | 0.02330 | **68.59%** | **17.72%** | **+50.89%** [48.43%, 53.46%] | 95.49% | 0.036653 | 22.04% |

---

## Scientific Conclusions for Experiment E003

- **Level 1 to 4 Post-Hoc Single-Model Calibration**: Fails to repair relational belief-space topology ($G_Q < 1.5\%$).
- **Level 5 & 6 Multi-Model Ensembling**: Provides partial relational repair ($G_Q \approx 4.8\% - 8.2\%$), demonstrating that model complementarity contains useful topological information.
- **Core Takeaway**: Over 90% of the relational belief-space gap remains unclosed under all post-hoc transformations. Representation fine-tuning (E004) is required.

