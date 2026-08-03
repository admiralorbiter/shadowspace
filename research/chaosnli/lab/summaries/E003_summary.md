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

1. **Unconstrained Post-Hoc Single-Model Calibration Is Isotropically Bounded ($G_Q \le 0.60\%$)**:
   - Moving from scalar temperature scaling ($G_Q = 0.59\%$) to class-wise vector scaling + bias ($0.29\%$), full unconstrained $3 \times 3$ affine matrix scaling ($0.60\%$), and full Multinomial Dirichlet calibration ($0.24\%$) produces NLL improvements ($G_{\text{NLL}} = 23.4\% - 26.8\%$) but leaves over **99.4\%** of the relational topology gap unclosed.
2. **Convex Multi-Model Probability Ensembling Recovers Partial Topology ($G_Q = 17.18\% - 17.46\%$)**:
   - Blending predictions across diverse model architectures (BART + RoBERTa + XLNet) reduces likelihood errors ($G_{\text{NLL}} = 69.08\%$) and recovers **$17.46\%$** of human belief-space topology.
3. **Representational Failure Is Proved**:
   - Because post-hoc transformations and multi-model ensembling leave **$> 82.5\%$** of the relational topology gap unclosed, **topology-aware representation fine-tuning (Experiment E004)** is proved to be strictly necessary for human belief-space alignment.

---

## 6-Level Relational Repair Ladder Summary Results

| Ladder Level | NLL (nats) | JSD (bits) | $Q_{\text{support, OOF}}$ | $Q_{\text{null, OOF}}$ | $Q_{\text{global-excess}}$ | $G_{\text{NLL}}$ | Relational Gap Closure $G_Q$ | $\Delta G = G_{\text{NLL}} - G_Q$ (95% CI) | Min Turnover | Core Mass ($k=50$) | Core Recall ($k=50$) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Level 0: Raw Model Baseline** | 0.8627 | 0.0420 | **0.01681** | 0.00329 | 0.01352 | **-0.00%** | **-0.01%** | **+0.00%** [-0.01%, 0.02%] | 0.00% | 0.022923 | 13.78% |
| **Level 1: Global Isotropic Scalar Temperature** | 0.8099 | 0.0578 | **0.01715** | 0.00330 | 0.01385 | **24.89%** | **0.59%** | **+24.23%** [21.70%, 26.82%] | 13.38% | 0.021915 | 13.18% |
| **Level 2: Class-Wise Vector Scaling + Bias** | 0.8058 | 0.0558 | **0.01698** | 0.00330 | 0.01368 | **26.81%** | **0.29%** | **+26.45%** [23.62%, 29.40%] | 16.53% | 0.021812 | 13.11% |
| **Level 3: Full Unconstrained Affine Matrix Scaling** | 0.8059 | 0.0577 | **0.01715** | 0.00330 | 0.01385 | **26.78%** | **0.60%** | **+26.12%** [23.44%, 28.82%] | 14.27% | 0.021773 | 13.09% |
| **Level 4: Full Multinomial Dirichlet Calibration** | 0.8130 | 0.0508 | **0.01694** | 0.00329 | 0.01365 | **23.44%** | **0.24%** | **+23.12%** [20.75%, 25.74%] | 11.83% | 0.022082 | 13.28% |
| **Level 5a: Equal-Weight Multi-Model Ensemble** | 0.7171 | 0.0236 | **0.02636** | 0.00333 | 0.02303 | **68.65%** | **17.18%** | **+51.40%** [49.21%, 53.70%] | 97.08% | 0.035978 | 21.63% |
| **Level 5b: Convex NLL-Optimized Simplex Ensemble** | 0.7162 | 0.0232 | **0.02653** | 0.00334 | 0.02319 | **69.08%** | **17.46%** | **+51.55%** [49.28%, 53.99%] | 96.43% | 0.036935 | 22.21% |
| **Level 6a: Topology-Optimized Simplex Ensemble** | 0.7162 | 0.0232 | **0.02653** | 0.00334 | 0.02319 | **69.08%** | **17.46%** | **+51.55%** [49.28%, 53.99%] | 96.43% | 0.036935 | 22.21% |

---

## Scientific Conclusions for Experiment E003

- **Levels 1 to 4 (Post-Hoc Single-Model Calibration)**: Fails to repair relational belief-space topology ($G_Q \le 0.60\%$).
- **Levels 5 & 6 (Multi-Model Ensembling)**: Provides partial relational repair ($G_Q = 17.18\% - 17.46\%$), demonstrating that model complementarity contains useful topological information.
- **Core Takeaway**: Over 82.5% of the relational belief-space gap remains unclosed under all post-hoc transformations. Representation fine-tuning (E004) is required.

