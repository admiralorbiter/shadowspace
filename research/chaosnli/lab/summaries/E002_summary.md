# E002: Pointwise Soft-Label Calibration vs. Relational Topology (Publication-Grade Cross-Fitted Pass)

**Experiment ID**: E002  
**Title**: Pointwise Soft-Label Calibration vs. Relational Neighborhood Recovery  
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

Experiment **E002** evaluates **Hypothesis H2**: *Does temperature scaling improve marginal probability alignment (NLL) without proportionately recovering relational human belief-space topology ($Q_{\text{support}}$)?*

### Rigorous Out-of-Fold Methodology Applied

1. **Coherent Per-Fold Graphs (No Temperature Averaging)**: Retains fold-specific temperatures ($T_{\text{NLL}, f}, T_{\text{JS}, f}, T_{\text{topology}, f}$). For each fold $f$, applies $T_f$ uniformly across all $N=3,113$ items to build a single coherent graph $W^{f, T_f}$, scoring ONLY held-out focal rows $i \in H_f$.
2. **Strict Training-Only Topology Target**: Search optimizes $Q_{\text{excess, train}}(T) = Q_{\text{support, train}}(T) - Q_{\text{null, train}}(T)$ over training items ONLY ($N_{\text{train}} \approx 2,490$) using 500 posterior draws and 250 common stratified permutations per grid candidate.
3. **Independent $k=50$ Core Target**: Core mass and recall metrics are evaluated out-of-fold against the true $k=50$ expected support matrix (`S_hellinger_k050.bin`).
4. **Identity-Normalized Min-Overlap Graph Turnover**: $\text{Turnover}_{\min}(T) = 1 - \frac{1}{Nk} \sum_{f=1}^5 \sum_{i \in H_f} \sum_j \min(W_{ij}^{f, T=1}, W_{ij}^{f, T_f})$, guaranteeing $\text{Turnover}_{\min}(1.0) = 0.00000$ exactly.
5. **1,000 Stratified Focal-Item Paired Bootstrap Iterations**: Computes non-parametric 95% CIs for $\Delta \text{NLL}$, $\Delta \text{JSD}$, $\Delta Q$, and $\Delta G = G_{\text{NLL}} - G_Q$.

### Key Scientific Findings

1. **$H2a_{\text{NLL}}$ Supported ($G_{\text{NLL}} \approx 24.8\% - 56.6\%$)**:
   - Out-of-fold soft-label cross-entropy NLL improves consistently under $T_{\text{NLL}} \approx 1.86 - 3.93$ across all 9 models (95% CIs exclude zero).
2. **$H2a_{\text{JS}}$ Contradicted (JS Divergence Increases)**:
   - Temperature calibration ($T_{\text{NLL}}$) softens probabilities, increasing prediction entropy above 1.1 bits and increasing symmetric JS divergence relative to human targets across all 9 models (95% CIs exclude zero).
3. **$H2b_{\text{NLL}}$ Confirmed ($G_{\text{NLL}} \gg G_Q \le 0.70\%$, 95% CIs Exclude Zero)**:
   - While pointwise likelihood gap closure $G_{\text{NLL}}$ reaches **24.8% to 56.6%**, relational topology gap closure $G_Q$ is **$\le 0.70\%$** across all models ($0.15\% - 0.70\%$). The 95% CI for $\Delta G = G_{\text{NLL}} - G_Q$ excludes zero for every model.
4. **$Q_{\text{profile-excess, OOF}}(T) \approx 0.00000$ Across All Conditions**:
   - Out-of-fold $Q_{\text{profile-excess, OOF}}(T) = Q_{\text{support, OOF}}(T) - Q_{\text{exact-profile-null, OOF}}(T)$ remains $\approx 0.00000$ across all 4 evaluated conditions.

---

## Detailed 5-Fold Coherent Cross-Fitted Model Calibration Results

| Model | $T_{\text{NLL}}$ (mean ± std) | $T_{\text{JSD}}$ (mean ± std) | $T_{\text{topology}}$ (mean ± std) | NLL ($T_{\text{raw}}$) | NLL ($T_{\text{cal}}$) | $G_{\text{NLL}}$ | JSD ($T_{\text{raw}}$) | JSD ($T_{\text{cal}}$) | $Q_{\text{raw, OOF}}$ | $Q_{\text{cal, OOF}}$ | Relational Gap Closure $G_Q$ | $\Delta G$ (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bart-large** | 1.86±0.03 | 0.83±0.00 | 4.60±1.42 | 0.8627 | **0.8101** | **24.81%** | 0.0420 | 0.0578 | **0.01681** | **0.01716** | **0.63%** | **+24.18%** [21.52%, 26.70%] |
| **roberta-large** | 2.16±0.02 | 0.85±0.01 | 6.26±1.94 | 0.9082 | **0.8259** | **31.96%** | 0.0489 | 0.0659 | **0.01492** | **0.01521** | **0.51%** | **+31.44%** [28.97%, 34.14%] |
| **xlnet-large** | 2.35±0.02 | 0.83±0.00 | 4.66±1.57 | 0.9474 | **0.8452** | **34.45%** | 0.0567 | 0.0748 | **0.01334** | **0.01372** | **0.62%** | **+33.85%** [31.16%, 36.39%] |
| **albert-xxlarge** | 2.59±0.02 | 0.86±0.01 | 6.52±3.32 | 0.9892 | **0.8578** | **38.82%** | 0.0636 | 0.0804 | **0.01153** | **0.01180** | **0.44%** | **+38.36%** [35.57%, 40.82%] |
| **bert-large** | 3.00±0.05 | 0.86±0.01 | 6.88±3.09 | 1.0658 | **0.8775** | **45.35%** | 0.0739 | 0.0892 | **0.01053** | **0.01076** | **0.36%** | **+45.02%** [42.86%, 47.16%] |
| **roberta-base** | 3.12±0.02 | 0.90±0.01 | 5.74±2.85 | 1.0938 | **0.8849** | **47.14%** | 0.0808 | 0.0930 | **0.01033** | **0.01044** | **0.16%** | **+46.99%** [44.74%, 49.12%] |
| **xlnet-base** | 3.53±0.02 | 0.94±0.00 | 7.44±2.26 | 1.1806 | **0.8975** | **53.42%** | 0.0891 | 0.0980 | **0.00934** | **0.00978** | **0.70%** | **+52.67%** [50.55%, 54.70%] |
| **distilbert** | 3.67±0.03 | 0.97±0.01 | 5.26±1.63 | 1.2114 | **0.9088** | **53.96%** | 0.0954 | 0.1025 | **0.00865** | **0.00886** | **0.35%** | **+53.63%** [51.55%, 55.58%] |
| **bert-base** | 3.93±0.01 | 2.72±0.03 | 4.96±2.82 | 1.2696 | **0.9193** | **56.59%** | 0.1062 | 0.1075 | **0.00786** | **0.00809** | **0.36%** | **+56.27%** [54.39%, 58.14%] |

---

## Out-of-Fold Condition Comparison & Structural Graph Turnover (k=10 Hellinger & k=50 Core)

| Model | Condition | NLL (nats) | JSD (bits) | $Q_{\text{support, OOF}}$ | $Q_{\text{null, OOF}}$ | $Q_{\text{global-excess}}$ | Min Turnover | Core Mass ($k=50$) | Core Recall ($k=50$) | Avg Entropy | Distance Var |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **bart-large** | T_raw (1.0) | 0.8627 | 0.0420 | **0.01681** | 0.00329 | 0.01352 | 0.00% | 0.022923 | 13.78% | 0.960 | 0.04873 |
| **bart-large** | T_NLL (calibrated) | 0.8101 | 0.0578 | **0.01716** | 0.00330 | 0.01386 | 13.34% | 0.021927 | 13.18% | 1.168 | 0.03617 |
| **bart-large** | T_JSD (pointwise oracle) | 0.9088 | 0.0406 | **0.01670** | 0.00329 | 0.01341 | 3.61% | 0.023277 | 14.00% | 0.881 | 0.05359 |
| **bart-large** | T_topology (relational oracle) | 0.8675 | 0.0906 | **0.01721** | 0.00330 | 0.01391 | 25.99% | 0.021574 | 12.97% | 1.401 | 0.01636 |
| **roberta-large** | T_raw (1.0) | 0.9082 | 0.0489 | **0.01492** | 0.00329 | 0.01163 | 0.00% | 0.018439 | 11.09% | 0.945 | 0.05078 |
| **roberta-large** | T_NLL (calibrated) | 0.8259 | 0.0659 | **0.01521** | 0.00329 | 0.01192 | 18.57% | 0.017629 | 10.60% | 1.191 | 0.03478 |
| **roberta-large** | T_JSD (pointwise oracle) | 0.9584 | 0.0479 | **0.01498** | 0.00329 | 0.01169 | 3.25% | 0.018721 | 11.26% | 0.879 | 0.05483 |
| **roberta-large** | T_topology (relational oracle) | 0.9018 | 0.1049 | **0.01551** | 0.00329 | 0.01222 | 31.60% | 0.017115 | 10.29% | 1.459 | 0.01101 |
| **xlnet-large** | T_raw (1.0) | 0.9474 | 0.0567 | **0.01334** | 0.00327 | 0.01007 | 0.00% | 0.016621 | 9.99% | 0.953 | 0.05009 |
| **xlnet-large** | T_NLL (calibrated) | 0.8452 | 0.0748 | **0.01372** | 0.00328 | 0.01044 | 19.89% | 0.015907 | 9.56% | 1.219 | 0.03282 |
| **xlnet-large** | T_JSD (pointwise oracle) | 1.0092 | 0.0557 | **0.01332** | 0.00327 | 0.01005 | 3.45% | 0.016698 | 10.04% | 0.882 | 0.05426 |
| **xlnet-large** | T_topology (relational oracle) | 0.8843 | 0.0962 | **0.01381** | 0.00328 | 0.01053 | 27.90% | 0.015670 | 9.42% | 1.383 | 0.01777 |
| **albert-xxlarge** | T_raw (1.0) | 0.9892 | 0.0636 | **0.01153** | 0.00327 | 0.00826 | 0.00% | 0.014243 | 8.56% | 0.947 | 0.05115 |
| **albert-xxlarge** | T_NLL (calibrated) | 0.8578 | 0.0804 | **0.01180** | 0.00327 | 0.00853 | 22.49% | 0.013935 | 8.38% | 1.237 | 0.03117 |
| **albert-xxlarge** | T_JSD (pointwise oracle) | 1.0470 | 0.0629 | **0.01154** | 0.00326 | 0.00828 | 2.86% | 0.014526 | 8.73% | 0.890 | 0.05440 |
| **albert-xxlarge** | T_topology (relational oracle) | 1.1453 | 0.1164 | **0.01197** | 0.00328 | 0.00870 | 32.63% | 0.014160 | 8.51% | 1.303 | 0.02482 |
| **bert-large** | T_raw (1.0) | 1.0658 | 0.0739 | **0.01053** | 0.00328 | 0.00726 | 0.00% | 0.013183 | 7.93% | 0.935 | 0.05262 |
| **bert-large** | T_NLL (calibrated) | 0.8775 | 0.0892 | **0.01076** | 0.00328 | 0.00747 | 25.88% | 0.012869 | 7.74% | 1.265 | 0.02827 |
| **bert-large** | T_JSD (pointwise oracle) | 1.1342 | 0.0733 | **0.01053** | 0.00328 | 0.00725 | 3.22% | 0.013145 | 7.90% | 0.884 | 0.05549 |
| **bert-large** | T_topology (relational oracle) | 0.9274 | 0.1134 | **0.01077** | 0.00328 | 0.00749 | 32.34% | 0.012785 | 7.69% | 1.425 | 0.01375 |
| **roberta-base** | T_raw (1.0) | 1.0938 | 0.0808 | **0.01033** | 0.00326 | 0.00707 | 0.00% | 0.013241 | 7.96% | 0.931 | 0.05266 |
| **roberta-base** | T_NLL (calibrated) | 0.8849 | 0.0930 | **0.01044** | 0.00326 | 0.00717 | 26.31% | 0.012952 | 7.79% | 1.276 | 0.02697 |
| **roberta-base** | T_JSD (pointwise oracle) | 1.1430 | 0.0806 | **0.01028** | 0.00326 | 0.00703 | 2.21% | 0.013228 | 7.95% | 0.897 | 0.05459 |
| **roberta-base** | T_topology (relational oracle) | 0.9188 | 0.1086 | **0.01038** | 0.00326 | 0.00711 | 30.05% | 0.012920 | 7.77% | 1.387 | 0.01711 |
| **xlnet-base** | T_raw (1.0) | 1.1806 | 0.0891 | **0.00934** | 0.00326 | 0.00609 | 0.00% | 0.011076 | 6.66% | 0.905 | 0.05503 |
| **xlnet-base** | T_NLL (calibrated) | 0.8975 | 0.0980 | **0.00978** | 0.00326 | 0.00652 | 29.03% | 0.011005 | 6.62% | 1.295 | 0.02466 |
| **xlnet-base** | T_JSD (pointwise oracle) | 1.2193 | 0.0890 | **0.00937** | 0.00326 | 0.00612 | 1.55% | 0.011063 | 6.65% | 0.883 | 0.05624 |
| **xlnet-base** | T_topology (relational oracle) | 0.9407 | 0.1196 | **0.00981** | 0.00326 | 0.00655 | 34.12% | 0.011057 | 6.65% | 1.456 | 0.01071 |
| **distilbert** | T_raw (1.0) | 1.2114 | 0.0954 | **0.00865** | 0.00325 | 0.00539 | 0.00% | 0.010337 | 6.22% | 0.914 | 0.05424 |
| **distilbert** | T_NLL (calibrated) | 0.9088 | 0.1025 | **0.00886** | 0.00325 | 0.00561 | 30.14% | 0.010157 | 6.11% | 1.311 | 0.02329 |
| **distilbert** | T_JSD (pointwise oracle) | 1.2299 | 0.0954 | **0.00864** | 0.00325 | 0.00539 | 0.73% | 0.010279 | 6.18% | 0.904 | 0.05477 |
| **distilbert** | T_topology (relational oracle) | 0.9222 | 0.1108 | **0.00879** | 0.00325 | 0.00554 | 32.88% | 0.010183 | 6.12% | 1.394 | 0.01605 |
| **bert-base** | T_raw (1.0) | 1.2696 | 0.1062 | **0.00786** | 0.00324 | 0.00462 | 0.00% | 0.008448 | 5.08% | 0.905 | 0.05512 |
| **bert-base** | T_NLL (calibrated) | 0.9193 | 0.1075 | **0.00809** | 0.00324 | 0.00485 | 31.06% | 0.008609 | 5.18% | 1.326 | 0.02178 |
| **bert-base** | T_JSD (pointwise oracle) | 0.9372 | 0.1048 | **0.00795** | 0.00325 | 0.00470 | 25.99% | 0.008442 | 5.08% | 1.202 | 0.03298 |
| **bert-base** | T_topology (relational oracle) | 0.9618 | 0.1143 | **0.00795** | 0.00324 | 0.00470 | 28.87% | 0.008558 | 5.15% | 1.318 | 0.02218 |

---

## Inferential Conclusions for Hypothesis H2

- **H2a (NLL Reduction)**: **SUPPORTED**. Out-of-fold soft-label cross-entropy ($NLL$) is consistently reduced under $T_{\text{NLL}}$ ($G_{\text{NLL}} = 24.8\% - 56.6\%$, 95% CIs exclude zero).
- **H2a (JSD Alignment)**: **REVERSED**. Temperature scaling increases prediction entropy, worsening symmetric JS divergence relative to human targets.
- **H2b (Relational Disconnect)**: **CONFIRMED**. Pointwise likelihood gap closure $G_{\text{NLL}}$ (24.8%–56.6%) dramatically exceeds relational topology gap closure ($G_Q \le 0.70\%$). Non-parametric bootstrap CIs for $\Delta G = G_{\text{NLL}} - G_Q$ exclude zero for all 9 models.
- **Implication for E003**: Post-hoc scalar temperature scaling alters pointwise entropy while leaving nearest-neighbor relational belief-space topology locked. E003 (Relational Topology Fine-Tuning & Representation Alignment) is necessary to close the relational topology gap.

