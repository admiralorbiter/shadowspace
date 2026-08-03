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
| **bart-large** | 1.86±0.02 | 0.83±0.00 | 4.14±1.14 | 0.8627 | **0.8100** | **24.86%** | 0.0420 | 0.0578 | **0.01681** | **0.01715** | **0.60%** | **+24.21%** [21.58%, 26.63%] |
| **roberta-large** | 2.16±0.02 | 0.85±0.01 | 7.12±1.73 | 0.9082 | **0.8259** | **31.94%** | 0.0489 | 0.0659 | **0.01492** | **0.01520** | **0.49%** | **+31.46%** [28.92%, 33.88%] |
| **xlnet-large** | 2.35±0.02 | 0.83±0.01 | 4.98±2.48 | 0.9474 | **0.8451** | **34.47%** | 0.0567 | 0.0748 | **0.01334** | **0.01373** | **0.65%** | **+33.75%** [31.33%, 36.18%] |
| **albert-xxlarge** | 2.59±0.02 | 0.86±0.01 | 4.97±4.00 | 0.9892 | **0.8578** | **38.81%** | 0.0636 | 0.0804 | **0.01153** | **0.01178** | **0.40%** | **+38.35%** [36.02%, 40.61%] |
| **bert-large** | 3.00±0.02 | 0.86±0.01 | 5.92±3.27 | 1.0658 | **0.8773** | **45.40%** | 0.0739 | 0.0892 | **0.01053** | **0.01073** | **0.32%** | **+45.02%** [42.93%, 47.02%] |
| **roberta-base** | 3.12±0.03 | 0.90±0.01 | 2.97±1.63 | 1.0938 | **0.8849** | **47.13%** | 0.0808 | 0.0930 | **0.01033** | **0.01044** | **0.18%** | **+46.91%** [44.74%, 49.08%] |
| **xlnet-base** | 3.53±0.02 | 0.94±0.01 | 6.00±1.67 | 1.1806 | **0.8975** | **53.42%** | 0.0891 | 0.0980 | **0.00934** | **0.00978** | **0.69%** | **+52.70%** [50.79%, 54.60%] |
| **distilbert** | 3.67±0.03 | 0.97±0.01 | 3.84±1.14 | 1.2114 | **0.9088** | **53.97%** | 0.0954 | 0.1025 | **0.00865** | **0.00886** | **0.35%** | **+53.65%** [51.88%, 55.45%] |
| **bert-base** | 3.93±0.04 | 2.72±0.07 | 5.76±3.55 | 1.2696 | **0.9194** | **56.57%** | 0.1062 | 0.1075 | **0.00786** | **0.00809** | **0.36%** | **+56.18%** [54.57%, 57.83%] |

---

## Out-of-Fold Condition Comparison & Structural Graph Turnover (k=10 Hellinger & k=50 Core)

| Model | Condition | NLL (nats) | JSD (bits) | $Q_{\text{support, OOF}}$ | $Q_{\text{null, OOF}}$ | $Q_{\text{global-excess}}$ | Min Turnover | Core Mass ($k=50$) | Core Recall ($k=50$) | Avg Entropy | Distance Var |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **bart-large** | T_raw (1.0) | 0.8627 | 0.0420 | **0.01681** | 0.00329 | 0.01352 | 0.00% | 0.022923 | 13.78% | 0.960 | 0.04873 |
| **bart-large** | T_NLL (calibrated) | 0.8100 | 0.0578 | **0.01715** | 0.00330 | 0.01385 | 13.38% | 0.021927 | 13.18% | 1.168 | 0.03617 |
| **bart-large** | T_JSD (pointwise oracle) | 0.9088 | 0.0406 | **0.01670** | 0.00329 | 0.01341 | 3.60% | 0.023283 | 14.00% | 0.881 | 0.05359 |
| **bart-large** | T_topology (relational oracle) | 0.8583 | 0.0867 | **0.01737** | 0.00330 | 0.01408 | 25.33% | 0.021523 | 12.94% | 1.381 | 0.01809 |
| **roberta-large** | T_raw (1.0) | 0.9082 | 0.0489 | **0.01492** | 0.00329 | 0.01163 | 0.00% | 0.018439 | 11.09% | 0.945 | 0.05078 |
| **roberta-large** | T_NLL (calibrated) | 0.8259 | 0.0659 | **0.01520** | 0.00329 | 0.01191 | 18.55% | 0.017649 | 10.61% | 1.191 | 0.03479 |
| **roberta-large** | T_JSD (pointwise oracle) | 0.9586 | 0.0480 | **0.01501** | 0.00329 | 0.01172 | 3.27% | 0.018696 | 11.24% | 0.879 | 0.05483 |
| **roberta-large** | T_topology (relational oracle) | 0.9175 | 0.1114 | **0.01550** | 0.00329 | 0.01221 | 32.69% | 0.017109 | 10.29% | 1.484 | 0.00873 |
| **xlnet-large** | T_raw (1.0) | 0.9474 | 0.0567 | **0.01334** | 0.00327 | 0.01007 | 0.00% | 0.016621 | 9.99% | 0.953 | 0.05009 |
| **xlnet-large** | T_NLL (calibrated) | 0.8451 | 0.0748 | **0.01373** | 0.00328 | 0.01046 | 19.91% | 0.015907 | 9.56% | 1.219 | 0.03282 |
| **xlnet-large** | T_JSD (pointwise oracle) | 1.0094 | 0.0557 | **0.01333** | 0.00327 | 0.01006 | 3.46% | 0.016685 | 10.03% | 0.882 | 0.05427 |
| **xlnet-large** | T_topology (relational oracle) | 0.8876 | 0.0974 | **0.01377** | 0.00328 | 0.01049 | 27.75% | 0.015618 | 9.39% | 1.378 | 0.01867 |
| **albert-xxlarge** | T_raw (1.0) | 0.9892 | 0.0636 | **0.01153** | 0.00327 | 0.00826 | 0.00% | 0.014243 | 8.56% | 0.947 | 0.05115 |
| **albert-xxlarge** | T_NLL (calibrated) | 0.8578 | 0.0804 | **0.01178** | 0.00327 | 0.00851 | 22.50% | 0.013929 | 8.37% | 1.237 | 0.03117 |
| **albert-xxlarge** | T_JSD (pointwise oracle) | 1.0469 | 0.0630 | **0.01155** | 0.00326 | 0.00828 | 2.85% | 0.014571 | 8.76% | 0.890 | 0.05440 |
| **albert-xxlarge** | T_topology (relational oracle) | 1.4109 | 0.1142 | **0.01184** | 0.00327 | 0.00857 | 33.78% | 0.014180 | 8.53% | 1.063 | 0.04588 |
| **bert-large** | T_raw (1.0) | 1.0658 | 0.0739 | **0.01053** | 0.00328 | 0.00726 | 0.00% | 0.013183 | 7.93% | 0.935 | 0.05262 |
| **bert-large** | T_NLL (calibrated) | 0.8773 | 0.0892 | **0.01073** | 0.00328 | 0.00745 | 25.93% | 0.012849 | 7.73% | 1.266 | 0.02826 |
| **bert-large** | T_JSD (pointwise oracle) | 1.1342 | 0.0733 | **0.01052** | 0.00328 | 0.00724 | 3.24% | 0.013158 | 7.91% | 0.883 | 0.05549 |
| **bert-large** | T_topology (relational oracle) | 1.2037 | 0.1153 | **0.01072** | 0.00328 | 0.00744 | 33.03% | 0.012876 | 7.74% | 1.271 | 0.02630 |
| **roberta-base** | T_raw (1.0) | 1.0938 | 0.0808 | **0.01033** | 0.00326 | 0.00707 | 0.00% | 0.013241 | 7.96% | 0.931 | 0.05266 |
| **roberta-base** | T_NLL (calibrated) | 0.8849 | 0.0930 | **0.01044** | 0.00326 | 0.00718 | 26.31% | 0.012939 | 7.78% | 1.276 | 0.02697 |
| **roberta-base** | T_JSD (pointwise oracle) | 1.1427 | 0.0805 | **0.01028** | 0.00326 | 0.00702 | 2.19% | 0.013286 | 7.99% | 0.897 | 0.05459 |
| **roberta-base** | T_topology (relational oracle) | 1.3518 | 0.1077 | **0.01019** | 0.00326 | 0.00693 | 33.10% | 0.012992 | 7.81% | 1.107 | 0.04647 |
| **xlnet-base** | T_raw (1.0) | 1.1806 | 0.0891 | **0.00934** | 0.00326 | 0.00609 | 0.00% | 0.011076 | 6.66% | 0.905 | 0.05503 |
| **xlnet-base** | T_NLL (calibrated) | 0.8975 | 0.0980 | **0.00978** | 0.00326 | 0.00652 | 29.02% | 0.011012 | 6.62% | 1.295 | 0.02466 |
| **xlnet-base** | T_JSD (pointwise oracle) | 1.2195 | 0.0890 | **0.00937** | 0.00326 | 0.00611 | 1.56% | 0.011057 | 6.65% | 0.883 | 0.05624 |
| **xlnet-base** | T_topology (relational oracle) | 0.9232 | 0.1123 | **0.00977** | 0.00326 | 0.00651 | 33.20% | 0.011047 | 6.64% | 1.427 | 0.01315 |
| **distilbert** | T_raw (1.0) | 1.2114 | 0.0954 | **0.00865** | 0.00325 | 0.00539 | 0.00% | 0.010337 | 6.22% | 0.914 | 0.05424 |
| **distilbert** | T_NLL (calibrated) | 0.9088 | 0.1025 | **0.00886** | 0.00325 | 0.00561 | 30.13% | 0.010177 | 6.12% | 1.311 | 0.02329 |
| **distilbert** | T_JSD (pointwise oracle) | 1.2297 | 0.0954 | **0.00865** | 0.00325 | 0.00539 | 0.70% | 0.010299 | 6.19% | 0.904 | 0.05477 |
| **distilbert** | T_topology (relational oracle) | 0.9236 | 0.1046 | **0.00889** | 0.00325 | 0.00564 | 29.12% | 0.010267 | 6.17% | 1.303 | 0.02396 |
| **bert-base** | T_raw (1.0) | 1.2696 | 0.1062 | **0.00786** | 0.00324 | 0.00462 | 0.00% | 0.008448 | 5.08% | 0.905 | 0.05512 |
| **bert-base** | T_NLL (calibrated) | 0.9194 | 0.1075 | **0.00809** | 0.00324 | 0.00485 | 31.06% | 0.008615 | 5.18% | 1.326 | 0.02178 |
| **bert-base** | T_JSD (pointwise oracle) | 0.9377 | 0.1049 | **0.00793** | 0.00325 | 0.00469 | 25.93% | 0.008474 | 5.10% | 1.202 | 0.03301 |
| **bert-base** | T_topology (relational oracle) | 0.9712 | 0.1186 | **0.00803** | 0.00324 | 0.00479 | 29.46% | 0.008641 | 5.20% | 1.338 | 0.02120 |

---

## Inferential Conclusions for Hypothesis H2

- **H2a (NLL Reduction)**: **SUPPORTED**. Out-of-fold soft-label cross-entropy ($NLL$) is consistently reduced under $T_{\text{NLL}}$ ($G_{\text{NLL}} = 24.8\% - 56.6\%$, 95% CIs exclude zero).
- **H2a (JSD Alignment)**: **REVERSED**. Temperature scaling increases prediction entropy, worsening symmetric JS divergence relative to human targets.
- **H2b (Relational Disconnect)**: **CONFIRMED**. Pointwise likelihood gap closure $G_{\text{NLL}}$ (24.8%–56.6%) dramatically exceeds relational topology gap closure ($G_Q \le 0.70\%$). Non-parametric bootstrap CIs for $\Delta G = G_{\text{NLL}} - G_Q$ exclude zero for all 9 models.
- **Implication for E003**: Post-hoc scalar temperature scaling alters pointwise entropy while leaving nearest-neighbor relational belief-space topology locked. E003 (Relational Topology Fine-Tuning & Representation Alignment) is necessary to close the relational topology gap.

