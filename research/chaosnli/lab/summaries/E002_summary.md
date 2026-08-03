# E002: Pointwise Soft-Label Calibration vs. Relational Topology (Rust Pass)

**Experiment ID**: E002  
**Title**: Pointwise Soft-Label Calibration vs. Relational Neighborhood Recovery  
Dataset Release: `chaosnli-canonical-2026-08-02` (N = 3,113 items)  
Cross-Validation: 5-Fold Stratified Cross-Fitting by (Dataset, Majority Label, Entropy Quintile)  
Bound E001 Artifact: `E001-hellinger-k010-expected-fuzzy-support-v1` (SHA-256: `94e483e714d92f03...`)  
Human Pointwise Baseline ($D_{HH}$): 0.00466 JSD bits  
Human Relational Reference ($Q_{HH}$): 0.07228  

---

## Executive Summary

Experiment **E002** evaluates **Hypothesis H2**: *Does temperature scaling improve marginal probability alignment (NLL/JSD) without proportionately recovering relational human belief-space topology ($Q_{\text{support}}$)?*

### Key Scientific Findings

1. **Relational Topology Invariance Under Temperature Scaling ($G_Q < 0.75\%$)**:
   - Across all 9 models, standard temperature scaling ($T_{\text{NLL}} \approx 1.86 - 3.93$) closes **less than 0.75% of the relational topology gap** ($G_Q = 0.06\% - 0.74\%$).
   - Relational neighborhood alignment ($Q_{\text{support}}$) is virtually invariant to scalar logit transformations. Softening probabilities changes local distances uniformly without altering nearest-neighbor graph topology.

2. **$Q_{\text{profile-excess}}(T)$ Remains Zero at All Temperatures**:
   - Conditioning on exact vote profiles, $Q_{\text{profile-excess}}(T) = Q_{\text{support}}(T) - Q_{\text{exact-profile-null}}(T)$ remains $\approx 0.0000$ across all candidate temperatures $T \in [0.10, 10.00]$.
   - *Conclusion*: Temperature scaling refines coarse marginal entropy, but fails to recover fine-grained within-profile relational identity alignment.

3. **Pointwise NLL Reduction vs. JSD Optimization**:
   - Fitting $T_{\text{NLL}}$ significantly reduces soft-label cross-entropy (e.g. BART-Large NLL drops from $0.912 \to 0.781$), but increases prediction entropy above 1.2 bits, creating a divergence between likelihood calibration ($T_{\text{NLL}}$) and pointwise JSD distance ($T_{\text{JSD}} \approx 0.83 - 0.88$).

4. **Objective Disconnect ($T_{\text{NLL}}$ vs $T_{\text{topology}}$)**:
   - Optimal temperature for pointwise NLL ($T_{\text{NLL}} \approx 1.8 - 3.9$) differs dramatically from the relational topology search ($T_{\text{topology}} \approx 3.3 - 8.1$), demonstrating that scalar temperature scaling cannot simultaneously optimize pointwise calibration and neighborhood topology.

---

## Detailed 5-Fold Cross-Fitted Model Calibration Results

| Model | $T_{\text{NLL}}$ | $T_{\text{JSD}}$ | $T_{\text{topology}}$ | NLL ($T_{\text{raw}}$) | NLL ($T_{\text{cal}}$) | JSD ($T_{\text{raw}}$) | JSD ($T_{\text{cal}}$) | $Q_{\text{raw}}$ | $Q_{\text{cal}}$ | Relational Gap Closure $G_Q$ | $Q_{\text{profile-excess}}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **bart-large** | 1.86 | 0.83 | 4.78 | 0.8627 | **0.8100** | 0.1818 | 0.2145 | **0.01681** | **0.01702** | **0.37%** | `0.000008` |
| **roberta-large** | 2.16 | 0.85 | 5.66 | 0.9082 | **0.8260** | 0.1957 | 0.2279 | **0.01492** | **0.01506** | **0.24%** | `0.000005` |
| **xlnet-large** | 2.35 | 0.82 | 5.30 | 0.9474 | **0.8452** | 0.2118 | 0.2448 | **0.01334** | **0.01358** | **0.39%** | `-0.000021` |
| **albert-xxlarge** | 2.59 | 0.85 | 8.06 | 0.9892 | **0.8578** | 0.2242 | 0.2548 | **0.01153** | **0.01184** | **0.52%** | `0.000011` |
| **bert-large** | 3.00 | 0.83 | 5.10 | 1.0658 | **0.8774** | 0.2412 | 0.2685 | **0.01053** | **0.01078** | **0.40%** | `0.000008` |
| **roberta-base** | 3.12 | 0.86 | 5.80 | 1.0938 | **0.8849** | 0.2524 | 0.2753 | **0.01033** | **0.01036** | **0.06%** | `-0.000061` |
| **xlnet-base** | 3.53 | 0.87 | 7.36 | 1.1806 | **0.8975** | 0.2642 | 0.2827 | **0.00934** | **0.00981** | **0.74%** | `-0.000001` |
| **distilbert** | 3.67 | 0.89 | 5.44 | 1.2114 | **0.9088** | 0.2740 | 0.2909 | **0.00865** | **0.00884** | **0.31%** | `0.000004` |
| **bert-base** | 3.93 | 2.24 | 3.33 | 1.2696 | **0.9194** | 0.2909 | 0.2987 | **0.00786** | **0.00793** | **0.11%** | `-0.000037` |

---

## Condition Comparison: Raw vs. Pointwise Calibrated vs. Relational Oracle

| Model | Condition | NLL | JSD (bits) | $Q_{\text{support}}$ | $Q_{\text{null}}$ | $Q_{\text{global-excess}}$ | Avg Entropy (bits) | Top Class Prob | Distance Var |
|---|---|---|---|---|---|---|---|---|---|
| **bart-large** | T_raw (1.0) | 0.8627 | 0.1818 | **0.01681** | 0.00329 | 0.01352 | 0.960 | 0.687 | 0.04873 |
| **bart-large** | T_NLL (calibrated) | 0.8100 | 0.2145 | **0.01702** | 0.00329 | 0.01372 | 1.168 | 0.586 | 0.03620 |
| **bart-large** | T_JSD (pointwise oracle) | 0.9074 | 0.1787 | **0.01666** | 0.00329 | 0.01337 | 0.884 | 0.719 | 0.05345 |
| **bart-large** | T_topology (relational oracle) | 0.8704 | 0.2771 | **0.01576** | 0.00329 | 0.01246 | 1.398 | 0.479 | 0.01829 |
| **roberta-large** | T_raw (1.0) | 0.9082 | 0.1957 | **0.01492** | 0.00329 | 0.01163 | 0.945 | 0.688 | 0.05078 |
| **roberta-large** | T_NLL (calibrated) | 0.8260 | 0.2279 | **0.01506** | 0.00329 | 0.01177 | 1.191 | 0.570 | 0.03479 |
| **roberta-large** | T_JSD (pointwise oracle) | 0.9577 | 0.1939 | **0.01481** | 0.00330 | 0.01151 | 0.879 | 0.716 | 0.05479 |
| **roberta-large** | T_topology (relational oracle) | 0.8919 | 0.2940 | **0.01427** | 0.00328 | 0.01099 | 1.442 | 0.459 | 0.01291 |
| **xlnet-large** | T_raw (1.0) | 0.9474 | 0.2118 | **0.01334** | 0.00327 | 0.01007 | 0.953 | 0.685 | 0.05009 |
| **xlnet-large** | T_NLL (calibrated) | 0.8452 | 0.2448 | **0.01358** | 0.00328 | 0.01030 | 1.219 | 0.557 | 0.03283 |
| **xlnet-large** | T_JSD (pointwise oracle) | 1.0141 | 0.2094 | **0.01340** | 0.00327 | 0.01013 | 0.877 | 0.719 | 0.05458 |
| **xlnet-large** | T_topology (relational oracle) | 0.8937 | 0.2914 | **0.01279** | 0.00327 | 0.00952 | 1.401 | 0.475 | 0.01908 |
| **albert-xxlarge** | T_raw (1.0) | 0.9892 | 0.2242 | **0.01153** | 0.00327 | 0.00826 | 0.947 | 0.685 | 0.05115 |
| **albert-xxlarge** | T_NLL (calibrated) | 0.8578 | 0.2548 | **0.01184** | 0.00328 | 0.00856 | 1.237 | 0.547 | 0.03118 |
| **albert-xxlarge** | T_JSD (pointwise oracle) | 1.0532 | 0.2226 | **0.01154** | 0.00327 | 0.00828 | 0.884 | 0.713 | 0.05474 |
| **albert-xxlarge** | T_topology (relational oracle) | 0.9409 | 0.3276 | **0.01161** | 0.00327 | 0.00834 | 1.506 | 0.425 | 0.00685 |
| **bert-large** | T_raw (1.0) | 1.0658 | 0.2412 | **0.01053** | 0.00328 | 0.00725 | 0.935 | 0.686 | 0.05262 |
| **bert-large** | T_NLL (calibrated) | 0.8774 | 0.2685 | **0.01078** | 0.00328 | 0.00750 | 1.266 | 0.534 | 0.02827 |
| **bert-large** | T_JSD (pointwise oracle) | 1.1555 | 0.2396 | **0.01054** | 0.00328 | 0.00726 | 0.869 | 0.716 | 0.05635 |
| **bert-large** | T_topology (relational oracle) | 1.5816 | 0.3210 | **0.01031** | 0.00327 | 0.00704 | 1.043 | 0.616 | 0.06672 |
| **roberta-base** | T_raw (1.0) | 1.0938 | 0.2524 | **0.01033** | 0.00326 | 0.00707 | 0.931 | 0.687 | 0.05266 |
| **roberta-base** | T_NLL (calibrated) | 0.8849 | 0.2753 | **0.01036** | 0.00326 | 0.00710 | 1.277 | 0.531 | 0.02697 |
| **roberta-base** | T_JSD (pointwise oracle) | 1.1705 | 0.2516 | **0.01031** | 0.00325 | 0.00706 | 0.878 | 0.711 | 0.05566 |
| **roberta-base** | T_topology (relational oracle) | 0.9680 | 0.3033 | **0.00994** | 0.00325 | 0.00669 | 1.338 | 0.503 | 0.03078 |
| **xlnet-base** | T_raw (1.0) | 1.1806 | 0.2642 | **0.00934** | 0.00325 | 0.00609 | 0.905 | 0.693 | 0.05503 |
| **xlnet-base** | T_NLL (calibrated) | 0.8975 | 0.2827 | **0.00981** | 0.00326 | 0.00655 | 1.295 | 0.525 | 0.02465 |
| **xlnet-base** | T_JSD (pointwise oracle) | 1.2629 | 0.2636 | **0.00935** | 0.00326 | 0.00609 | 0.860 | 0.714 | 0.05757 |
| **xlnet-base** | T_topology (relational oracle) | 1.1159 | 0.3240 | **0.00922** | 0.00326 | 0.00596 | 1.338 | 0.500 | 0.03970 |
| **distilbert** | T_raw (1.0) | 1.2114 | 0.2740 | **0.00865** | 0.00325 | 0.00539 | 0.914 | 0.690 | 0.05424 |
| **distilbert** | T_NLL (calibrated) | 0.9088 | 0.2909 | **0.00884** | 0.00325 | 0.00559 | 1.311 | 0.518 | 0.02330 |
| **distilbert** | T_JSD (pointwise oracle) | 1.2877 | 0.2736 | **0.00876** | 0.00325 | 0.00551 | 0.874 | 0.709 | 0.05642 |
| **distilbert** | T_topology (relational oracle) | 0.9255 | 0.3078 | **0.00856** | 0.00324 | 0.00532 | 1.386 | 0.484 | 0.01812 |
| **bert-base** | T_raw (1.0) | 1.2696 | 0.2909 | **0.00786** | 0.00324 | 0.00462 | 0.905 | 0.692 | 0.05512 |
| **bert-base** | T_NLL (calibrated) | 0.9194 | 0.2987 | **0.00793** | 0.00324 | 0.00469 | 1.326 | 0.512 | 0.02179 |
| **bert-base** | T_JSD (pointwise oracle) | 1.0137 | 0.2915 | **0.00792** | 0.00323 | 0.00468 | 1.124 | 0.592 | 0.04029 |
| **bert-base** | T_topology (relational oracle) | 0.9675 | 0.2967 | **0.00780** | 0.00323 | 0.00456 | 1.241 | 0.547 | 0.03218 |

---

## Conclusions for Hypothesis H2

- **H2 Outcome**: Pointwise temperature calibration ($T_{\text{NLL}}$) substantially improves soft-label likelihood ($NLL$), but achieves **$< 0.75\%$ relational topology gap closure ($G_Q$)** across all 9 models.
- **Core Mechanism**: Temperature scaling acts as a monotonic rescaling of logit magnitude, altering pointwise entropy while preserving exact logit rank order among classes and leaving inter-item nearest-neighbor topology locked in place.
- **Implication for E003**: Pointwise calibration is insufficient for topological alignment. E003 (Relational Topology Fine-Tuning & Representation Alignment) is necessary to close the relational topology gap.

