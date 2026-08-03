# E002: Pointwise Soft-Label Calibration vs. Relational Topology (Coherent Cross-Fitted Pass)

**Experiment ID**: E002  
**Title**: Pointwise Soft-Label Calibration vs. Relational Neighborhood Recovery  
**Status**: `pilot_requires_graph_crossfit_rerun`  
Dataset Release: `chaosnli-canonical-2026-08-02` (N = 3,113 items)  
Cross-Validation: 5-Fold Stratified Coherent Cross-Fitting by (Dataset, Majority Label, Empirical Entropy Quintile)  
Bound E001 Artifact (k=10): `E001-hellinger-k010-expected-fuzzy-support-v1` (SHA-256: `94e483e714d92f03...`)  
Bound E001 Artifact (k=50): `S_hellinger_k050.bin` (SHA-256: `2da027e261d9a74a...`)  
Human Soft-Label Entropy Floor ($H(p)$): 0.65062 nats  
Human Relational Reference ($Q_{HH}$): 0.07228  

---

## Executive Summary

Experiment **E002** evaluates **Hypothesis H2**: *Does temperature scaling improve marginal probability alignment (NLL) without proportionately recovering relational human belief-space topology ($Q_{\text{support}}$)?*

### Key Methodological Fixes Applied

1. **Coherent Single-Temperature Full-Dataset Graphs**: For each fold, $T_f$ is fitted on training items, then applied across ALL $N=3,113$ items to construct a single coherent graph $W^{f, T_f}$, scoring held-out rows $i \in H_f$.
2. **Strict Training-Only Topology Target**: Training topology search optimizes $Q_{\text{excess, train}}(T) = Q_{\text{support, train}}(T) - Q_{\text{null, train}}(T)$ against posterior support matrices constructed over training items ONLY ($N_{\text{train}} \approx 2,490$).
3. **Independent $k=50$ Core Target**: Core mass and recall metrics are evaluated against the true $k=50$ expected support matrix (`S_hellinger_k050.bin`).
4. **NLL Gap Closure ($G_{\text{NLL}}$)**: Defined relative to the empirical human soft-label entropy floor $H(p) = 0.65062$ nats: $G_{\text{NLL}} = \frac{\text{NLL}_{\text{raw}} - \text{NLL}_{\text{cal}}}{\text{NLL}_{\text{raw}} - H(p)}$.

### Key Scientific Findings

1. **$H2a_{\text{NLL}}$ Supported ($G_{\text{NLL}} \approx 24.9\% - 56.6\%$)**:
   - Soft-label cross-entropy NLL improves consistently under $T_{\text{NLL}} \approx 1.86 - 3.93$ across all 9 models.
2. **$H2a_{\text{JS}}$ Contradicted (JS Divergence Increases)**:
   - Temperature calibration ($T_{\text{NLL}}$) softens probabilities, increasing prediction entropy above 1.2 bits and increasing symmetric JS divergence relative to human targets across all 9 models.
3. **$H2b_{\text{NLL}}$ Confirmed ($G_{\text{NLL}} \gg G_Q < 0.70\%$)**:
   - While pointwise likelihood gap closure $G_{\text{NLL}}$ reaches **24.9% to 56.6%**, relational topology gap closure $G_Q$ is **$< 0.70\%$** across all models ($0.16\% - 0.70\%$).
4. **$Q_{\text{profile-excess}}(T)$ Remains Zero at All Temperatures**:
   - $Q_{\text{profile-excess}}(T) = Q_{\text{support}}(T) - Q_{\text{exact-profile-null}}(T)$ remains $\approx 0.0000$ across all candidate temperatures $T \in [0.10, 10.00]$.

---

## Detailed 5-Fold Coherent Cross-Fitted Model Calibration Results

| Model | $T_{\text{NLL}}$ | $T_{\text{JSD}}$ | $T_{\text{topology}}$ | NLL ($T_{\text{raw}}$) | NLL ($T_{\text{cal}}$) | $G_{\text{NLL}}$ | JSD ($T_{\text{raw}}$) | JSD ($T_{\text{cal}}$) | $Q_{\text{raw}}$ | $Q_{\text{cal}}$ | Relational Gap Closure $G_Q$ | $Q_{\text{profile-excess}}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **bart-large** | 1.86 | 0.83 | 5.20 | 0.8627 | **0.8099** | **24.91%** | 0.0420 | 0.0578 | **0.01681** | **0.01714** | **0.59%** | `0.000034` |
| **roberta-large** | 2.16 | 0.85 | 4.70 | 0.9082 | **0.8258** | **31.98%** | 0.0489 | 0.0659 | **0.01492** | **0.01521** | **0.51%** | `-0.000002` |
| **xlnet-large** | 2.35 | 0.83 | 4.42 | 0.9474 | **0.8451** | **34.48%** | 0.0567 | 0.0748 | **0.01334** | **0.01372** | **0.65%** | `-0.000018` |
| **albert-xxlarge** | 2.59 | 0.86 | 6.85 | 0.9892 | **0.8577** | **38.83%** | 0.0636 | 0.0804 | **0.01153** | **0.01178** | **0.41%** | `0.000002` |
| **bert-large** | 3.00 | 0.86 | 4.90 | 1.0658 | **0.8773** | **45.41%** | 0.0739 | 0.0892 | **0.01053** | **0.01074** | **0.33%** | `0.000032` |
| **roberta-base** | 3.12 | 0.90 | 3.65 | 1.0938 | **0.8848** | **47.15%** | 0.0808 | 0.0929 | **0.01033** | **0.01043** | **0.16%** | `-0.000052` |
| **xlnet-base** | 3.53 | 0.94 | 7.04 | 1.1806 | **0.8975** | **53.43%** | 0.0891 | 0.0980 | **0.00934** | **0.00978** | **0.70%** | `-0.000005` |
| **distilbert** | 3.67 | 0.97 | 4.20 | 1.2114 | **0.9087** | **53.98%** | 0.0954 | 0.1025 | **0.00865** | **0.00886** | **0.34%** | `-0.000034` |
| **bert-base** | 3.93 | 2.72 | 3.72 | 1.2696 | **0.9193** | **56.59%** | 0.1062 | 0.1075 | **0.00786** | **0.00809** | **0.36%** | `-0.000036` |

---

## Condition Comparison & Structural Graph Turnover (k=10 Hellinger & k=50 Core)

| Model | Condition | NLL (nats) | JSD (bits) | $Q_{\text{support}}$ | $Q_{\text{null}}$ | $Q_{\text{global-excess}}$ | Graph Turnover | Core Mass ($k=50$) | Core Recall ($k=50$) | Avg Entropy | Distance Var |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **bart-large** | T_raw (1.0) | 0.8627 | 0.0420 | **0.01681** | 0.00329 | 0.01352 | 0.00% | 0.022923 | 13.78% | 0.960 | 0.04873 |
| **bart-large** | T_NLL (calibrated) | 0.8099 | 0.0578 | **0.01714** | 0.00330 | 0.01384 | 13.38% | 0.021927 | 13.18% | 1.168 | 0.03617 |
| **bart-large** | T_JSD (pointwise oracle) | 0.9088 | 0.0406 | **0.01670** | 0.00329 | 0.01341 | 3.58% | 0.023283 | 14.00% | 0.881 | 0.05359 |
| **bart-large** | T_topology (relational oracle) | 0.8854 | 0.0984 | **0.01736** | 0.00330 | 0.01406 | 27.83% | 0.021394 | 12.86% | 1.446 | 0.01219 |
| **roberta-large** | T_raw (1.0) | 0.9082 | 0.0489 | **0.01492** | 0.00329 | 0.01163 | 0.00% | 0.018439 | 11.09% | 0.945 | 0.05078 |
| **roberta-large** | T_NLL (calibrated) | 0.8258 | 0.0659 | **0.01521** | 0.00329 | 0.01192 | 18.57% | 0.017636 | 10.60% | 1.191 | 0.03478 |
| **roberta-large** | T_JSD (pointwise oracle) | 0.9583 | 0.0479 | **0.01498** | 0.00329 | 0.01169 | 3.28% | 0.018728 | 11.26% | 0.879 | 0.05483 |
| **roberta-large** | T_topology (relational oracle) | 0.8742 | 0.0935 | **0.01556** | 0.00329 | 0.01228 | 30.08% | 0.017205 | 10.34% | 1.414 | 0.01500 |
| **xlnet-large** | T_raw (1.0) | 0.9474 | 0.0567 | **0.01334** | 0.00327 | 0.01007 | 0.00% | 0.016621 | 9.99% | 0.953 | 0.05009 |
| **xlnet-large** | T_NLL (calibrated) | 0.8451 | 0.0748 | **0.01372** | 0.00328 | 0.01044 | 19.90% | 0.015901 | 9.56% | 1.219 | 0.03282 |
| **xlnet-large** | T_JSD (pointwise oracle) | 1.0092 | 0.0556 | **0.01332** | 0.00327 | 0.01005 | 3.46% | 0.016711 | 10.05% | 0.882 | 0.05426 |
| **xlnet-large** | T_topology (relational oracle) | 0.8782 | 0.0951 | **0.01384** | 0.00328 | 0.01056 | 29.16% | 0.015695 | 9.44% | 1.400 | 0.01645 |
| **albert-xxlarge** | T_raw (1.0) | 0.9892 | 0.0636 | **0.01153** | 0.00327 | 0.00826 | 0.00% | 0.014243 | 8.56% | 0.947 | 0.05115 |
| **albert-xxlarge** | T_NLL (calibrated) | 0.8577 | 0.0804 | **0.01178** | 0.00327 | 0.00851 | 22.51% | 0.013942 | 8.38% | 1.237 | 0.03117 |
| **albert-xxlarge** | T_JSD (pointwise oracle) | 1.0469 | 0.0629 | **0.01157** | 0.00326 | 0.00830 | 2.87% | 0.014539 | 8.74% | 0.890 | 0.05439 |
| **albert-xxlarge** | T_topology (relational oracle) | 0.9251 | 0.1146 | **0.01211** | 0.00328 | 0.00884 | 32.83% | 0.013832 | 8.32% | 1.486 | 0.00862 |
| **bert-large** | T_raw (1.0) | 1.0658 | 0.0739 | **0.01053** | 0.00328 | 0.00726 | 0.00% | 0.013183 | 7.93% | 0.935 | 0.05262 |
| **bert-large** | T_NLL (calibrated) | 0.8773 | 0.0892 | **0.01074** | 0.00328 | 0.00746 | 25.92% | 0.012862 | 7.73% | 1.266 | 0.02826 |
| **bert-large** | T_JSD (pointwise oracle) | 1.1342 | 0.0733 | **0.01053** | 0.00328 | 0.00725 | 3.23% | 0.013164 | 7.92% | 0.884 | 0.05549 |
| **bert-large** | T_topology (relational oracle) | 0.8990 | 0.1031 | **0.01085** | 0.00328 | 0.00757 | 31.79% | 0.012843 | 7.72% | 1.410 | 0.01512 |
| **roberta-base** | T_raw (1.0) | 1.0938 | 0.0808 | **0.01033** | 0.00326 | 0.00707 | 0.00% | 0.013241 | 7.96% | 0.931 | 0.05266 |
| **roberta-base** | T_NLL (calibrated) | 0.8848 | 0.0929 | **0.01043** | 0.00326 | 0.00717 | 26.32% | 0.012959 | 7.79% | 1.277 | 0.02697 |
| **roberta-base** | T_JSD (pointwise oracle) | 1.1426 | 0.0805 | **0.01028** | 0.00325 | 0.00703 | 2.17% | 0.013267 | 7.98% | 0.897 | 0.05459 |
| **roberta-base** | T_topology (relational oracle) | 0.8874 | 0.0963 | **0.01048** | 0.00326 | 0.00721 | 28.35% | 0.012933 | 7.78% | 1.326 | 0.02248 |
| **xlnet-base** | T_raw (1.0) | 1.1806 | 0.0891 | **0.00934** | 0.00326 | 0.00609 | 0.00% | 0.011076 | 6.66% | 0.905 | 0.05503 |
| **xlnet-base** | T_NLL (calibrated) | 0.8975 | 0.0980 | **0.00978** | 0.00326 | 0.00652 | 29.01% | 0.011009 | 6.62% | 1.295 | 0.02466 |
| **xlnet-base** | T_JSD (pointwise oracle) | 1.2191 | 0.0890 | **0.00936** | 0.00326 | 0.00611 | 1.53% | 0.011057 | 6.65% | 0.884 | 0.05624 |
| **xlnet-base** | T_topology (relational oracle) | 0.9359 | 0.1187 | **0.00981** | 0.00326 | 0.00654 | 34.76% | 0.011089 | 6.67% | 1.477 | 0.00881 |
| **distilbert** | T_raw (1.0) | 1.2114 | 0.0954 | **0.00865** | 0.00325 | 0.00539 | 0.00% | 0.010337 | 6.22% | 0.914 | 0.05424 |
| **distilbert** | T_NLL (calibrated) | 0.9087 | 0.1025 | **0.00886** | 0.00325 | 0.00561 | 30.14% | 0.010170 | 6.12% | 1.311 | 0.02329 |
| **distilbert** | T_JSD (pointwise oracle) | 1.2294 | 0.0954 | **0.00865** | 0.00325 | 0.00540 | 0.70% | 0.010292 | 6.19% | 0.904 | 0.05477 |
| **distilbert** | T_topology (relational oracle) | 0.9106 | 0.1052 | **0.00887** | 0.00325 | 0.00562 | 31.60% | 0.010177 | 6.12% | 1.353 | 0.01955 |
| **bert-base** | T_raw (1.0) | 1.2696 | 0.1062 | **0.00786** | 0.00324 | 0.00462 | 0.01% | 0.008448 | 5.08% | 0.905 | 0.05512 |
| **bert-base** | T_NLL (calibrated) | 0.9193 | 0.1075 | **0.00809** | 0.00324 | 0.00485 | 31.06% | 0.008615 | 5.18% | 1.326 | 0.02178 |
| **bert-base** | T_JSD (pointwise oracle) | 0.9371 | 0.1048 | **0.00795** | 0.00325 | 0.00470 | 25.99% | 0.008461 | 5.09% | 1.202 | 0.03299 |
| **bert-base** | T_topology (relational oracle) | 0.9197 | 0.1067 | **0.00811** | 0.00324 | 0.00487 | 30.45% | 0.008603 | 5.17% | 1.308 | 0.02340 |

---

## Conclusions for Hypothesis H2

- **H2a (NLL Reduction)**: **SUPPORTED**. Pointwise temperature scaling ($T_{\text{NLL}}$) consistently reduces soft-label cross-entropy ($G_{\text{NLL}} = 24.9\% - 56.6\%$).
- **H2a (JSD Alignment)**: **CONTRADICTED**. Temperature scaling increases prediction entropy, worsening symmetric JS divergence.
- **H2b (Relational Disconnect)**: **CONFIRMED**. Pointwise likelihood gap closure $G_{\text{NLL}}$ dramatically exceeds relational topology gap closure ($G_Q < 0.70\%$). Scalar logit scaling cannot recover relational belief-space topology.
- **Implication for E003**: Relational topology alignment requires representation learning / topology fine-tuning (E003), as post-hoc scalar transformations leave neighborhood graphs locked.

