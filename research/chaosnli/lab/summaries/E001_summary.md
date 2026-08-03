# E001: Expected Fuzzy Edge-Support Graph Summary (Rigorous Pass)

**Experiment ID**: E001  
**Title**: Expected Fuzzy Edge-Support Graph Construction & Model-Human Relational Comparison  
Dataset Release: `chaosnli-canonical-2026-08-02` (N = 3,113 items: SNLI=1514, MNLI=1599)  
Posterior Draws: B = 500 Dirichlet draws (alpha = [0.5, 0.5, 0.5])  
Monte Carlo Stratified Permutations: B_null = 10,000 per model/metric/scale  
Cross-Fitted Human Baseline (Q_HH): Split-half draw cross-fitting (Half A vs Half B)  

---

## Executive Summary

Experiment **E001** constructs the expected fuzzy edge-support graph S_ij(k) = E[w_ij(k) | human votes] across 500 posterior Dirichlet draws. Model-selected nearest-neighbor mass W_ij^m(k) is evaluated against S_ij(k) to measure average human posterior support Q_support(m, S) = (1 / Nk) * sum_{ij} W_ij^m S_ij.

### Key Findings

1. **Model Relational Mass Exceeds Stratified Null**:
   - Model-selected edge mass selects edges with significantly higher human posterior support than expected under 10,000 stratified item-identity permutations (p_MC = 0.00010; 0/10,000 null exceedances for all 9 models).
   - Top models (BART-Large: 5.11x, RoBERTa-Large: 4.53x, XLNet-Large: 4.08x) select edge mass with average human posterior support up to 5.11x higher than the stratified null baseline (Q_null ≈ 0.00329).

2. **Model Ranking Invariance Across Metric & Scale (Kendall's W = 1.0000)**:
   - The exact model ordering was invariant across all twelve metric/scale configurations in the rigorous run (Kendall's W = 1.0000, mean Kendall tau = 1.0000).
   - **Leading Tier (Ranks 1–3)**: BART-Large (#1), RoBERTa-Large (#2), XLNet-Large (#3).
   - **Mid Tier (Ranks 4–6)**: ALBERT-xxLarge (#4), BERT-Large (#5), RoBERTa-Base (#6).
   - **Base Tier (Ranks 7–9)**: XLNet-Base (#7), DistilBERT (#8), BERT-Base (#9).

3. **Within-Family Model-Scale Ordering**:
   - Larger model variants consistently outperform corresponding base/smaller variants within the same family (RoBERTa-Large > RoBERTa-Base, XLNet-Large > XLNet-Base, BERT-Large > BERT-Base).

4. **Exact Vote-Profile Conditioned Control**:
   - When permuting items ONLY among examples with identical 100-vote human distributions (1,604 profile groups), no model significantly exceeds the conditional exact-profile null (p >= 0.1399 for BART-Large).
   - *Conclusion*: The results are consistent with the observed relational alignment being explained by exact vote-profile structure; no significant residual within-profile identity alignment was detected.

5. **Human High-Support Core Graph**:
   - At k=50, posterior edges with support S_ij >= 0.50 form a graph with **mean directed out-degree 8.29 edges/node** (density = 0.26%).
   - High-support core (S_ij >= 0.80) forms a tight structure with **mean directed out-degree 0.90 edges/node** (density = 0.028%).

---

## Detailed Model Edge Support & Null Statistics (k=10, Hellinger)

| Model | Q_support | Q_null (95% Permutation-Null Interval) | Monte Carlo p | Null Ratio | Human Recovery R_m | Exact-Profile p |
|---|---|---|---|---|---|---|
| **bart-large** | **0.01681** | 0.00329 [0.00307, 0.00353] | 0.00010 (0/10k) | **5.11x** | 19.59% | 0.1758 |
| **roberta-large** | **0.01492** | 0.00329 [0.00306, 0.00352] | 0.00010 (0/10k) | **4.53x** | 16.85% | 0.4915 |
| **xlnet-large** | **0.01334** | 0.00327 [0.00305, 0.00350] | 0.00010 (0/10k) | **4.08x** | 14.60% | 0.7323 |
| **albert-xxlarge** | **0.01153** | 0.00327 [0.00305, 0.00350] | 0.00010 (0/10k) | **3.53x** | 11.98% | 0.4006 |
| **bert-large** | **0.01053** | 0.00328 [0.00305, 0.00351] | 0.00010 (0/10k) | **3.21x** | 10.51% | 0.2997 |
| **roberta-base** | **0.01033** | 0.00326 [0.00303, 0.00349] | 0.00010 (0/10k) | **3.17x** | 10.25% | 0.8861 |
| **xlnet-base** | **0.00934** | 0.00326 [0.00304, 0.00349] | 0.00010 (0/10k) | **2.87x** | 8.82% | 0.4585 |
| **distilbert** | **0.00865** | 0.00325 [0.00303, 0.00348] | 0.00010 (0/10k) | **2.66x** | 7.81% | 0.3457 |
| **bert-base** | **0.00786** | 0.00324 [0.00302, 0.00348] | 0.00010 (0/10k) | **2.42x** | 6.69% | 0.8252 |

*Cross-fitted Human-Human Baseline Q_HH(k=10) = 0.07228.*

---

## Seed Schedule Sensitivity Diagnostic

| Schedule | BART-Large Q | RoBERTa-Large Q | ALBERT-xxLarge Q | Top Model Rank Order | High-Support Corr (S >= 0.50) |
|---|---|---|---|---|---|
| **sequential (42+b)** | 0.01681 | 0.01492 | 0.01153 | bart-large, roberta-large, xlnet-large | 1.000000 |
| **stride (42+1000b)** | 0.01676 | 0.01496 | 0.01153 | bart-large, roberta-large, xlnet-large | 0.941858 |
| **independent_alt (1001+b)** | 0.01679 | 0.01494 | 0.01153 | bart-large, roberta-large, xlnet-large | 0.947774 |

---

## Independent Subdataset Topology Replication (k=10, Hellinger)

### SNLI Independent Subdataset (N = 1,514 items)

| Model | Q_support (SNLI) | Q_null (SNLI) | Monte Carlo p | Human Recovery R_m (SNLI) |
|---|---|---|---|---|
| **bart-large** | **0.03560** | 0.00661 | < 0.001 | 24.40% |
| **roberta-large** | **0.03081** | 0.00664 | < 0.001 | 20.34% |
| **xlnet-large** | **0.02741** | 0.00660 | < 0.001 | 17.51% |
| **albert-xxlarge** | **0.02525** | 0.00662 | < 0.001 | 15.67% |
| **bert-large** | **0.02266** | 0.00660 | < 0.001 | 13.51% |
| **roberta-base** | **0.02187** | 0.00661 | < 0.001 | 12.83% |
| **xlnet-base** | **0.01926** | 0.00660 | < 0.001 | 10.65% |
| **distilbert** | **0.01804** | 0.00661 | < 0.001 | 9.62% |
| **bert-base** | **0.01685** | 0.00660 | < 0.001 | 8.62% |

*Cross-fitted SNLI Human-Human Baseline Q_HH = 0.12546.*

### MNLI Independent Subdataset (N = 1,599 items)

| Model | Q_support (MNLI) | Q_null (MNLI) | Monte Carlo p | Human Recovery R_m (MNLI) |
|---|---|---|---|---|
| **bart-large** | **0.02777** | 0.00626 | < 0.001 | 20.80% |
| **roberta-large** | **0.02543** | 0.00625 | < 0.001 | 18.55% |
| **xlnet-large** | **0.02333** | 0.00626 | < 0.001 | 16.51% |
| **albert-xxlarge** | **0.01903** | 0.00626 | < 0.001 | 12.35% |
| **bert-large** | **0.01793** | 0.00627 | < 0.001 | 11.29% |
| **roberta-base** | **0.01775** | 0.00628 | < 0.001 | 11.10% |
| **xlnet-base** | **0.01705** | 0.00626 | < 0.001 | 10.43% |
| **distilbert** | **0.01570** | 0.00625 | < 0.001 | 9.14% |
| **bert-base** | **0.01398** | 0.00626 | < 0.001 | 7.46% |

*Cross-fitted MNLI Human-Human Baseline Q_HH = 0.10964.*

---

## Structured Binary Artifact Manifests

| Artifact File | Metric | k | Shape | Matrix SHA-256 (f32) | Object IDs SHA-256 |
|---|---|---|---|---|---|
| `S_hellinger_k010.bin` | hellinger | 10 | 3113x3113 | `94e483e714d92f03...` | `121c49cbd40b171d...` |
| `S_hellinger_k020.bin` | hellinger | 20 | 3113x3113 | `c2f6fbb42201e673...` | `121c49cbd40b171d...` |
| `S_hellinger_k050.bin` | hellinger | 50 | 3113x3113 | `2da027e261d9a74a...` | `121c49cbd40b171d...` |
| `S_jensen_shannon_k010.bin` | jensen_shannon | 10 | 3113x3113 | `bdbabf9abaa80130...` | `121c49cbd40b171d...` |
| `S_jensen_shannon_k020.bin` | jensen_shannon | 20 | 3113x3113 | `cdbba38396522c4f...` | `121c49cbd40b171d...` |
| `S_jensen_shannon_k050.bin` | jensen_shannon | 50 | 3113x3113 | `8648c5cc3718c5f5...` | `121c49cbd40b171d...` |
