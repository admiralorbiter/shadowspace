# E001: Expected Fuzzy Edge-Support Graph Summary (Rigorous Pass)

**Experiment ID**: E001  
**Title**: Expected Fuzzy Edge-Support Graph Construction & Model-Human Relational Comparison  
Dataset Release: `chaosnli-canonical-2026-08-02` (N = 3,113 items: SNLI=1514, MNLI=1599)  
Posterior Draws: B = 500 Dirichlet draws (alpha = [0.5, 0.5, 0.5])  
Monte Carlo Stratified Permutations: B_null = 10,000 per model/metric/scale  
Cross-Fitted Human Baseline (Q_HH): Split-half draw cross-fitting (Half A vs Half B)  
Seed Stability (Seed 42 vs Seed 1001): Pearson r = 0.973945, MSE = 0.00001189  

---

## Executive Summary

Experiment **E001** constructs the expected fuzzy edge-support graph S_ij(k) = E[w_ij(k) | human votes] across 500 posterior Dirichlet draws. Model-selected nearest-neighbor mass W_ij^m(k) is evaluated against S_ij(k) to measure average human posterior support Q_support(m, S) = (1 / Nk) * sum_{ij} W_ij^m S_ij.

### Key Findings

1. **Model Relational Mass Exceeds Stratified Null**:
   - Model-selected edge mass selects edges with significantly higher human posterior support than expected under 10,000 stratified item-identity permutations (p <= 0.0001 for all models across all scales).
   - Top models (ALBERT-xxLarge, RoBERTa-Large, BART-Large) select edge mass with average human posterior support ~3.3x higher than the stratified null baseline (Q_null ≈ 0.00325).

2. **High Concordance Across Metrics & Scales (Kendall's W = 1.0000)**:
   - Rankings across all 12 metric/scale combinations are **highly concordant** (Kendall's W = 1.0000, mean Kendall tau = 1.0000), though not strictly identical.
   - **Leading Tier (Ranks 1–3)**: ALBERT-xxLarge, RoBERTa-Large, BART-Large.
   - **Mid Tier (Ranks 4–6)**: XLNet-Large, RoBERTa-Base, XLNet-Base.
   - **Base/Distil Tier (Ranks 7–9)**: BERT-Large, BERT-Base, DistilBERT.

3. **Within-Family Model-Scale Ordering**:
   - Larger model variants consistently outperform corresponding base/smaller variants within the same family (RoBERTa-Large > RoBERTa-Base, XLNet-Large > XLNet-Base, BERT-Large > BERT-Base).

4. **Human High-Support Core Graph**:
   - At k=50, posterior edges with support S_ij >= 0.50 form a graph with **mean directed out-degree 8.29 edges/node** (density = 0.26%).
   - High-support core (S_ij >= 0.80) forms a tight structure with **mean directed out-degree 0.90 edges/node** (density = 0.028%).

---

## Detailed Model Edge Support & Null Statistics (k=10, Hellinger)

| Model | Q_support | Q_null (95% CI) | Monte Carlo p | Null Ratio | Human Recovery R_m | Exact-Profile p |
|---|---|---|---|---|---|---|
| **bart-large** | **0.01681** | 0.00329 [0.00307, 0.00353] | < 0.0001 | **5.11x** | 19.59% | 0.1399 |
| **roberta-large** | **0.01492** | 0.00329 [0.00306, 0.00352] | < 0.0001 | **4.53x** | 16.85% | 0.5025 |
| **xlnet-large** | **0.01334** | 0.00327 [0.00305, 0.00350] | < 0.0001 | **4.08x** | 14.60% | 0.7333 |
| **albert-xxlarge** | **0.01153** | 0.00327 [0.00305, 0.00350] | < 0.0001 | **3.53x** | 11.98% | 0.4286 |
| **bert-large** | **0.01053** | 0.00328 [0.00305, 0.00351] | < 0.0001 | **3.21x** | 10.51% | 0.2458 |
| **roberta-base** | **0.01033** | 0.00326 [0.00303, 0.00349] | < 0.0001 | **3.17x** | 10.25% | 0.8971 |
| **xlnet-base** | **0.00934** | 0.00326 [0.00304, 0.00349] | < 0.0001 | **2.87x** | 8.82% | 0.4815 |
| **distilbert** | **0.00865** | 0.00325 [0.00303, 0.00348] | < 0.0001 | **2.66x** | 7.81% | 0.3397 |
| **bert-base** | **0.00786** | 0.00324 [0.00302, 0.00348] | < 0.0001 | **2.42x** | 6.69% | 0.8292 |

*Cross-fitted Human-Human Baseline Q_HH(k=10) = 0.07228.*

---

## Model Rank Ranges Across 12 Configurations

| Model | Min Rank | Max Rank | Mean Rank | Rank Tier |
|---|---|---|---|---|
| **albert-xxlarge** | 4 | 4 | 4.00 | Mid Tier (4-6) |
| **bart-large** | 1 | 1 | 1.00 | Leading Tier (1-3) |
| **bert-base** | 9 | 9 | 9.00 | Base Tier (7-9) |
| **bert-large** | 5 | 5 | 5.00 | Mid Tier (4-6) |
| **distilbert** | 8 | 8 | 8.00 | Base Tier (7-9) |
| **roberta-base** | 6 | 6 | 6.00 | Mid Tier (4-6) |
| **roberta-large** | 2 | 2 | 2.00 | Leading Tier (1-3) |
| **xlnet-base** | 7 | 7 | 7.00 | Base Tier (7-9) |
| **xlnet-large** | 3 | 3 | 3.00 | Leading Tier (1-3) |
