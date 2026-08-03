# E001: Posterior Edge-Support Graph Summary

**Experiment ID**: E001  
**Title**: Posterior Edge-Support Graph Construction & Model Comparison  
**Dataset Release**: `chaosnli-canonical-2026-08-02` ($N = 3,113$ items: SNLI=1514, MNLI=1599)  
**Posterior Draws**: $B = 500$ Dirichlet draws ($\alpha = [0.5, 0.5, 0.5]$)  

---

## Executive Summary

Experiment **E001** constructs the posterior edge-support graph $S_{ij}(k) = \mathbb{P}(j \in \mathcal{N}_i(k) \mid \text{human votes})$ across 500 posterior-predictive draws. This evaluates how reliably item pairs stay connected in nearest-neighbor graphs under human posterior variance and tests whether model prediction spaces capture these high-support relational bonds.

### Key Findings

1. **Model Relational Signal Exceeds Stratified Null**:
   - All 9 NLI models significantly exceed the stratified-null permutation baseline ($Q_{null} \approx 0.0033$ at $k=10$, $p < 0.001$).
   - Large models (ALBERT-xxLarge, RoBERTa-Large, BART-Large) recover nearly **$3.3\times$** the edge support of random chance.

2. **Model Ranking is Exceptionally Stable Across Geometries**:
   - The model ranking under Posterior Edge Support is **100% consistent across all 3 probability metrics** (Hellinger, Jensen-Shannon, Total Variation) and **all 4 neighborhood scales** ($k \in \{5, 10, 20, 50\}$):
     1. **ALBERT-xxLarge** ($Q_{edge} = 0.01083$)
     2. **RoBERTa-Large** ($Q_{edge} = 0.01063$)
     3. **BART-Large** ($Q_{edge} = 0.01052$)
     4. **XLNet-Large** ($Q_{edge} = 0.01028$)
     5. **RoBERTa-Base** ($Q_{edge} = 0.00957$)
     6. **XLNet-Base** ($Q_{edge} = 0.00934$)
     7. **BERT-Large** ($Q_{edge} = 0.00915$)
     8. **BERT-Base** ($Q_{edge} = 0.00875$)
     9. **DistilBERT** ($Q_{edge} = 0.00791$)

3. **High-Support Human Core Graph**:
   - At $k=50$, posterior edges with support $\ge 0.50$ form a graph with **mean degree 8.29 edges/node** (density = $0.26\%$).
   - Edges with support $\ge 0.80$ form a tight core of **mean degree 0.90 edges/node** (density = $0.028\%$).

---

## Detailed Model Edge Support Overlap ($k=10$, Hellinger)

| Model | $Q_{edge}(m, S)$ | $\Delta_{edge}(m)$ | $Q_{null}$ | Null Ratio ($Q / Q_{null}$) |
|---|---|---|---|---|
| **ALBERT-xxLarge** | **0.01083** | 0.98917 | 0.00327 | **3.31x** |
| **RoBERTa-Large** | **0.01063** | 0.98937 | 0.00325 | **3.27x** |
| **BART-Large** | **0.01052** | 0.98948 | 0.00325 | **3.24x** |
| **XLNet-Large** | **0.01028** | 0.98972 | 0.00325 | **3.16x** |
| **RoBERTa-Base** | **0.00957** | 0.99043 | 0.00325 | **2.94x** |
| **XLNet-Base** | **0.00934** | 0.99066 | 0.00327 | **2.86x** |
| **BERT-Large** | **0.00915** | 0.99085 | 0.00325 | **2.81x** |
| **BERT-Base** | **0.00875** | 0.99125 | 0.00331 | **2.64x** |
| **DistilBERT** | **0.00791** | 0.99209 | 0.00331 | **2.39x** |
