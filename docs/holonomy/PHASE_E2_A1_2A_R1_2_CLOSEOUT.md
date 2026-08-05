# Phase E2-A1.2a-R1.2 Closeout Report

**Date**: 2026-08-04  
**Status**: COMPLETED & FROZEN  
**Code Commit**: [`000c6960e4c6458bd10b2df0de830bec01458077`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy)  
**Results Commit**: [`ef0d669ef9697621f55e271dcc4e693c577be120`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy)  
**Branch**: `research/ambiguity-holonomy`  

---

## 1. Summary of Scientific & Methodological Findings

Phase E2-A1.2a-R1.2 successfully established a rigorous, duplicate-free live-model audit protocol for pretrained natural language classifiers (`FacebookAI/roberta-large-mnli`, pinned revision `2a8f12d27941090092df78e4ba6f0928eb5eac98`).

### Verified Results:
1. **Heavy-Tailed Pointwise Renaming Sensitivity**:
   - Evaluated across 300 unique base orbits (1,200 directed transformation edges).
   - Mean displacement $\|\Delta z\|_2 = 0.2006$, Median $= 0.0903$, 95th percentile $= 0.6933$, 99th percentile $= 1.7654$, Maximum $= 5.2320$.
   - **Label Flips**: 6 top-label flips observed across 1,200 directed edges ($0.50\%$).
   - Pointwise JSD: Mean $0.001495$, Max $0.220042$.

2. **Global Commutator & Transport Fit**:
   - $T_a$ held-out $R^2 = 0.9919$, $T_b$ held-out $R^2 = 0.9914$.
   - Relative skill vs Identity: $T_a = -0.10\%$, $T_b = -0.06\%$ (identity $\hat{z} = z_{\text{src}}$ already achieves $R^2 \approx 0.992$).
   - Global canonical commutator $S_H = 4.4 \times 10^{-5}$ (`DESCRIPTIVELY_SMALL`).
   - **SLSQP Constrained Rademacher Wild Bootstrap**: $p = 0.9950$ (`NOT_REJECTED`).
   - **Rename-Context Interaction Test**: Mean interaction norm $0.043343$, permutation $p = 0.1069$ (`NOT_REJECTED`).

---

## 2. Decision & Milestone Freeze

- **NLI Milestone Freeze**: The natural language NLI holonomy phase is officially complete and frozen at R1.2.
- **Transferable Outcome**: The primary transferable finding is that pretrained NLI models exhibit heavy-tailed pointwise sensitivity to proper name substitutions, but learned global linear transport maps do not exhibit noncommutativity beyond identity.
- **Next Phase**: Transition to **Applied Educational Counterfactual AI Audit** (`research/education-counterfactual-audit`).
