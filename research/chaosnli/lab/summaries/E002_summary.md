# E002: Model Logit Temperature Calibration Summary

**Experiment ID**: E002  
**Title**: Model Logit Temperature Calibration against Posterior Edge-Support Topology  
**Dataset Release**: `chaosnli-canonical-2026-08-02` ($N = 3,113$ items)  
**Temperatures Evaluated**: $T \in \{0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0\}$  
**Posterior Draws**: 500 draws across 2 probability geometries (Hellinger, Jensen-Shannon) and 4 scales ($k \in \{5, 10, 20, 50\}$)  

---

## Executive Summary

Experiment **E002** investigates whether uncalibrated model logit overconfidence (low entropy) accounts for the low model-human relational alignment ($Q_{edge} \approx 0.01$) observed in E001. By sweeping temperature $T \in [0.1, 5.0]$, E002 tests if logit temperature scaling can bridge the topological gap to human posterior edge support.

### Major Finding: Topological Mismatch is Invariant to Temperature Scaling

1. **Marginal Improvement (< 3.5%)**:
   - Optimal temperature $T^*_m$ yields only tiny relative gains in $Q_{edge}$ (e.g., $+3.38\%$ for RoBERTa-Base at $T^*=0.5$, $+0.47\%$ for BART-Large at $T^*=1.2$).
   - No model achieved the pre-registered Go Criteria threshold of a $\ge 15\%$ improvement.

2. **Severe Degradation at Low Temperatures**:
   - Over-sharpening model logits ($T = 0.1$) collapses relational alignment down to $Q_{edge} \approx 0.0033$, matching the random stratified-null baseline.

3. **Scientific Conclusion**:
   - **Logit overconfidence is NOT the primary cause of model-human relational mismatch.**
   - Simple post-hoc temperature calibration cannot repair nearest-neighbor manifold distortion. This strongly motivates **Program B: Manifold-Preserving Alignment**, which introduces explicit soft-label loss constraints during training.

---

## Temperature Scaling Performance ($k=10$, Hellinger)

| Model | Base $Q_{edge} (T=1.0)$ | Optimal $T^*$ | Optimal $Q_{edge}(T^*)$ | Relative Gain | Status vs Go Criteria |
|---|---|---|---|---|---|
| **ALBERT-xxLarge** | 0.01083 | 1.2 | 0.01088 | +0.46% | Missed (< 15%) |
| **RoBERTa-Large** | 0.01063 | 1.0 | 0.01063 | +0.00% | Missed (< 15%) |
| **BART-Large** | 0.01052 | 1.2 | 0.01058 | +0.57% | Missed (< 15%) |
| **XLNet-Large** | 0.01028 | 1.0 | 0.01028 | +0.00% | Missed (< 15%) |
| **RoBERTa-Base** | 0.00957 | 0.5 | 0.00989 | +3.34% | Missed (< 15%) |
| **XLNet-Base** | 0.00934 | 1.0 | 0.00934 | +0.00% | Missed (< 15%) |
| **BERT-Large** | 0.00915 | 1.0 | 0.00915 | +0.00% | Missed (< 15%) |
| **BERT-Base** | 0.00875 | 0.8 | 0.00882 | +0.80% | Missed (< 15%) |
| **DistilBERT** | 0.00791 | 0.8 | 0.00798 | +0.88% | Missed (< 15%) |
