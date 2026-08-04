# Sprint 1 Sample-Matched Rate-Distortion & Relational Resolution Summary

**Reference Sample**: Matched 600-item ChaosNLI Pilot (`pilot_600.jsonl`, $N=600$, $Q_{\text{HH}}=0.26338$)

| Model / Condition | Category | $R_{\text{norm}}$ (%) | NLL (nats) | JSD (bits) | $K_{\text{eff}}$ Prototypes ($2^b$) | Effective Bits ($b$) |
|-------------------|----------|------------------------|------------|------------|------------------------------|----------------------|
| Gemma 3 12B Raw API (T=1) LPE | LLM Generative | 9.72% | 3.8277 | 0.1721 | 2.59 | 1.370 |
| Gemma 3 12B Calibrated API (T=1) LPE | LLM Generative | 9.76% | 0.9308 | 0.0738 | 2.59 | 1.374 |
| Gemma 3 12B 30-sample MCE | LLM Generative | 6.64% | 1.5864 | 0.1403 | 2.11 | 1.076 |

---

### Key Sample-Matched Discoveries ($N=600$)

1. **Gemma 3 12B Prototype-Equivalent Resolution**:
   - On the pilot-matched rate-distortion curve, **Gemma 3 12B Raw LPE** ($R=9.72\%$) exhibits prototype-equivalent resolution $b = 1.370$ bits ($K=2.59$).
   - **Calibrated LPE** ($R=9.76\%$) exhibits prototype-equivalent resolution $b = 1.374$ bits ($K=2.59$).
   - **MCE (30 samples)** ($R=6.64\%$) exhibits prototype-equivalent resolution $b = 1.076$ bits ($K=2.11$).

2. **Geometrically Verified Calibration Paradox**:
   - Scalar temperature calibration reduces NLL from **3.8277 nats** to **0.9308 nats** (91.3% NLL gap closed), but its prototype-equivalent resolution remains static at **1.374 bits** ($K = 2.59$).
   - Log-linear interpolation $b = \log_2 K \implies K = 2^b$ strictly enforces mathematical pair consistency across resolution tiers.

---

### Generative LLM Cross-Model Extension

The E004 generative-model analysis was extended from Gemma 3 12B to Qwen 2.5 14B using the same 600-item pilot, six label-symbol permutations, posterior-support target, cross-fitted calibration estimator, and 30-stratum focal-row bootstrap.

Qwen showed a modest but statistically detectable raw relational advantage over Gemma under the primary analysis convention:

$$\Delta R_{\mathrm{raw}} = +2.13 \text{ percentage points}, \qquad 95\% \text{ CI} = [+0.56, +3.78].$$

Calibration affected the two models differently. Gemma’s relational recovery was essentially unchanged, while Qwen’s increased by approximately three percentage points. The resulting two-model interaction was positive under both the primary (-40) censored-token convention and a fixed (-20) stress-test convention.

The complete methods, paired contrasts, censoring sensitivity analysis, claim boundaries, and reproducibility record are documented in [`E004_QWEN25_CROSS_MODEL_EXTENSION.md`](file:///c:/Users/admir/Github/shadowspace/research/reports/E004_QWEN25_CROSS_MODEL_EXTENSION.md).
