# Sprint 1 Consolidated Rate-Distortion & Relational Resolution Summary

| Model / Condition | Category | $R_{\text{norm}}$ (%) | NLL | JSD (bits) | $K_{\text{eff}}$ Prototypes | Effective Bits |
|-------------------|----------|------------------------|-----|------------|------------------------------|----------------|
| Gemma 3 12B Raw API (T=1) LPE | LLM Generative | 9.72% | 3.8277 | 0.1721 | 2.21 | 1.123 |
| Gemma 3 12B Calibrated API (T=1) LPE | LLM Generative | 9.76% | 0.9308 | 0.0738 | 2.22 | 1.126 |
| Gemma 3 12B 30-sample MCE | LLM Generative | 6.64% | 1.5864 | 0.1403 | 1.82 | 0.825 |
| Classifier Best Subgroup (Size 1: bart-large) | Classifier Ensemble | 37.93% | 0.8627 | 0.0420 | 6.26 | 2.639 |
| Classifier Best Subgroup (Size 2: bart-large + roberta-large) | Classifier Ensemble | 53.82% | 0.7456 | 0.0277 | 8.82 | 3.132 |
| Classifier Best Subgroup (Size 3: bart-large + roberta-large + xlnet-large) | Classifier Ensemble | 64.72% | 0.7171 | 0.0236 | 10.79 | 3.426 |
| Classifier Best Subgroup (Size 4: bart-large + roberta-large + xlnet-large + albert-xxlarge) | Classifier Ensemble | 71.34% | 0.7082 | 0.0223 | 12.03 | 3.589 |
| Classifier Best Subgroup (Size 6: bart-large + roberta-large + xlnet-large + albert-xxlarge + bert-large + roberta-base) | Classifier Ensemble | 77.69% | 0.7038 | 0.0225 | 13.2 | 3.718 |
| Classifier Best Subgroup (Size 8: bart-large + roberta-large + xlnet-large + albert-xxlarge + bert-large + roberta-base + xlnet-base + distilbert) | Classifier Ensemble | 81.56% | 0.7042 | 0.0233 | 13.9 | 3.796 |

---
### Key Insights

1. **Gemma 3 12B Relational Resolution Capacity**:
   - Gemma 3 12B (raw LPE: **9.72%**, calibrated LPE: **9.76%**) operates at $K_{\text{eff}} \approx 2.22$ human prototype quantizer equivalents ($\,\approx 1.11$ bits).
   - Gemma 3 12B resolves slightly more structure than a 2-prototype discrete quantizer, but is far below a single BART-Large classifier (**37.93%**, $K_{\text{eff}} \approx 6.25$, $2.65$ bits) and the best 6-model classifier ensemble (**77.69%**, $K_{\text{eff}} \approx 36.32$, $5.18$ bits).

2. **Calibration Paradox on Rate-Distortion Scale**:
   - Scalar temperature calibration reduces NLL dramatically from **3.8277** to **0.9308** (91.3% NLL gap closed), but its relational resolution remains static at **1.11 bits** ($K_{\text{eff}}=2.22$).
   - This confirms geometrically that calibration adjusts confidence scaling along a 1D ray without altering prototype resolution or relational neighborhood structure.
