# E004 Stage 1B Experiment Report: Relational Alignment of Generative LLM Judgment Distributions

**Experiment**: E004 — Relational Alignment of Generative LLM Judgment Distributions  
**Model**: Gemma 3 12B through Ollama  
**Dataset**: ChaosNLI pilot ($N=600$ items)  
**Primary Prompt Version**: v2  
**Primary Symbol Set**: A / B / C across all six semantic label permutations  
**Primary Result Commit**: `b8fd158a0d25ae1d150def19d9ec307c06bda94d` / `d599ec6` / `a7a4f49`  
**Git Release Tag**: `e004-stage1b-primary-frozen-v1`  
**Status**: Primary scientific analysis complete; repository freeze metadata synchronized & tagged.

---

## Executive Summary

This experiment tested whether an instruction-tuned generative language model recovers the relational geometry of human disagreement, rather than only predicting the majority label or matching each item’s vote distribution independently.

Gemma 3 12B was evaluated using two probability-elicitation methods:
1. **Log-probability estimation (LPE)**: candidate-token probabilities for entailment, neutral, and contradiction were reconstructed under all six label-symbol permutations.
2. **Monte Carlo estimation (MCE)**: the model generated 30 sampled decisions per item—five samples under each of the six label mappings.

The central result is a **sharp separation between pointwise calibration and relational alignment**.

Cross-fitted scalar score rescaling reduced API ($T=1$) LPE negative log-likelihood from $3.8277$ to $0.9308$, closing approximately **$91.3\%$** of the gap to the empirical human target entropy. Brier score and Jensen–Shannon divergence also improved substantially. However, normalized relational recovery remained effectively unchanged:
$$R_{\text{raw LPE}} = 9.72\%,\qquad R_{\text{calibrated LPE}} = 9.76\%.$$

A coherent 30-stratum item bootstrap found no statistically supported relational improvement:
$$\Delta R_{\text{calibrated}-\text{raw}}\approx +0.03\text{ percentage points},\qquad 95\%\ \mathrm{CI}=[-0.94, +1.03].$$

MCE achieved similar majority-label accuracy ($65.33\%$) but worse distributional ($\text{NLL} = 1.5864$) and relational ($R = 6.64\%$) estimates. The relational difference from calibrated LPE was statistically negative ($95\%\ \text{CI}: [-4.27\%, -2.07\%]$).

A matched finite-sample simulation based on the API ($T=1$) LPE distributions placed actual MCE at the 80.1st percentile of the simulated distribution, with a two-sided Monte Carlo tail probability of $p = 0.4110$. Thus, the observed MCE relational score is compatible with the degradation expected from estimating a distribution using only 30 samples.

**The main scientific conclusion is:**
> **Calibration repairs probability scale, not disagreement geometry.**

---

## 1. Research Question

The experiment asked:
> *Do modern instruction-tuned generative LLMs exhibit non-random relational alignment with human judgment-distribution geometry, how do log-probability and repeated-sampling estimates compare, and does pointwise calibration improve relational recovery?*

This question differs from ordinary classification evaluation. Traditional evaluation asks whether a model:
- Predicts the majority label;
- Assigns high probability to human labels;
- Has low NLL, Brier score, or Jensen–Shannon divergence.

E004 additionally asks whether the model preserves the item-to-item organization of human ambiguity:
- Which items are close in human judgment space?
- Which ambiguity profiles form neighborhoods?
- Which human-supported edges are recovered?
- Does calibration improve those relationships, or merely soften the same underlying ordering?

---

## 2. Experimental Design

### 2.1 Dataset
The experiment used a balanced 600-item ChaosNLI pilot sampled across:
- SNLI and MNLI datasets;
- Human majority label;
- Human entropy quintile.

The analysis used explicit 30-stratum folds ($30\text{ strata}$). Five-fold cross-validation was used for scalar calibration.

### 2.2 Human Target
Each item contains 100 human annotations over entailment, neutral, and contradiction. Human vote distributions were converted into a posterior expected fuzzy $k$-nearest-neighbor support matrix under Hellinger geometry.
- Primary neighborhood scale: $k=10$.
- Split-half human reference: $Q_{\mathrm{HH}}=0.26338$.
- Empirical target entropy: $H_{\mathrm{target}}=0.6543\text{ nats}$.

### 2.3 Prompt and Label-Order Controls
Every item was evaluated under all six permutations of the semantic labels across the symbols A, B, and C. This design reduces dependence on arbitrary label position or token identity.

Mean permutation-level variability under API ($T=1$) LPE was $\overline d_{\mathrm{perm}}^{H}=0.0753$ in Hellinger distance, showing that label ordering was non-negligible and permutation averaging was warranted.

### 2.4 LPE Conditions
Two direct LPE conditions were collected:
1. API request temperature $T=0.0$, retained as a diagnostic elicitation ($3,600$ requests).
2. API request temperature $T=1.0$, treated as the primary uncalibrated LPE condition ($3,600$ requests).

Candidate-token mass for A, B, and C was effectively 1.0, ruling out material probability leakage to other first-token choices.

### 2.5 Cross-Fitted Scalar Calibration
For each fold $f$:
1. Fit scalar temperature $T_f^*$ on the other four folds.
2. Apply $T_f^*$ to every item to construct one complete fold-specific probability matrix.
3. Build the complete $600\times600$ model graph.
4. Score only focal rows in held-out fold $f$.
5. Aggregate held-out focal-row contributions across folds.

Fitted fold temperatures were approximately $[10.55, 10.08, 10.31, 10.35, 10.51]$, with mean $\overline T^*=10.36$. This is interpreted as post-hoc calibration of API-returned candidate scores, not recovery of undocumented intrinsic model logits.

### 2.6 MCE Condition
MCE sampled five decisions under each of the six label mappings ($6\times5=30\text{ samples per item}$). Total requests: $600\times30=18{,}000$.

The canonical file contains exactly 18,000 unique, valid records. Item distributions were estimated with Jeffreys smoothing:
$$\hat{p}_{ic} = \frac{n_{ic}+0.5}{30+1.5}.$$

---

## 3. Metrics

### 3.1 Pointwise Metrics
- Majority-label accuracy
- Soft-label negative log-likelihood (NLL)
- Brier score
- Jensen–Shannon divergence (JSD)

### 3.2 Relational Metrics
For a model graph $W$ and human posterior support matrix $S$:
$$Q_{\text{support}} = \frac{1}{Nk}\sum_{i,j}W_{ij}S_{ij}.$$

The exact dataset-stratified block-density null ($Q_{\text{null}}^{\text{block}}$) preserves SNLI/MNLI graph mass while randomizing identity within dataset blocks ($Q_{\text{null}}^{\text{block}} \approx 0.01683$).

Normalized relational recovery is:
$$R_{\mathrm{norm}} = \frac{Q_{\mathrm{support}}-Q_{\mathrm{null}}}{Q_{\mathrm{HH}}-Q_{\mathrm{null}}}\times 100\%.$$

### 3.3 Gap Closure
Pointwise NLL gap closure:
$$G_{\mathrm{NLL}} = \frac{\mathrm{NLL}_{\mathrm{raw}}-\mathrm{NLL}_{\mathrm{condition}}}{\mathrm{NLL}_{\mathrm{raw}}-H_{\mathrm{target}}}\times 100\%.$$

Relational $Q$-gap closure ($G_Q$) is defined analogously against the raw condition and human reference.

---

## 4. Primary Results

### Main Benchmark Table ($N=600$ Pilot Items)

| Method / Condition | Accuracy | NLL | Brier | JSD | $Q_{\text{supp}}\ (k=10)$ | $R_{\text{norm}}\ (\%)$ | $G_{\text{NLL}}\ (\%)$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **API ($T=0$) LPE (Diagnostic)** | 64.83% | 3.8597 | 0.3908 | 0.1733 | 0.03775 | 8.48% | — |
| **API ($T=1$) LPE (Primary Raw)** | **65.00%** | **3.8277** | **0.3889** | **0.1721** | **0.04077** | **9.72%** | **0.00%** |
| **Calibrated API ($T=1$) LPE ($T^*=10.36$)** | **64.83%** | **0.9308** | **0.1547** | **0.0738** | **0.04088** | **9.76%** | **91.29%** |
| **MCE (30 Samples at API $T=1$)** | **65.33%** | **1.5864** | **0.3559** | **0.1403** | **0.03311** | **6.64%** | **70.63%** |
| **MCE Finite-Noise Control Sim** | — | 1.5958 | — | — | 0.03268 | 6.52% | — |

### Core-Scale ($k=50$) Support & Excess Metrics

| Method / Condition | $Q_{\text{supp}}\ (k=50)$ | $Q_{\text{global\_excess}}$ | $Q_{\text{profile\_excess}}$ | $\text{CoreMass}_{\tau50}$ | $\text{CoreRecall}_{\tau50}$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **API ($T=1$) LPE (Primary Raw)** | 0.17390 | 0.02397 | 0.02812 | 0.18781 | 0.03756 |
| **Calibrated API ($T=1$) LPE** | 0.17631 | 0.02405 | 0.02740 | 0.19091 | 0.03818 |
| **MCE (30 Samples)** | 0.15906 | 0.01637 | 0.02422 | 0.16113 | 0.03223 |

---

## 5. Statistical Comparisons

Common 30-stratum item bootstraps used 1,000 resamples.

### 5.1 Calibration versus Raw API ($T=1$) LPE
Pointwise improvement was large and statistically clear:
- NLL difference: $95\%\ \mathrm{CI}=[-3.0731, -2.7447]$ ($p < 0.001$).
- Brier difference: $95\%\ \mathrm{CI}=[-0.2523, -0.2173]$ ($p < 0.001$).
- JSD difference: $95\%\ \mathrm{CI}=[-0.1050, -0.0923]$ ($p < 0.001$).
- Relational difference: $\Delta R_{\text{norm}} = \mathbf{+0.03\%}$ ($95\%\ \mathrm{CI}=[-0.94\%, +1.03\%]$, **spans zero**).

Thus, scalar calibration dramatically improved probability fit but did not produce a supported change in human relational recovery.

### 5.2 MCE versus Calibrated LPE
- MCE was pointwise worse: $\Delta \text{NLL} = +0.6571$ ($95\%\ \text{CI}: [+0.6122, +0.7048]$).
- MCE was relationally worse: $\Delta R_{\text{norm}} = -3.15\%$ ($95\%\ \text{CI}: [-4.27\%, -2.07\%]$).
- Majority-label accuracy difference was small and not statistically supported: $+0.33\%$ ($95\%\ \text{CI}: [-1.00\%, +1.83\%]$).

---

## 6. Finite-Sample Control for MCE

To determine whether MCE’s lower relational score required a distinct stochastic-generation explanation, the experiment simulated the MCE protocol directly from the six API ($T=1$) LPE distributions for each item over 1,000 trials.

Simulation results:
- $\operatorname{mean}(R_{\mathrm{sim}}) = 6.52\%$ ($95\%\text{ simulation interval} = [6.25\%, 6.80\%]$).
- Simulated NLL mean: $1.5958$ ($95\%\ \text{CI}: [1.5899, 1.6018]$).

Actual MCE ($R_{\mathrm{MCE}} = 6.64\%$) was at the **80.1st percentile** of the simulation distribution, with a two-sided Monte Carlo tail probability of **$p_{\mathrm{MC}} = 0.4110$**.

**Interpretation:**  
The actual MCE relational score is compatible with the behavior expected when only 30 samples are used to estimate the matched continuous LPE distributions. The experiment found no evidence of additional relational degradation beyond this finite-sample control.

---

## 7. Replicate-Subset Sensitivity

All exact balanced subsets of the five replicates per mapping were enumerated ($\binom{5}{1}=5$, $\binom{5}{2}=10$, $\binom{5}{3}=10$, $\binom{5}{4}=5$, $\binom{5}{5}=1$).

| Samples per Item | Replicates per Mapping | Number of Subsets | Median NLL | NLL Range | Median $R_{\text{norm}}$ | $R_{\text{norm}}$ Range |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **6** | 1 | 5 | 1.1634 | 1.1608–1.1693 | 6.41% | 6.09%–6.43% |
| **12** | 2 | 10 | 1.3280 | 1.3230–1.3336 | 6.61% | 6.26%–6.93% |
| **18** | 3 | 10 | 1.4389 | 1.4331–1.4425 | 6.66% | 6.44%–6.81% |
| **24** | 4 | 5 | 1.5218 | 1.5171–1.5225 | 6.61% | 6.59%–6.71% |
| **30** | 5 | 1 | 1.5864 | fixed | 6.64% | fixed |

*Note: The rising NLL with sample count reflects shrinking Jeffreys uniform prior mass ($0.5 / (N_{\text{samples}} + 1.5)$); as sample count rises, the estimate reveals more of the model’s sharply distributed sampling behavior.*

---

## 8. Interpretation

### 8.1 What Calibration Fixed
Scalar calibration corrected probability scale: large NLL improvement ($91.3\%$), large Brier improvement, large JSD improvement, with almost unchanged majority accuracy.

### 8.2 What Calibration Did Not Fix
Calibration did not materially reorganize the model toward the human neighborhood structure ($9.72\% \to 9.76\%$). The model became much better at assigning appropriate probability mass to labels without learning new item-to-item relationships.

### 8.3 What MCE Measured
MCE is a noisy empirical estimate of the model’s sampled decision distribution. With 30 samples, its distribution estimates lose substantial relational resolution even when majority-label accuracy remains similar.

### 8.4 Broader Scientific Implication
Pointwise calibration and relational recovery are distinct axes of model quality. A model may be:
- Well-calibrated but relationally weak;
- Relationally informative but poorly calibrated;
- Accurate on majority labels while compressing meaningful ambiguity structure.

---

## 9. Claim Boundaries

This experiment supports claims about:
- Aggregate ChaosNLI vote-distribution geometry;
- Gemma 3 12B under the frozen prompt and Ollama runtime;
- API-returned candidate-token scores;
- 30-sample MCE under six label mappings;
- Hellinger $k$-nearest-neighbor relational recovery;
- Cross-fitted scalar rescaling.

It does **not** establish:
- Demographic minority erasure;
- Individual annotator belief structures;
- Universal LLM behavior across all model families;
- Intrinsic model logits independent of the inference runtime;
- Causal equivalence between LPE and MCE;
- That $9.7\%$ is a universal model-resolution limit.

---

## 10. Relationship to the Broader ChaosNLI Program

E004 extends classifier findings to a modern local generative model:
- **E002**: Classifier temperature calibration greatly improves pointwise fit but barely changes relational recovery.
- **E004**: Gemma exhibits the exact same separation; calibration closes $\approx 91\%$ of the NLL gap while relational recovery remains unchanged.
- **E005**: Model size weakly increases residual conditional resolution.
- **E007**: Model diversity contributes complementary relational information.
- **E008**: Model and ensemble recovery can be expressed on a prototype-equivalent resolution scale.

---

## 11. Reproducibility and Source Artifacts

**Primary Code & Summary Source Files:**
- [`research/chaosnli/lab/e004_ollama_runner.py`](file:///c:/Users/admir/Github/shadowspace/research/chaosnli/lab/e004_ollama_runner.py)
- [`research/chaosnli/lab/e004_paper_ready_analysis.py`](file:///c:/Users/admir/Github/shadowspace/research/chaosnli/lab/e004_paper_ready_analysis.py)
- [`research/chaosnli/artifacts/E004/summaries/E004_gemma3_12b_paper_ready_summary.json`](file:///c:/Users/admir/Github/shadowspace/research/chaosnli/artifacts/E004/summaries/E004_gemma3_12b_paper_ready_summary.json)
- [`research/chaosnli/results/E004_STAGE1B_PROVENANCE.json`](file:///c:/Users/admir/Github/shadowspace/research/chaosnli/results/E004_STAGE1B_PROVENANCE.json)
- [`research/chaosnli/lab/registry/E004.toml`](file:///c:/Users/admir/Github/shadowspace/research/chaosnli/lab/registry/E004.toml)

**Raw Response Records:**
- API ($T=0$) LPE: 3,600 valid records (`pilot600_gemma3-12b_v2_abc_lpe.jsonl`)
- API ($T=1$) LPE: 3,600 valid records (`pilot600_gemma3-12b_v2_abc_t10_lpe.jsonl`)
- MCE: 18,000 valid records (`pilot600_gemma3-12b_v2_abc_mce.jsonl`)

**Durable Release Tag:**  
`e004-stage1b-primary-frozen-v1`

---

## 12. Manuscript-Ready Abstract

> Human judgment disagreement contains relational structure that is not captured by majority-label accuracy or itemwise calibration alone. We evaluated Gemma 3 12B on 600 ChaosNLI items using candidate-token log-probability estimation under six label-symbol permutations and Monte Carlo estimation from 30 sampled decisions per item. Five-fold cross-fitted scalar rescaling reduced API ($T=1$) soft-label NLL from 3.83 to 0.93, closing 91.3% of the gap to the empirical target entropy, while normalized relational recovery remained effectively unchanged at approximately 9.7%. Thirty-sample Monte Carlo estimates achieved comparable majority accuracy ($65.3\%$) but lower relational recovery of 6.6%; this value was compatible with a matched finite-sample simulation based on the continuous API ($T=1$) distributions ($p=0.411$). These results show that probability calibration can repair score scale without recovering additional human disagreement geometry and that repeated sampling may require substantially larger budgets to preserve relational structure.
