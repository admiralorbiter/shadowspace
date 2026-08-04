# E004 Cross-Model Extension: Qwen 2.5 14B versus Gemma 3 12B

**Experiment family**: E004 — Relational alignment of generative LLM judgment distributions  
**Dataset**: ChaosNLI pilot ($N=600$)  
**Primary elicitation**: API ($T=1.0$) log-probability estimation under all six ($S_3$) label-symbol permutations  
**Models**: Gemma 3 12B and Qwen 2.5 14B  
**Analysis status**: Audited and frozen  
**Freeze branch**: `chaosnli`  
**Reported freeze commit**: `22685d0`  
**Primary result**: `research/chaosnli/results/E004_qwen2.5_14b_summary.json`  
**Primary provenance**: `research/chaosnli/results/E004_qwen2.5_14b_PROVENANCE.json`  

---

## Executive Summary

This cross-model extension tested whether the separation between pointwise calibration and relational disagreement recovery observed for Gemma 3 12B generalizes to a different instruction-tuned model.

Qwen 2.5 14B was evaluated on the same 600 ChaosNLI items, with the same prompt, symbol set, six label permutations, human posterior-support target, neighborhood scale, analytic null, cross-fitting scheme, and bootstrap design used in the frozen Gemma 3 analysis.

Qwen was more severely miscalibrated than Gemma before post-hoc correction:

$$\text{NLL}_{\text{Qwen,raw}} = 5.2986, \qquad \text{NLL}_{\text{Gemma,raw}} = 3.8277.$$

Nevertheless, Qwen recovered more human disagreement geometry:

$$R_{\text{Qwen,raw}} = 11.85\%, \qquad R_{\text{Gemma,raw}} = 9.72\%.$$

The paired 30-stratum bootstrap difference was:

$$\Delta R_{\text{raw}} = R_{\text{Qwen,raw}} - R_{\text{Gemma,raw}} = +2.13 \text{ percentage points}, \qquad 95\% \text{ CI} = [+0.56, +3.78].$$

Cross-fitted scalar calibration greatly improved Qwen’s pointwise probability estimates and produced a larger relational change than it did for Gemma:

$$R_{\text{Qwen,cal}} = 14.86\%, \qquad R_{\text{Gemma,cal}} = 9.76\%.$$

Under the primary censored-token convention, the calibrated cross-model difference was:

$$\Delta R_{\text{cal}} = +5.11 \text{ percentage points}, \qquad 95\% \text{ CI} = [+3.36, +6.67].$$

The point estimate slightly exceeded the preregistered five-percentage-point practical margin, but the confidence interval included smaller effects and therefore did not establish that the population-level difference exceeded five points.

The calibration effect also differed between the two models:

$$\Delta R_{\text{Qwen,calibration}} = +3.01 \text{ points}, \qquad \Delta R_{\text{Gemma,calibration}} = +0.04 \text{ points}.$$

The resulting two-model interaction was:

$$\Delta\Delta R = +2.97 \text{ points}, \qquad 95\% \text{ CI} = [+1.33, +4.55].$$

This interaction remained positive under a fixed (-20) censored-token stress test:

$$\Delta\Delta R_{\text{stress}} = +1.80 \text{ points}, \qquad 95\% \text{ CI} = [+0.14, +3.35].$$

The evidence supports three conclusions:

1. Qwen 2.5 14B recovered modestly more human disagreement geometry than Gemma 3 12B on the matched pilot.
2. The direction of this advantage was robust to two treatments of censored candidate probabilities.
3. Scalar calibration affected the relational geometry of these two models differently, motivating—but not yet proving—a broader model-family hypothesis.

---

## 1. Research Questions

This extension addressed four questions:

1. Does the calibration-versus-relational-alignment separation observed in Gemma replicate in another model?
2. Does Qwen recover more human disagreement geometry than Gemma on the same items?
3. Does scalar calibration change Qwen’s relational recovery more than it changes Gemma’s?
4. Are those conclusions robust to uncertainty from candidate symbols that fell outside Ollama’s returned top-20 log-probability list?

---

## 2. Matched Experimental Design

### 2.1 Dataset and Human Target

Both models were evaluated on the same 600-item ChaosNLI pilot.

The sample was balanced across source dataset, human majority label, and human entropy quintile.

The human relational target was the frozen posterior expected fuzzy $k$-nearest-neighbor support matrix with:

$$k=10, \qquad Q_{\mathrm{HH}} = 0.26337965.$$

### 2.2 Probability Elicitation

Each item was evaluated under all six permutations of the three semantic labels across the symbols A, B, and C.

For each model:

$$600 \text{ items} \times 6 \text{ permutations} = 3,600 \text{ requests}.$$

The primary condition used API temperature ($T=1.0$). Reasoning output was disabled for Qwen using `reasoning_effort = none`.

### 2.3 Cross-Fitted Scalar Calibration

The analysis reproduced the frozen E004 estimator:

1. Construct the 30 registered strata.
2. Assign five folds by rank within each stratum modulo five.
3. Fit one scalar temperature ($T_f^*$) on the other four folds using `minimize_scalar`.
4. Apply ($T_f^*$) independently to every permutation-specific score vector.
5. Average the six calibrated semantic distributions.
6. Apply ($T_f^*$) to all 600 items to build one coherent fold-specific graph.
7. Score only held-out focal rows.
8. Aggregate held-out focal-row contributions across folds.

Qwen’s fitted fold temperatures were:

$$[17.2041, 17.0689, 17.3332, 17.2896, 17.0594],$$

with mean:

$$\overline{T}^* = 17.1910.$$

### 2.4 Regression Gate

Before evaluating Qwen, the generic pipeline reran the frozen Gemma 3 responses.

It numerically reproduced all frozen Gemma pointwise and relational estimands and all five fitted temperatures within predetermined metric-specific tolerances ranging from $10^{-6}$ to $10^{-4}$.

The regression gate covered raw and calibrated NLL, Brier score, JSD, $Q_{\mathrm{support}}$, analytic nulls, normalized relational recovery, mean fitted temperature, and all five fold-specific temperatures.

### 2.5 Bootstrap Uncertainty

Uncertainty was estimated using 1,000 common 30-stratum focal-row bootstrap samples.

For calibrated estimators, each bootstrap replicate resampled both the held-out focal-row support contribution and the corresponding item-level fold-specific null.

The same bootstrap samples were used for Gemma and Qwen, allowing paired estimates of raw and calibrated cross-model differences, within-model calibration gains, and the calibration-by-model interaction.

---

## 3. Primary Results

| Model and Condition | NLL (nats) | JSD (nats) | JSD (bits) | $Q_{\mathrm{support}}$ | $R_{\mathrm{norm}}$ | Effective Bits ($b$) | $K_{\mathrm{eff}}$ |
|---|---|---|---|---|---|---|---|
| Gemma 3 12B raw LPE | 3.8277 | 0.1721 | 0.2483 | 0.04077 | 9.72% | 1.370 | 2.59 |
| Gemma 3 12B calibrated LPE | 0.9308 | 0.0738 | 0.1064 | 0.04088 | 9.76% | 1.374 | 2.59 |
| Qwen 2.5 14B raw LPE | 5.2986 | 0.1646 | 0.2375 | 0.04595 | 11.85% | 1.574 | 2.98 |
| Qwen 2.5 14B calibrated LPE | 0.8837 | 0.0605 | 0.0873 | 0.05342 | 14.86% | 1.816 | 3.52 |

*Here, $K_{\mathrm{eff}}$ is a prototype-equivalent resolution scale. It should not be interpreted as the literal number of internal states or prototypes represented by a model.*

Under the primary censored-token convention:

$$R_{\text{Qwen,raw}} = 11.85\%, \qquad 95\% \text{ CI} = [10.54, 13.22],$$

and:

$$R_{\text{Qwen,cal}} = 14.86\%, \qquad 95\% \text{ CI} = [13.40, 16.29].$$

---

## 4. Paired Cross-Model Contrasts

### 4.1 Raw Contrast

Under the primary (-40) floor:

$$\Delta R_{\text{raw}} = +2.13 \text{ points}, \qquad 95\% \text{ CI} = [+0.56, +3.78].$$

The paired interval excluded zero, supporting a positive raw relational difference at the nominal 95% level. The difference remained below the preregistered five-point practical margin.

In prototype-equivalent terms:

$$\Delta b_{\text{raw}} = +0.204 \text{ bits}.$$

### 4.2 Calibrated Contrast

Under the primary (-40) floor:

$$\Delta R_{\text{cal}} = +5.11 \text{ points}, \qquad 95\% \text{ CI} = [+3.36, +6.67].$$

The interval excluded zero, supporting a positive calibrated difference. The point estimate exceeded five percentage points, but the lower confidence limit did not. The data therefore do not establish that the population-level effect exceeds the five-point practical margin.

In prototype-equivalent terms:

$$\Delta b_{\text{cal}} = +0.442 \text{ bits}.$$

---

## 5. Calibration-by-Model Interaction

For Qwen:

$$\Delta R_{\text{Qwen}} = R_{\text{Qwen,cal}} - R_{\text{Qwen,raw}} = +3.01 \text{ points}.$$

For Gemma:

$$\Delta R_{\text{Gemma}} = R_{\text{Gemma,cal}} - R_{\text{Gemma,raw}} = +0.04 \text{ points}.$$

The two-model interaction was:

$$\Delta R_{\text{Qwen}} - \Delta R_{\text{Gemma}} = +2.97 \text{ points}, \qquad 95\% \text{ CI} = [+1.33, +4.55].$$

This supports the conclusion that scalar calibration affected relational recovery differently for Qwen 2.5 14B and Gemma 3 12B.

Because only one model from each lineage was evaluated, this result should be called a two-model interaction, not definitive evidence of a general model-family effect.

---

## 6. Censored-Candidate Sensitivity Analysis

In 46 of the 3,600 Qwen requests, at least one candidate symbol was not present in the returned top-20 candidate list.

These records were evaluated under two conventions:
- **Bound A — primary floor**: assign the missing candidate logprob (-40).
- **Bound B — fixed stress test**: assign the missing candidate logprob (-20).

*Bound B is a deliberately less-extreme stress test. It is not the observed twentieth-token threshold for each request.*

| Estimate | Bound A: (-40) floor | Bound B: (-20) stress test |
|---|---|---|
| Qwen raw ($R_{\mathrm{norm}}$) | 11.85% [10.54%, 13.22%] | 12.25% [10.94%, 13.62%] |
| Qwen calibrated ($R_{\mathrm{norm}}$) | 14.86% [13.40%, 16.29%] | 14.09% [12.63%, 15.52%] |
| Raw Qwen–Gemma contrast | +2.13 [0.56, 3.78] | +2.53 [0.89, 4.28] |
| Calibrated Qwen–Gemma contrast | +5.11 [3.36, 6.67] | +4.33 [2.66, 6.02] |
| Qwen calibration gain | +3.01 | +1.84 |
| Two-model interaction | +2.97 [1.33, 4.55] | +1.80 [0.14, 3.35] |

*All differences are in percentage points.*

The absolute shift between the two censoring conventions was 0.40 points for raw relational recovery and 0.77 points for calibrated relational recovery.

The direction of the Qwen raw advantage, calibrated advantage, within-model calibration gain, and two-model interaction remained positive under both bounds.

The magnitudes—especially the calibrated contrast and interaction—were sensitive to the censoring convention. The stress test therefore supports directional robustness, not invariance of effect size.

---

## 7. Interpretation

### 7.1 Calibration and Relational Recovery Remain Distinct

Both models were severely miscalibrated before correction.

Calibration dramatically reduced pointwise error for both models, but its relational effect differed:
- Gemma’s relational recovery was nearly unchanged.
- Qwen’s relational recovery increased by approximately three points under the primary convention.

Scalar calibration therefore cannot be treated as universally topology-invariant. Its relational effect appears to depend on the geometry of the model’s original probability field.

### 7.2 Better Pointwise Calibration Does Not Explain Qwen’s Raw Advantage

Before calibration, Qwen had worse NLL than Gemma but better relational recovery.

Qwen’s raw relational advantage therefore cannot be explained by better pointwise probability fit. This reinforces the distinction between probability-scale fidelity and relational disagreement fidelity.

### 7.3 Model Choice Affects Pluralistic Resolution

On the matched pilot, Qwen reached a higher prototype-equivalent resolution tier than Gemma:

$$K_{\mathrm{eff,raw}}: 2.98 \text{ versus } 2.59,$$

and:

$$K_{\mathrm{eff,cal}}: 3.52 \text{ versus } 2.59.$$

This suggests that architecture, training data, or post-training can affect how many human disagreement patterns a model resolves. The experiment does not identify which factor caused the difference.

### 7.4 The Interaction is a Hypothesis Generator

The positive two-model interaction suggests that calibration may expose relational structure already latent in Qwen’s score geometry but not in Gemma’s.

A broader family-level conclusion requires additional models, ideally Gemma 4 12B, Qwen3 14B or a Qwen scale ladder, and at least one independent family.

---

## 8. Claim Boundaries

### Supported
- Qwen 2.5 14B had higher raw relational recovery than Gemma 3 12B on the matched 600-item pilot.
- The paired raw difference was positive under both censoring conventions.
- Under the primary convention, Qwen had a larger calibrated relational point estimate than Gemma.
- The two-model calibration interaction was positive under both censoring conventions.
- The generic pipeline numerically reproduced the frozen Gemma analysis within predetermined tolerances.
- Pointwise calibration and relational recovery are empirically separable.

### Not Established
- Qwen is universally more pluralistic than Gemma.
- The true calibrated cross-model difference exceeds five percentage points.
- All Qwen-family models respond to calibration differently from all Gemma-family models.
- The interaction is caused by architecture rather than data, post-training, quantization, or inference-runtime behavior.
- Prototype-equivalent states correspond to literal internal model states.
- The results generalize beyond the ChaosNLI pilot.

---

## 9. Paper-Safe Wording

### Raw Comparison
On the same 600 ChaosNLI items, Qwen 2.5 14B recovered modestly more human disagreement geometry than Gemma 3 12B. Under the primary censored-token convention, the paired 30-stratum bootstrap difference was (+2.13) percentage points, with a 95% interval of ([+0.56, +3.78]). The positive difference persisted under a fixed (-20) stress-test convention, although its magnitude remained below the preregistered five-point practical margin.

### Calibrated Comparison
Under the primary censored-token convention, Qwen’s calibrated relational recovery exceeded Gemma’s by (5.11) percentage points, with a 95% interval of ([+3.36, +6.67]). The point estimate slightly exceeded the five-point practical margin, but the confidence interval included smaller effects. Under the fixed (-20) stress test, the calibrated difference remained positive but decreased to (4.33) percentage points.

### Two-Model Interaction
Calibration increased relational recovery more for Qwen 2.5 14B than for Gemma 3 12B. The paired interaction was (+2.97) percentage points under the primary convention and (+1.80) points under the fixed (-20) stress test; both bootstrap intervals excluded zero. This supports a model-specific calibration interaction and motivates broader testing across additional model families.

---

## 10. Reproducibility

### Primary Artifacts
- `research/chaosnli/results/E004_qwen2.5_14b_summary.json`
- `research/chaosnli/results/E004_qwen2.5_14b_PROVENANCE.json`
- `research/chaosnli/lab/analyze_llm_lpe.py`
- `research/chaosnli/results/E008_pilot_600_curve.json`
- `research/chaosnli/results/E008_pilot_600_provenance.json`

### Frozen Inputs
- `research/chaosnli/artifacts/E004/manifests/pilot_600.jsonl`
- `research/chaosnli/artifacts/E004/pilot_support/S_hellinger_k010_pilot.bin`
- `research/chaosnli/artifacts/E004/raw_responses/pilot600_gemma3-12b_v2_abc_t10_lpe.jsonl`
- `research/chaosnli/artifacts/E004/raw_responses/pilot600_qwen2.5-14b_v2_abc_t10_lpe.jsonl`
- `research/chaosnli/artifacts/E004/summaries/E004_gemma3_12b_paper_ready_summary.json`

### Frozen Hashes
- **Qwen response SHA-256**: `45221d113d2218036c0dfec0ade257d0b1b10d5f9e5af003b2d92d56d37c12c7`
- **Pilot manifest SHA-256**: `318c67354242771d2812b0d26dbd3d89e084ecc0afece250cae453d6531f4851`
- **Pilot support SHA-256**: `29ff35cfddbd1a4c30858eb2feac58aeb0c087aff99850736007a2a49fb40245`

---

## 11. Recommended Next Experiment

The next model should test whether the interaction generalizes beyond these two models.

Recommended order:
1. **Gemma 4 12B** — same-family successor to Gemma 3.
2. **Qwen3 14B** — later Qwen generation at a similar resource scale.
3. **One independent family** such as Llama 3.1 8B or Phi-4 14B.

For every new model, use the same 600 items, API (T=1), six permutations, reasoning controls, support matrix, regression-gated analysis pipeline, and common paired 30-stratum bootstrap samples.

The central next question is: *Does the calibration-induced relational gain track model family, model generation, model scale, or some other property of the score geometry?*
