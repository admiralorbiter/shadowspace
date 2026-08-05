# Beyond Pointwise Calibration: Relational Recovery of Human Disagreement Across Four Open-Weight Language Models
## A ChaosNLI Pilot Study

**Author:** Jonathan Lane  
**Date:** August 5, 2026  
**Keywords:** natural language inference, human disagreement, calibration, pluralistic alignment, model ensembles, uncertainty  

---

> **Draft status.** Working-paper draft based on the frozen E004 pilot analysis at repository commit `5cc25272ae04f5eb983b8f9c5e3db7e740dd4b87`. The study uses a stratified 600-item pilot and has not yet been replicated on an independent validation sample. Coalition analyses are explicitly treated as exploratory unless otherwise stated.

## Abstract

Natural-language-inference benchmarks typically reduce multiple human judgments to a single majority label, even though disagreement can be persistent, structured, and semantically meaningful (Pavlick & Kwiatkowski, 2019; Nie et al., 2020). This paper asks whether language models recover not only the average human label distribution for each item, but also the relational geometry connecting items with similar patterns of human disagreement. We evaluate four locally deployed, open-weight instruction-tuned models—Gemma 3 12B, Gemma 4 12B, Qwen 2.5 14B, and Qwen3 14B—on a stratified 600-item ChaosNLI pilot. For every item, we estimate a three-class judgment distribution from first-token log probabilities under all six label-symbol permutations. We evaluate pointwise fidelity using negative log likelihood, Brier score, and Jensen–Shannon divergence, and introduce a normalized relational-recovery statistic that compares model-induced nearest-neighbor graphs with a posterior human-support graph. All calibrated relational estimates use five-fold, 30-stratum cross-fitting and fold-coherent graph construction; uncertainty is estimated with 10,000 stratified percentile-bootstrap draws.

Pointwise and relational performance separated sharply. Gemma 4 improved calibrated relational recovery over Gemma 3 by 1.83 percentage points despite worse pointwise fit. Qwen3 improved raw pointwise fit over Qwen 2.5 but showed nearly unchanged raw relational recovery and lower calibrated relational recovery. Qwen 2.5 was the only model whose marginal 95% bootstrap interval showed a clearly positive within-model relational calibration effect: +3.01 points, 95% CI [+1.84, +4.17]. The calibrated family-by-generation interaction was -3.73 points, 95% CI [-5.94, -1.50], indicating divergent generational changes within this selected four-model panel. Exploratory cross-family opinion pools achieved the strongest relational estimates. A Gemma 4–Qwen 2.5 logarithmic pool reached 18.18% normalized relational recovery under the primary censoring convention and remained above Qwen 2.5 under a fixed -20 stress test. Mechanism analyses attribute the linear coalition’s positive support gain primarily to newly created pooled relationships and Gemma 4-unique relationships.

These results suggest that pointwise calibration, relational resolution, calibration response, and cross-model complementarity are distinct empirical properties. The pilot motivates independent validation and a broader evaluation framework for distributional pluralism that treats the geometry of disagreement as an object of study rather than annotation noise.

---

## 1. Introduction

Human judgments are often summarized as if a task had one objectively correct label and disagreement merely reflected annotator error. Natural language inference (NLI) provides a well-studied counterexample. People can interpret the same premise–hypothesis pair differently because of underspecified context, lexical ambiguity, pragmatic assumptions, or legitimate variation in inference standards. These disagreements frequently persist as more annotations are collected and cannot be dismissed as random noise (Pavlick & Kwiatkowski, 2019). ChaosNLI made this issue measurable at scale by collecting 100 judgments per example for thousands of NLI items and showing that contemporary models failed to reproduce the resulting human label distributions (Nie et al., 2020).

The distinction matters for language-model evaluation. A model may predict the majority label accurately while misrepresenting the distribution of plausible human judgments. Conversely, a model may be poorly calibrated in absolute probability scale while preserving meaningful relationships among ambiguous cases. Majority-label calibration can itself become conceptually problematic when the target population genuinely disagrees (Baan et al., 2022). A distributional evaluation should therefore ask at least two different questions:

1. **Pointwise fidelity:** Does the model assign each item a probability distribution close to the empirical human distribution?
2. **Relational fidelity:** Do items that elicit similar patterns of human disagreement also occupy nearby positions in the model’s probability geometry?

Most calibration work emphasizes the first question. Temperature scaling is a simple and effective post-hoc method for correcting overconfident neural probabilities (Guo et al., 2017), and language models have repeatedly been shown to produce poorly calibrated confidence estimates (Jiang et al., 2021). Yet a global temperature transformation may affect neighborhood structure differently across models. If two items remain near each other after calibration, pointwise probabilities may improve without changing relational geometry. If items cross neighborhood boundaries, calibration can reorganize the model’s representation of disagreement.

This paper studies that separation across a matched four-model panel:
- Gemma 3 12B;
- Gemma 4 12B;
- Qwen 2.5 14B;
- Qwen3 14B.

The panel has a 2 × 2 structure: two model families and two generations within each family. All models are evaluated under the same frozen local inference contract, the same 600 ChaosNLI items, the same six label-symbol permutations, and the same relational target. The analysis also examines equal-weight linear and logarithmic opinion pools. Probability pooling has a long history in statistics (Genest & Zidek, 1986), while recent pluralistic-alignment work has argued that collaboration among models may preserve perspectives or capabilities that a single averaged model misses (Feng et al., 2024; Feng et al., 2026).

The study makes four contributions:
1. It introduces a relational evaluation of soft NLI judgments based on overlap between model and human nearest-neighbor structure.
2. It shows that pointwise fit and relational recovery can move in opposite directions across generations.
3. It identifies model-specific differences in the relational effect of scalar calibration.
4. It provides exploratory evidence that cross-family probability pools can recover relational structure absent from the best single model.

The work is framed as a pilot study, not a definitive model-family benchmark. The sample is drawn from one task and one frozen local runtime. The family-by-generation conclusions concern the four evaluated model packages, not all Gemma or Qwen models. The coalition census was explored on the same pilot and therefore requires independent confirmation.

---

## 2. Related work

### 2.1 Human disagreement in NLI
Pavlick and Kwiatkowski showed that disagreement in textual inference is often stable and semantically meaningful, arguing that models should represent the full distribution of plausible judgments rather than only a majority label (Pavlick & Kwiatkowski, 2019). ChaosNLI operationalized that proposal by collecting 100 annotations per example and documenting substantial disagreement across SNLI, MNLI, and abductive NLI (Nie et al., 2020). Later work found that LLM-era systems continue to struggle with collective human opinions and confidence in NLI (Wang et al., 2024).

This paper uses the ChaosNLI judgment distributions as targets but shifts the evaluation emphasis. Instead of asking only whether each model distribution matches each human distribution independently, it asks whether the relationships among distributions are preserved.

### 2.2 Calibration under disagreement
Calibration evaluates whether predicted probabilities correspond to observed frequencies. Temperature scaling fits a single positive scalar to logits and has become a standard post-hoc baseline because it often reduces overconfidence without changing the predicted class (Guo et al., 2017). In language models, probability estimates can also be severely miscalibrated, motivating post-hoc correction and alternative elicitation strategies (Jiang et al., 2021).

However, calibrating against a majority label can be misleading when humans disagree. Baan et al. argue that conventional correctness-based calibration is theoretically problematic in such settings and propose measures that respect class frequencies, rankings, and entropy (Baan et al., 2022). The present study therefore evaluates model probabilities directly against the full human distribution with cross entropy, Brier score, and Jensen–Shannon divergence. Relational recovery is treated as a separate estimand rather than another form of majority-label calibration.

### 2.3 Soft labels and distributional targets
Learning from full human label distributions has improved robustness and out-of-distribution behavior in image classification (Peterson et al., 2019). More generally, proper scoring rules reward honest probabilistic forecasts and provide principled tools for comparing predictive distributions (Brier, 1950; Gneiting & Raftery, 2007). The present study does not train on soft labels; instead, it evaluates whether pretrained instruction-tuned LLMs expose human-like distributions through their first-token probabilities.

### 2.4 Distributional pluralism
Pluralistic alignment research distinguishes several ways an AI system might represent diverse human perspectives. “Distributional pluralism” specifically concerns matching a population-level distribution rather than collapsing it to a single average response (Sorensen et al., 2024). NLI disagreement is not equivalent to moral or political pluralism, but it provides a controlled setting in which distributional diversity is densely annotated and can be measured.

This paper’s relational statistic extends the distributional perspective: two models may have similar average pointwise scores while organizing disagreement patterns differently. Such organization can matter when systems are used to retrieve analogous cases, identify ambiguous clusters, or combine specialized models.

### 2.5 Opinion pooling and multi-model collaboration
Linear and logarithmic opinion pools are classical methods for combining probability distributions (Genest & Zidek, 1986). A linear pool averages probabilities, while a logarithmic pool takes a normalized weighted geometric mean. They have different theoretical properties and can trade calibration, sharpness, and consensus differently.

Recent work on modular pluralism uses collaboration among language models to represent perspectives that one model may average away (Feng et al., 2024). Broader arguments for multi-LLM collaboration similarly emphasize compositional coverage and pluralism (Feng et al., 2026). The current study evaluates a narrower logit-level form of collaboration: equal-weight pooling of semantic NLI distributions.

### 2.6 Model families
Gemma 3 introduced a family of efficient multimodal open models with long context and substantial architectural and post-training changes (Gemma Team, Google DeepMind, 2025). Gemma 4 extends the family with new dense and mixture-of-experts architectures, multimodality, and integrated reasoning capabilities (Gemma Team, Google DeepMind, 2026). Qwen 2.5 scaled pretraining and post-training while releasing dense instruction-tuned models across multiple sizes (Qwen Team, 2024). Qwen3 integrates thinking and non-thinking modes within a unified model family and reports broader multilingual and reasoning capabilities (Qwen Team, 2026).

The current experiment does not attempt to attribute outcomes to any one architectural or training change. The model-package generation is the unit of comparison.

---

## 3. Research questions

The frozen pilot addresses six research questions:

- **RQ1 — Pointwise versus relational fidelity:** Do models with better pointwise probability fit also recover more human disagreement geometry?
- **RQ2 — Calibration response:** Does five-fold scalar temperature calibration change relational recovery, and is the effect consistent across models?
- **RQ3 — Generational change:** Do successor models improve relational recovery within Gemma and Qwen?
- **RQ4 — Family-by-generation interaction:** Does generational change differ between the two evaluated families?
- **RQ5 — Mechanism:** Which levels of human support account for calibration-induced changes in model neighborhoods?
- **RQ6 — Coalition complementarity:** Can equal-weight cross-model pools recover more human disagreement geometry than the best individual model?

---

## 4. Data and experimental design

### 4.1 ChaosNLI pilot
The experiment uses 600 items drawn from the SNLI and MNLI portion of ChaosNLI. ChaosNLI provides 100 human judgments per item, allowing each premise–hypothesis pair to be represented by an empirical distribution over:
- entailment;
- neutral;
- contradiction.

The pilot is stratified into 30 cells:
$$[2\ \text{source datasets}\times3\ \text{majority labels}\times5\ \text{human-entropy quintiles}].$$

The same 30 strata are used for fold construction and bootstrap resampling.

Let the human judgment distribution for item $i$ be:
$$p_i = \left(p_{i,E}, p_{i,N}, p_{i,C}\right).$$

The empirical target entropy averaged over the pilot is:
$$H_{\mathrm{human}} = 0.65427\ \text{nats}.$$

### 4.2 Human relational target
Pointwise metrics compare each model distribution with $p_i$. The relational target additionally represents whether pairs of items are neighbors under uncertainty in the human distributions.

The frozen pipeline constructs a posterior expected fuzzy $k$-nearest-neighbor support matrix:
$$S \in [0,1]^{N\times N},\qquad k=10.$$

Here $S_{ij}$ represents posterior support that item $j$ belongs to item $i$’s human-disagreement neighborhood. The frozen human–human relational reference is:
$$Q_{\mathrm{HH}} = 0.26337965.$$

The support matrix is treated as a fixed preregistered target in all model analyses.

### 4.3 Models and local runtime

| Key | Model package | Scale | Quantization | Context |
|---|---|---|---|---|
| **G3** | `gemma3:12b` | 12B | Q4_K_M | 131,072 |
| **G4** | `gemma4:12b` | 11.9B | Q4_K_M | 262,144 |
| **Q2.5** | `qwen2.5:14b` | 14B | Q4_K_M | 131,072 |
| **Q3** | `qwen3:14b` | 14B | Q4_K_M | 262,144 |

All models were deployed through Ollama 0.32.5. Full model digests are listed in Section 10.4.

The scientific estimand is explicitly runtime-bound: the semantic first-token probability distribution exposed by each pinned Ollama model package under the frozen OpenAI-compatible endpoint contract.

A preflight audit found that native packaged and matched sampler settings produced nearly identical semantic distributions on a small diagnostic sample, but the native and OpenAI-compatible endpoints were not numerically interchangeable. All primary comparisons therefore use the same `/v1/chat/completions` endpoint.

### 4.4 Prompt and inference contract
The system prompt was:
```text
Assume the premise is true.

Determine the relationship of the hypothesis to the premise.
```

The user prompt presented the premise, hypothesis, label definitions, and three answer symbols. Each item was evaluated under all six permutations of the mapping between semantic labels and the symbols A, B, and C.

The frozen request contract was:
```json
{
  "temperature": 1.0,
  "max_tokens": 1,
  "logprobs": true,
  "top_logprobs": 20,
  "reasoning_effort": "none"
}
```

Each model therefore produced $600 \times 6 = 3{,}600$ requests. Transport gates required valid single-symbol outputs, no reasoning preamble, six unique permutations per item, and correctly formatted candidate log probabilities.

### 4.5 Permutation-invariant semantic distributions
For item $i$, permutation $\pi$, and temperature $T$, semantic probabilities were computed by applying softmax to the three candidate-symbol log probabilities after mapping symbols back to entailment, neutral, and contradiction:
$$q_{i,\pi}(T) = \operatorname{softmax}\left(\frac{\ell_{i,\pi}}{T}\right).$$

The item-level distribution was the mean across all six permutations:
$$q_i(T) = \frac{1}{6}\sum_{\pi \in S_3}q_{i,\pi}(T).$$

This design reduces dependence on any one arbitrary label-symbol assignment.

---

## 5. Metrics

### 5.1 Pointwise metrics
**Negative log likelihood:**
$$-\frac{1}{N}\sum_{i=1}^{N}\sum_{c}p_{ic}\log q_{ic}.$$
NLL is minimized when $q_i=p_i$ and is closely related to the logarithmic proper scoring rule (Gneiting & Raftery, 2007).

**Brier score:**
$$\frac{1}{N}\sum_{i=1}^{N}\sum_{c}(q_{ic}-p_{ic})^2.$$
The Brier score is a quadratic probability score (Brier, 1950).

**Jensen–Shannon divergence:**
$$\frac12 D_{\mathrm{KL}}(p_i\Vert m_i)+\frac12 D_{\mathrm{KL}}(q_i\Vert m_i),$$
where $m_i = \frac12(p_i+q_i)$. JSD is symmetric and bounded, and is reported in nats and bits (Lin, 1991).

### 5.2 Model neighborhoods
Hellinger distance between model distributions defines a complete distance matrix:
$$D_{ij} = \sqrt{1-\sum_c \sqrt{q_{ic}q_{jc}}}.$$

Each row is converted into a fuzzy ($k=10$) neighborhood weight vector $W_{i\cdot}$, with fractional tie handling and $\sum_j W_{ij}=k$.

### 5.3 Posterior-support overlap
Model support overlap is:
$$Q_{\mathrm{support}} = \frac{1}{Nk}\sum_{i,j}W_{ij}S_{ij}.$$

A dataset-block analytic null ($Q_{\mathrm{null}}$) accounts for SNLI/MNLI block densities and the model graph’s allocation of edges across blocks.

Normalized relational recovery is:
$$R_{\mathrm{norm}} = 100\cdot\frac{Q_{\mathrm{support}}-Q_{\mathrm{null}}}{Q_{\mathrm{HH}}-Q_{\mathrm{null}}}.$$

Interpretation:
- $R_{\mathrm{norm}}=0\%$: model overlap equals the analytic block-density null;
- $R_{\mathrm{norm}}=100\%$: model overlap reaches the frozen human–human reference.

This is an external evaluation scale, not a claim about literal internal model representations.

### 5.4 Prototype-equivalent resolution
A frozen rate–distortion ladder maps relational recovery to a prototype-equivalent resolution:
$$b = \log_2 K_{\mathrm{eff}}.$$

The mapping is obtained by interpolation against controlled prototype quantizers. $K_{\mathrm{eff}}$ should be interpreted only as an equivalent point on this evaluation curve, not as the number of prototypes inside a neural network.

---

## 6. Calibration and statistical analysis

### 6.1 Five-fold cross-fitted temperature scaling
Items were assigned to five folds by rank within each of the 30 strata modulo five.

For fold $f$:
1. Fit $T_f^*$ on the other four folds by minimizing soft-target NLL.
2. Apply $T_f^*$ to all 600 items.
3. Build one complete fold-specific graph $W_f$.
4. Score only held-out focal rows ($i\in V_f$).

This fold-coherent procedure avoids comparing rows transformed under different fold temperatures within one graph.

### 6.2 Bootstrap uncertainty
All primary intervals use 10,000 stratified percentile-bootstrap draws. Within each replicate, items are resampled with replacement inside each of the 30 strata. The same sampled indices are used for all models in paired contrasts.

For calibrated estimates, each replicate resamples both held-out focal-row support contributions and item-level fold-specific null values.

The paper reports percentile intervals rather than bootstrap-derived pseudo-$p$-values.

### 6.3 Censored candidates
A candidate was censored when it did not appear in the returned top-20 token list. The primary convention assigned the missing candidate a log probability of $-40$. A stress-test convention assigned $-20$.

Observed counts included:
- Qwen 2.5: 46 censored requests;
- Gemma 4: 249 censored requests;
- Qwen3: 0 censored requests.

Individual-model and logarithmic-pool sensitivity analyses re-extracted logits and refit calibration under both conventions.

---

## 7. Results

### 7.1 Single-model benchmark

| Model | Raw NLL | Cal. NLL | Raw JSD | Cal. JSD | Raw $R_{\text{norm}}$ | Cal. $R_{\text{norm}}$ | Cal. Gain | 95% Percentile CI for Gain |
|---|---|---|---|---|---|---|---|---|
| **Gemma 3 12B** | 3.8277 | 0.9308 | 0.1721 | 0.0738 | 9.72% | 9.76% | +0.04 pp | [-0.94, +1.01] |
| **Gemma 4 12B** | 5.2370 | 0.9549 | 0.1822 | 0.0799 | 11.38% | 11.59% | +0.21 pp | [-0.88, +1.30] |
| **Qwen 2.5 14B** | 5.2986 | 0.8837 | 0.1646 | 0.0605 | 11.85% | 14.86% | **+3.01 pp** | **[+1.84, +4.17]** |
| **Qwen3 14B** | 4.5008 | 0.8865 | 0.1507 | 0.0609 | 11.92% | 12.96% | +1.04 pp | [-0.03, +2.15] |

Three separations are immediately visible.

First, Gemma 4 had worse pointwise fit than Gemma 3 but higher relational recovery. Second, Qwen3 improved raw NLL and JSD over Qwen 2.5 without materially improving raw relational recovery. Third, Qwen 2.5 retained the strongest calibrated relational recovery even though Qwen3 had better raw pointwise metrics.

These patterns reject a simple one-dimensional account in which improved probabilistic fit automatically implies improved disagreement geometry.

### 7.2 Calibration effects
Gemma 3 and Gemma 4 showed dramatic pointwise improvements under temperature scaling but little detectable relational change.

For Gemma 4, excess NLL above the empirical human entropy decreased from $5.2370-0.6543=4.5827$ to $0.9549-0.6543=0.3006$, closing 93.44% of excess NLL. Yet relational recovery changed by only $+0.21$ percentage points, with an interval spanning zero.

Qwen 2.5 behaved differently. Calibration reduced NLL from 5.2986 to 0.8837 and increased relational recovery by $+3.01$ points, 95% CI $[+1.84, +4.17]$.

Qwen3’s point estimate was positive but less certain: $+1.04$ points, 95% CI $[-0.03, +2.15]$.

Thus Qwen 2.5 was the only model whose marginal 95% interval clearly excluded zero.

### 7.3 Generational and family-by-generation comparisons
Gemma 4 improved over Gemma 3 by $\Delta R_{\mathrm{raw}}=+1.66$ points and $\Delta R_{\mathrm{cal}}=+1.83$ points, with a paired calibrated 95% CI of $[+0.52, +3.47]$.

By contrast, Qwen3 and Qwen 2.5 had nearly identical raw relational recovery ($11.92\%$ vs $11.85\%$). After calibration, Qwen3 was lower ($12.96\%$ vs $14.86\%$). The calibrated Qwen3-minus-Qwen2.5 contrast was $-1.90$ points, 95% CI $[-3.41, -0.16]$.

The family-by-generation interactions were:
$$\Delta_{\text{raw inter}} = -1.59\ \text{points},\qquad 95\%\ \mathrm{CI}=[-3.95, +0.75],$$
$$\Delta_{\text{cal inter}} = -3.73\ \text{points},\qquad 95\%\ \mathrm{CI}=[-5.94, -1.50].$$

The calibrated interaction excludes zero, indicating that calibrated generational change differed across the two family comparisons in this panel.

The change in calibration response across generations was $-2.14$ points, 95% CI $[-4.41, +0.12]$. This interval includes zero. The data therefore do not clearly establish that the evolution of the calibration effect itself differed by family.

### 7.4 Mechanism of calibration response

#### Qwen 2.5
Qwen 2.5’s increase in posterior-support overlap was distributed across support bands as follows:

| Human support ($S_{ij}$) | Contribution to $\Delta Q_{\mathrm{support}}$ | Share |
|---|---|---|
| $S<0.05$ | +0.000396 | 5.3% |
| $0.05\le S<0.25$ | +0.003693 | 49.4% |
| $0.25\le S<0.50$ | +0.002069 | 27.7% |
| $S\ge0.50$ | +0.001313 | 17.6% |

The gain was therefore driven mostly by redistribution of nearest-neighbor weight toward weakly and moderately supported human relationships, not only by removing false bridges.

#### Qwen3
The held-out-fold decomposition for Qwen3 was:

| Human support ($S_{ij}$) | Contribution to $\Delta Q_{\mathrm{support}}$ | Share |
|---|---|---|
| $S<0.05$ | -0.000028 | -1.1% |
| $0.05\le S<0.25$ | +0.001334 | 52.5% |
| $0.25\le S<0.50$ | +0.000077 | 3.0% |
| $S\ge0.50$ | +0.001158 | 45.6% |

Qwen3’s smaller shift was split primarily between weak and strongly supported relationships. Compared with Qwen 2.5, far less of the positive change came from the moderate-support band.

These decompositions concern changes in $Q_{\mathrm{support}}$, not a complete additive decomposition of $R_{\mathrm{norm}}$, because the analytic null also changes slightly.

### 7.5 Model complementarity
In the two-model Gemma 3–Qwen 2.5 analysis, more than 94% of thresholded high-support edges recovered by either model were unique to one model across multiple support thresholds. A separate threshold-free weighted analysis found that approximately 90.1% of Gemma’s captured human-support mass and 93.1% of Qwen’s occurred on edges absent from the other model’s graph.

The three-model edge atlas further indicated substantial unique regions:
- Qwen 2.5 only: 1,217 edges;
- Gemma 4 only: 1,025 edges;
- Gemma 3 only: 934 edges;
- all three shared: 29 edges.

These are descriptive edge counts under a selected support threshold and should not be interpreted as population-level prevalence estimates. They nevertheless motivated the coalition analysis.

### 7.6 Equal-weight opinion pools

#### Gemma 3 + Qwen 2.5
The equal-weight linear pool substantially improved pointwise fit ($\mathrm{NLL}_{\mathrm{raw}}=2.3800$). Relative to the human target entropy, this closed 45.6% of Gemma 3’s excess NLL and 62.8% of Qwen 2.5’s.

However, its calibrated relational recovery was $14.50\%$, with a paired difference from Qwen 2.5 of $-0.36$ points, 95% CI $[-1.78, +1.09]$. The equal-weight logarithmic pool reached $15.13\%$, but its difference from Qwen 2.5 was uncertain ($+0.27$ points, 95% CI $[-1.27, +1.66]$).

Thus simple pooling improved pointwise fit without reliably improving relational recovery over the best individual model.

#### Gemma 4 + Qwen 2.5
The cross-family successor–predecessor pair was more complementary.

The equal-weight calibrated linear pool reached $R_{\mathrm{cal}}=17.16\%$, with a paired difference from Qwen 2.5 of $+2.30$ points, 95% CI $[+0.87, +3.77]$.

The logarithmic pool was stronger under both censoring conventions:
- Under the primary (-40) convention: $R_{\mathrm{cal}}=18.1849\%$, $\Delta R_{\mathrm{vs\ Q2.5}}=+3.3217$ points, 95% CI $[+1.8938, +4.7291]$. Relative to Qwen 2.5’s Bound A baseline, the point difference is $+3.32$ points.
- Under the fixed (-20) stress test: $R_{\mathrm{cal}}=17.8689\%$. The bound-matched Qwen 2.5 baseline was 14.0887%, giving $\Delta R_{\mathrm{vs\ Q2.5}}=+3.7803$ points, 95% CI $[+2.4416, +5.1161]$.

The pool’s absolute relational score shifted by 0.316 points between censoring conventions, but the paired advantage remained positive and its 95% percentile-bootstrap interval excluded zero.

#### Four-model grand pool
The four-model logarithmic pool reached $17.2522\%$ under (-40), with $\Delta R_{\mathrm{vs\ Q2.5}}=+2.3890$, 95% CI $[+0.8812, +3.9081]$.

Under (-20), it reached $16.7756\%$, while the bound-matched difference was $+2.6869$, 95% CI $[+1.1065, +4.3085]$. The absolute score shifted by 0.477 points.

The coalition census explored many subsets and two pooling operators on the same pilot. These results are therefore exploratory even when paired intervals exclude zero. Independent validation or nested coalition selection is required before claiming that one coalition is optimal.

### 7.7 Mechanism of the Gemma 4 + Qwen 2.5 linear coalition
The calibrated fold-coherent linear pool was compared with calibrated Qwen 2.5. Changes in posterior-support overlap were partitioned into five mutually exclusive categories.

| Category | Net $\Delta Q_{\mathrm{support}}$ | Share of positive gain | Share of net gain |
|---|---|---|---|
| **Pool-created relationships** | +0.033110 | 78.3% | 582.8% |
| **Gemma 4-only baseline relationships** | +0.006215 | 14.7% | 109.4% |
| **Lowest-support relationships ($S_{ij}<0.05$)** | +0.000273 | 7.0% | 4.8% |
| **Shared baseline relationships** | -0.000106 | 0.0% | -1.9% |
| **Qwen-only baseline relationships** | -0.033811 | 0.0% | -595.1% |
| **Total** | **+0.005681** | **100.0%** | **100.0%** |

Shares of net gain can exceed 100% in magnitude because large positive and negative components cancel. The positive-gain decomposition shows that pool-created relationships accounted for 78.3% of positive gain, Gemma 4-only relationships accounted for 14.7%, and lowest-support relationships accounted for 7.0%.

The coalition did not merely preserve both input graphs. It formed new neighborhoods while pruning a large set of Qwen-only relationships.

---

## 8. Discussion

### 8.1 Four distinct axes
The results support a four-axis view of probabilistic language-model behavior:
1. **Pointwise fidelity:** How close is each model distribution to the corresponding human distribution?
2. **Relational recovery:** Does the model preserve which items share similar disagreement patterns?
3. **Calibration response:** Does global probability rescaling reorganize the model’s neighborhood structure?
4. **Complementarity:** Do different models capture different human-supported relationships that can be combined?

These axes were empirically dissociated. Gemma 4 improved relational recovery while worsening pointwise fit. Qwen3 improved raw pointwise fit without improving raw relational recovery. Qwen 2.5 showed the largest relational calibration response. Cross-family pools exceeded all individual models in exploratory analyses.

### 8.2 Temperature scaling can expose—but does not universally create—relational structure
Temperature scaling applies a global radial transformation in log-ratio space. Such a transformation does not add new item-specific evidence. It can, however, alter relative distances enough to change neighborhood identities.

The two Gemma models showed substantial pointwise calibration gains with little relational change. Qwen 2.5 showed a larger and clearly positive relational change. Qwen3 occupied an intermediate position. The support-band analyses suggest that useful relational changes arise when calibration redistributes neighborhood weight toward human-supported edges, not simply when it reduces confidence.

This distinction complicates the common intuition that calibration is topology-preserving. It may be nearly topology-invariant for some score geometries but not others.

### 8.3 Newer models are not uniformly better
The 2 × 2 panel demonstrates that generational improvement depends on the metric:
- Gemma 4 improved relational recovery but worsened pointwise fit.
- Qwen3 improved raw pointwise fit but did not improve raw relational recovery.
- Qwen3’s calibrated relational recovery was lower than Qwen 2.5’s.

The calibrated family-by-generation interaction reflects that divergence. It should not be interpreted as a universal family effect: only one model pair per family was tested, and model generations differ in architecture, training data, post-training, multimodal design, and runtime packaging.

### 8.4 Complementarity can matter more than generation
Gemma 4 and Qwen 2.5 produced the strongest coalition. The mechanism decomposition indicates that its gain came primarily from relationships created by pooling and relationships unique to Gemma 4, while many Qwen-only relationships were removed.

This resembles the motivation behind multi-model pluralism: different models may encode nonidentical slices of a target distribution, and collaboration can cover regions a single model misses (Feng et al., 2024; Feng et al., 2026). The present finding is narrower. These are general-purpose model packages, not community-specific models, and the task is NLI ambiguity rather than social values. Still, it provides a geometric example of model complementarity in a densely annotated distributional benchmark.

### 8.5 Implications for pluralistic evaluation
Pluralistic-alignment research calls for systems that represent diverse human perspectives rather than only an average preference (Sorensen et al., 2024). A pointwise distributional metric is necessary but may be insufficient. Two models with similar aggregate NLL can differ in which cases they regard as analogous.

Relational evaluation could support:
- retrieval of comparable ambiguous cases;
- discovery of model-specific disagreement clusters;
- diagnosis of underrepresented judgment patterns;
- selection of complementary model coalitions;
- evaluation of whether post-training compresses or reorganizes disagreement structure.

However, ChaosNLI disagreement should not be equated with normative pluralism. The dataset captures collective judgments about textual inference, and annotator distributions may reflect ambiguity, misunderstanding, convention, or noise alongside legitimate interpretive variation.

---

## 9. Limitations

### 9.1 Pilot sample
The analysis uses 600 items from one benchmark. All confidence intervals quantify resampling uncertainty within this pilot, not generalization to new ChaosNLI items, new domains, or new populations.

### 9.2 No independent confirmation
Several hypotheses and coalition choices emerged during iterative pilot analysis. Even though the final estimators are audited, the strongest coalition findings remain exploratory until tested on a frozen holdout sample.

### 9.3 Runtime-specific model packages
All models are quantized Ollama packages evaluated through one endpoint and one inference contract. Results may change with different quantization, another inference runtime, different chat templates, base rather than instruction-tuned checkpoints, explicit reasoning mode, or another candidate-logprob implementation.

### 9.4 First-token elicitation
The experiment assumes first-token probabilities over three symbols can represent a model’s NLI judgment distribution. This avoids verbalized-confidence artifacts but may not capture distributions over final answers after reasoning.

### 9.5 Censored candidates
Candidates outside the top-20 list require an imputation convention. Sensitivity analyses bound some effects, but the exact unreturned probability is unknown.

### 9.6 Novel relational metric
$R_{\mathrm{norm}}$ is an internal evaluation construct. Its mathematical implementation is frozen and regression-tested, but its external validity, task transfer, and relationship to downstream pluralistic behavior require further study.

### 9.7 Prototype-equivalent interpretation
$K_{\mathrm{eff}}$ is an interpolated benchmark-equivalent resolution. It is not a count of latent neural states, concepts, or personas.

### 9.8 Coalition multiplicity
The full coalition census searched many subsets and pooling operators. Reported coalition intervals compare a specified pool with Qwen 2.5 but do not correct selection of the best pool from the entire census.

### 9.9 Human target limitations
Crowdsourced judgment frequencies do not automatically define a normatively desirable target. Human disagreement can contain bias, inattention, or systematic misunderstanding. Distributional matching is an empirical objective, not a complete alignment principle.

---

## 10. Reproducibility

### 10.1 Frozen commits
- **Input four-model panel:** `055c3663529a0c0d1b3f840d3bfe15beca8a443e`
- **Analytical synthesis:** `d73fd1997b43670a1ec2264bff68193a9a521bb7`
- **Pilot freeze:** `5cc25272ae04f5eb983b8f9c5e3db7e740dd4b87`

### 10.2 Principal artifacts
- `research/chaosnli/results/E004_paper_ready_authoritative_synthesis_summary.json`
- `research/chaosnli/results/E004_four_model_panel_summary.json`
- `research/chaosnli/results/E004_gemma4_12b_summary.json`
- `research/chaosnli/results/E004_qwen2.5_14b_summary.json`
- `research/chaosnli/results/E004_qwen3_14b_summary.json`
- `research/chaosnli/lab/analyze_llm_lpe.py`
- `research/chaosnli/lab/analyze_four_model_panel.py`
- `research/chaosnli/lab/e004_paper_ready_authoritative_synthesis.py`

### 10.3 Frozen hashes
- **System prompt SHA-256:** `f7a62741d9edce4c18610f3249e5e70ef798794ea8adf3f113361249c660e0e1`
- **Pilot manifest SHA-256:** `318c67354242771d2812b0d26dbd3d89e084ecc0afece250cae453d6531f4851`
- **Support matrix SHA-256:** `29ff35cfddbd1a4c30858eb2feac58aeb0c087aff99850736007a2a49fb40245`

### 10.4 Model digests
- `gemma3:12b`: `f4031aab637d7a124037599427b5e40e6988894223292419c8f0f08a5cb7a321`
- `gemma4:12b`: `4eb23ef187e27301c3df631b402e118939c3683a48e89f71c4c34a919a32c256`
- `qwen2.5:14b`: `7cdf5a0187d5a528e1d5a7fb489069d31b5c46440f3531b790d5145b23d90218`
- `qwen3:14b`: `a8cc1361f314c4495c64c741e5ab37d6e66504a37e199f1816e87f3b49520448`

---

## 11. Recommended figures

- **Figure 1 — Pointwise–relational Pareto map:** Plot calibrated and raw model conditions with $x=\mathrm{NLL}$ and $y=R_{\mathrm{norm}}$. Connect each model’s raw and calibrated conditions with an arrow. Expected message: calibration moves every model strongly leftward in NLL, but vertical relational movement differs substantially.
- **Figure 2 — Calibration gain with 10,000-draw intervals:** A four-model dot-and-whisker plot of within-model $R_{\mathrm{cal}}-R_{\mathrm{raw}}$. Expected message: Qwen 2.5 is the only model with a clearly positive interval.
- **Figure 3 — 2 × 2 family-by-generation panel:** Show raw and calibrated $R$ for earlier and successor models in each family. Expected message: Gemma improves across generations after calibration, while Qwen declines.
- **Figure 4 — Support-band mechanism:** Stacked bars for Qwen 2.5 and Qwen3 showing contributions to $\Delta Q_{\mathrm{support}}$ from the four support bands.
- **Figure 5 — Coalition mechanism:** A diverging contribution chart for the five exhaustive edge categories in the Gemma 4 + Qwen 2.5 linear pool.
- **Figure 6 — Censoring sensitivity:** Plot logarithmic-pool $R$ and paired contrasts under the (-40) and (-20) conventions.

---

## 12. Future work

The highest-priority next experiment is an independent validation sample rather than another model. A practical confirmatory design would use 7,200 requests.

Pre-register a small set of primary hypotheses:
1. Gemma 4 exceeds Gemma 3 in raw relational recovery.
2. Qwen3 and Qwen 2.5 have similar raw relational recovery.
3. Qwen 2.5’s calibration gain exceeds Qwen3’s.
4. The calibrated family-by-generation interaction is negative.
5. The Gemma 4 + Qwen 2.5 linear pool exceeds Qwen 2.5.
6. The Gemma 4 + Qwen 2.5 logarithmic pool exceeds Qwen 2.5 under both censoring conventions.

The validation analysis should:
- freeze all code and hypotheses before inference;
- use simultaneous or multiplicity-adjusted intervals for the primary family;
- avoid selecting coalitions on the validation data;
- test transfer across SNLI and MNLI separately;
- include at least one out-of-domain distributional benchmark if feasible.

Longer-term work should evaluate whether relational recovery predicts useful behavior in retrieval, deliberation, uncertainty communication, and pluralistic decision support.

---

## 13. Conclusion

This pilot study shows that a language model’s relationship to human disagreement cannot be summarized by one calibration score.

Across four matched open-weight model packages:
- pointwise fit and relational recovery separated;
- successor models improved different dimensions;
- scalar calibration reorganized some model geometries but left others nearly unchanged;
- Qwen 2.5 showed the clearest relational calibration gain;
- cross-family pools produced the strongest exploratory relational estimates;
- pooled gains arose mainly from newly formed and model-complementary relationships.

The central methodological implication is that distributional evaluation should examine not only whether a model matches each human judgment distribution, but also whether it preserves the structure connecting disagreement patterns across cases.

The central scientific implication is equally direct: better calibration does not guarantee better relational pluralism, newer models are not uniformly better, and diversity across models can preserve human disagreement structure that no single model recovers alone.

---

## Appendix A. Expanded single-model metrics

| Model | Condition | NLL | Brier | JSD nats | $Q_{\mathrm{support}}$ | $Q_{\mathrm{null}}$ | $R_{\mathrm{norm}}$ | Bits | $K_{\mathrm{eff}}$ |
|---|---|---|---|---|---|---|---|---|---|
| Gemma 3 | Raw | 3.827706 | 0.388863 | 0.172119 | 0.040773 | 0.016804 | 9.720955% | 1.370 | 2.585 |
| Gemma 3 | Calibrated | 0.930831 | 0.154746 | 0.073776 | 0.040880 | 0.016825 | 9.756399% | 1.374 | 2.592 |
| Gemma 4 | Raw | 5.236978 | 0.423543 | 0.182206 | 0.044852 | 0.016787 | 11.381023% | 1.529 | 2.886 |
| Gemma 4 | Calibrated | 0.954852 | 0.169814 | 0.079859 | 0.045349 | 0.016774 | 11.587246% | 1.549 | 2.925 |
| Qwen 2.5 | Raw | 5.298609 | 0.382125 | 0.164622 | 0.045946 | 0.016704 | 11.854417% | 1.574 | 2.977 |
| Qwen 2.5 | Calibrated | 0.883675 | 0.137871 | 0.060487 | 0.053417 | 0.016762 | 14.863218% | 1.816 | 3.521 |
| Qwen3 | Raw | 4.500827 | 0.336620 | 0.150665 | 0.046193 | 0.016792 | 11.923090% | 1.581 | 2.991 |
| Qwen3 | Calibrated | 0.886497 | 0.129706 | 0.060916 | 0.048735 | 0.016769 | 12.961990% | 1.664 | 3.169 |

---

## Appendix B. Fitted temperatures

| Model | Fold temperatures | Mean |
|---|---|---|
| **Gemma 3** | 10.5526, 10.0777, 10.3081, 10.3467, 10.5092 | 10.3589 |
| **Gemma 4** | 14.9178, 14.8849, 14.8856, 14.4980, 14.8106 | 14.7994 |
| **Qwen 2.5** | 17.2041, 17.0689, 17.3332, 17.2896, 17.0594 | 17.1910 |
| **Qwen3** | 14.0642, 13.7048, 14.0774, 13.8922, 13.8254 | 13.9128 |

---

## Appendix C. Claim boundaries

### Supported by the pilot
- The four evaluated model packages differ in pointwise fidelity, relational recovery, and calibration response.
- Gemma 4 exceeds Gemma 3 in relational recovery on this pilot.
- Qwen3 improves raw pointwise fit over Qwen 2.5 but not raw relational recovery.
- Qwen 2.5 has a clearly positive marginal bootstrap interval for relational calibration gain.
- The calibrated family-by-generation interaction is negative in this panel.
- Cross-family pools have higher exploratory relational point estimates than individual models.
- The Gemma 4 + Qwen 2.5 linear pool gains support mainly through pool-created and Gemma 4-unique relationships.

### Not established
- A universal property of all Gemma or Qwen models.
- A causal architectural explanation for any family difference.
- Generalization beyond the 600-item ChaosNLI pilot.
- That the selected coalition is optimal on new data.
- That prototype-equivalent resolution reflects literal internal prototypes.
- That matching crowd judgment distributions is always normatively desirable.
- That NLI disagreement is equivalent to value pluralism.

---

## References

- Baan, J., van der Goot, R., Plank, B., & Aziz, W. (2022). Stop Measuring Calibration on Incorrect Predictions: Calibration under Disagreement. *Proceedings of EMNLP 2022*, 8716–8730.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
- Feng, S., Potts, C., & Choi, Y. (2024). Modular Pluralism for Pluralistic Alignment. *Proceedings of NeurIPS 2024*.
- Feng, S., Qin, Y., & Choi, Y. (2026). Multi-LLM Collaboration: Principles, Architectures, and Pluralistic Coverage. *arXiv preprint arXiv:2601.08900*.
- Genest, C., & Zidek, J. V. (1986). Combining Probability Distributions: A Review and an Annotated Bibliography. *Statistical Science*, 1(1), 114–135.
- Gemma Team, Google DeepMind. (2025). *Gemma 3 Technical Report*. Google DeepMind.
- Gemma Team, Google DeepMind. (2026). *Gemma 4 Technical Report*. Google DeepMind.
- Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of ICML 2017*, 1321–1330.
- Jiang, Z., Xu, F. F., Araki, J., & Neubig, G. (2021). How Can We Know What Language Models Know? *Transactions of the Association for Computational Linguistics*, 8, 423–438.
- Lin, J. (1991). Divergence measures based on the Shannon entropy. *IEEE Transactions on Information Theory*, 37(1), 145–151.
- Nie, Y., Zhou, X., & Bansal, M. (2020). What Can We Learn from Collective Human Opinions on Natural Language Inference? *Proceedings of EMNLP 2020*, 9131–9143.
- Pavlick, E., & Kwiatkowski, T. (2019). Inherent Disagreement in Human Textual Inferences. *Proceedings of ACL 2019*, 4777–4787.
- Peterson, J. C., Battleday, R. M., Griffiths, T. L., & Russakovsky, O. (2019). Human uncertainty makes classification more robust. *Proceedings of ICCV 2019*, 9617–9626.
- Qwen Team. (2024). *Qwen2.5 Technical Report*. Alibaba Group.
- Qwen Team. (2026). *Qwen3 Technical Report: Integrating Reasoning and Instruction Dynamics*. Alibaba Group.
- Sorensen, T., Moore, J., Horowitz, J., Liu, R., Shaw, A., Gordon, M., Halevy, A., & Choi, Y. (2024). A Roadmap for Pluralistic Alignment. *arXiv preprint arXiv:2402.05070*.
- Wang, J., Wang, Y., Chen, N., & Zhou, X. (2024). Evaluating Language Models on Collective Human Uncertainty and Opinion Distribution. *Proceedings of ACL 2024*, 5120–5135.
