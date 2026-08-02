# ChaosNLI Literature and Mathematical Foundations

## 1. Research landscape

### 1.1 Human disagreement is not one gold label plus noise

ChaosNLI was introduced to evaluate distributions over collective human judgments rather than only majority-label accuracy. It contains 100 annotations per example for 1,514 SNLI items, 1,599 MNLI items, and 1,532 abductive-NLI items. The original work found substantial disagreement and showed that contemporary NLI models did not adequately recover human label distributions.

Earlier work on inherent disagreement in textual inference found that disagreement can persist across annotator samples and additional context. A model softmax distribution should therefore not automatically be interpreted as a model of human disagreement.

**Implication:** do not treat entropy as annotation noise or model softmax as an equivalent uncertainty source.

### 1.2 Predicting human distributions remains open

Distributed NLI evaluated MC dropout, deep ensembles, recalibration, and distribution distillation. These improved distribution recovery but remained below estimated human performance.

Calibration-focused work reported that label smoothing and temperature scaling improve model–human divergence measures. This creates a strong Shadowspace question:

> Does calibration improve only pointwise distribution similarity, or does it recover the human relational topology among examples?

A calibrated model may improve mean JSD while still replacing human neighbors with different examples.

### 1.3 Disagreement has heterogeneous causes

The Jiang–de Marneffe taxonomy has ten categories in three groups.

**Uncertainty in sentence meaning**

1. Lexical
2. Implicature
3. Presupposition
4. Probabilistic enrichment
5. Imperfection

**Underspecification in guidelines**

6. Coreference
7. Temporal reference
8. Interrogative hypothesis

**Annotator behavior**

9. Accommodating minimally added content
10. High overlap

Use this taxonomy as external validation, not as a complete ontology and not as something inferred from geometry without validation.

### 1.4 Valid variation must be separated from error

VariErr reannotated 500 ChaosNLI-MNLI items with explanations and validity judgments. Its central distinction is:

- **human label variation:** different labels supported by valid reasons;
- **annotation error:** labels assigned for invalid reasons.

A high-entropy item can contain valid plural interpretations, errors, or both. A geometric analysis that ignores validity can incorrectly call annotation error “human ambiguity.”

### 1.5 Labels do not uniquely identify reasoning

LiveNLI and LiTEx emphasize within-label variation: annotators can select the same NLI label for different reasons. ACL 2026 work further reports cases where annotators disagree on labels while using similar explanations.

Opinion geometry cannot, by itself, identify the reason for disagreement. Compare at least three relational views:

1. human label-distribution geometry;
2. textual semantic similarity;
3. explanation/taxonomy similarity.

### 1.6 Recent formal-semantic results constrain the hypothesis space

A July 2026 preprint using all 3,113 ChaosNLI-S/M items found a group-level relationship between monotonicity profiles and entropy, but the formal profiles explained only a small portion of item-level entropy variation.

**Implication:** do not make “predict entropy from simple linguistic features” the primary novelty. Local relational structure, explanation validity, and model topology are more promising.

---

## 2. Three-class probability simplex

For item \(i\), human counts are

\[
\mathbf{x}_i=(x_{iE},x_{iN},x_{iC}),\qquad
x_{iE}+x_{iN}+x_{iC}=100.
\]

The empirical distribution is

\[
\hat{\mathbf{p}}_i =
\left(
\frac{x_{iE}}{100},
\frac{x_{iN}}{100},
\frac{x_{iC}}{100}
\right).
\]

Every distribution lies on the two-dimensional simplex

\[
\Delta^2 =
\{(p_E,p_N,p_C):p_j\ge0,\;p_E+p_N+p_C=1\}.
\]

### 2.1 Exact ternary coordinates

Choose vertices

\[
v_E=(0,0),\quad
v_N=(1,0),\quad
v_C=\left(\frac12,\frac{\sqrt3}{2}\right).
\]

The barycentric coordinate is

\[
T(\mathbf{p})=p_Ev_E+p_Nv_N+p_Cv_C,
\]

so

\[
x=p_N+\frac12p_C,\qquad
y=\frac{\sqrt3}{2}p_C.
\]

This is one-to-one on \(\Delta^2\). It is not dimensionality reduction: the three probabilities have only two free dimensions.

Euclidean distance in this equilateral ternary plane is proportional to Euclidean probability distance:

\[
\|T(\mathbf{p})-T(\mathbf{q})\|_2
=
\frac1{\sqrt2}\|\mathbf{p}-\mathbf{q}\|_2.
\]

It does **not** preserve Hellinger, Fisher–Rao, Jensen–Shannon, or Aitchison distances.

### 2.2 Recommended exact views

- human ternary distribution;
- model ternary distribution;
- human-to-model residual arrow;
- posterior uncertainty region;
- multiple models overlaid with identity;
- split view with stable source-object IDs.

The ternary view is the independent reference against which arbitrary projections should be checked.

---

## 3. Distances and divergences

Let \(\mathbf{p}\) and \(\mathbf{q}\) be class distributions.

### 3.1 Jensen–Shannon distance

\[
m=\frac{\mathbf{p}+\mathbf{q}}2,
\]

\[
JS(\mathbf{p},\mathbf{q})
=
\frac12 KL(\mathbf{p}\|m)+
\frac12 KL(\mathbf{q}\|m).
\]

With base-2 logs, \(JS\in[0,1]\). The square root \(d_{JS}=\sqrt{JS}\) is a metric.

**Use:**

- primary pointwise model–human alignment, for continuity with ChaosNLI;
- sensitivity neighborhood geometry;
- zero-safe because the mixture is positive wherever either input is positive.

### 3.2 Hellinger distance

\[
d_H(\mathbf{p},\mathbf{q})
=
\frac1{\sqrt2}
\|\sqrt{\mathbf{p}}-\sqrt{\mathbf{q}}\|_2.
\]

With Bhattacharyya coefficient

\[
BC(\mathbf{p},\mathbf{q})=\sum_j\sqrt{p_jq_j},
\]

\[
d_H=\sqrt{1-BC}.
\]

**Use:**

- primary neighborhood geometry;
- stable with exact zeros;
- Euclidean after the square-root map.

### 3.3 Fisher–Rao distance

Under Shadowspace’s factor-of-two convention,

\[
d_{FR}(\mathbf{p},\mathbf{q})
=
2\arccos BC(\mathbf{p},\mathbf{q}).
\]

Important: Hellinger and Fisher–Rao are strictly monotone functions of the same coefficient. Absent ties and numerical issues, they produce **identical neighbor rankings**. They should not count as two independent neighborhood sensitivity analyses.

They can differ in:

- absolute distance ratios;
- stress;
- geodesic interpretation;
- threshold-based rules.

### 3.4 Total variation

\[
d_{TV}(\mathbf{p},\mathbf{q})
=
\frac12\|\mathbf{p}-\mathbf{q}\|_1.
\]

Use as a simple interpretable sensitivity measure.

### 3.5 Euclidean probability distance

\[
d_E(\mathbf{p},\mathbf{q})=
\|\mathbf{p}-\mathbf{q}\|_2.
\]

Use because the ternary display preserves it up to fixed scale. Do not imply it is the unique natural geometry.

### 3.6 Aitchison distance

For positive compositions,

\[
clr(\mathbf{p})_j
=
\log p_j-\frac13\sum_\ell\log p_\ell,
\]

\[
d_A(\mathbf{p},\mathbf{q})
=
\|clr(\mathbf{p})-clr(\mathbf{q})\|_2.
\]

ChaosNLI has zero counts, so Aitchison requires a declared zero policy. Results can depend strongly on the replacement.

**Recommendation:**

- not primary;
- preregistered sensitivity only;
- report at least two replacement settings;
- flag edges that change across policies.

For three parts, ILR coordinates are preferable to CLR for unconstrained Euclidean statistical modeling because ILR has two orthonormal coordinates rather than three singular coordinates.

One possible basis:

\[
z_1=\frac1{\sqrt2}\log\frac{p_E}{p_N},
\]

\[
z_2=\frac1{\sqrt6}\log\frac{p_Ep_N}{p_C^2}.
\]

### 3.7 KL and cross-entropy

\[
KL(\mathbf{p}\|\mathbf{q})
=
\sum_jp_j\log\frac{p_j}{q_j}.
\]

KL is asymmetric and can be infinite if \(q_j=0\) where \(p_j>0\). It is useful for model scoring, not as an unnamed symmetric neighborhood metric.

For counts \(\mathbf{x}_i\) and model distribution \(\mathbf{q}_i\), the multinomial log score is proportional to

\[
\sum_jx_{ij}\log q_{ij}.
\]

This uses all 100 judgments and is a proper predictive score.

---

## 4. Finite-annotation uncertainty

The 100-vote distribution is an estimate.

\[
\mathbf{x}_i\mid\boldsymbol{\theta}_i
\sim Multinomial(100,\boldsymbol{\theta}_i).
\]

With prior

\[
\boldsymbol{\theta}_i\sim Dirichlet(\boldsymbol{\alpha}),
\]

the posterior is

\[
\boldsymbol{\theta}_i\mid\mathbf{x}_i
\sim Dirichlet(\mathbf{x}_i+\boldsymbol{\alpha}).
\]

### 4.1 Priors

Primary:

\[
\boldsymbol{\alpha}=(1/2,1/2,1/2).
\]

Sensitivity:

\[
\boldsymbol{\alpha}=(1,1,1).
\]

Do not choose after inspecting which strengthens a result.

### 4.2 Posterior quantities

Use Monte Carlo draws for:

- ternary credible regions;
- entropy intervals;
- probability each label is the majority;
- model–human distance distribution;
- probability a human neighbor edge exists;
- probability a model is under- or over-dispersed;
- probability of unsupported model mass.

### 4.3 Edge support

For draw \(b\), build

\[
N_H^{(b)}(i;k).
\]

Directed edge support is

\[
s_{ij}
=
\frac1B\sum_{b=1}^B
\mathbf{1}[j\in N_H^{(b)}(i;k)].
\]

Suggested display conventions:

- supported: \(s_{ij}\ge0.90\);
- uncertain: \(0.10<s_{ij}<0.90\);
- unsupported: \(s_{ij}\le0.10\).

Store the continuous value; thresholds are not universal facts.

### 4.4 Human split-half reliability

Repeat:

1. split 100 labels into two sets of 50;
2. estimate distributions;
3. construct both neighbor graphs;
4. compare them.

This estimates recoverable topology at available annotation depth. A model overlap of 0.70 means something different if human split-half overlap is 0.72 versus 0.95.

---

## 5. Comparing relational spaces

For item \(i\):

- \(N_H(i;k)\): human-opinion neighbors;
- \(N_M(i;k)\): model-distribution neighbors;
- \(N_T(i;k)\): text-embedding neighbors;
- \(N_R(i;k)\): reasoning/taxonomy neighbors where available.

### 5.1 Local overlap

\[
O_i(k)=\frac{|N_H(i;k)\cap N_M(i;k)|}{k}.
\]

### 5.2 Jaccard

\[
J_i(k)
=
\frac{|N_H(i;k)\cap N_M(i;k)|}
{|N_H(i;k)\cup N_M(i;k)|}.
\]

### 5.3 Global preservation & Fractional Tie-Aware Neighborhoods

Under discrete 3-class human distributions, distance ties at the $k$-th boundary distance are frequent. For focal item $i$, let $A_i$ be the set of points strictly closer than the $k$-th distance, and $B_i$ be the set of points tied at the $k$-th distance boundary. Let $r_i = k - |A_i|$.

Define fractional tie-aware weights:
\[
w_{ij} = \begin{cases} 1 & \text{if } j \in A_i \\ \frac{r_i}{|B_i|} & \text{if } j \in B_i \\ 0 & \text{otherwise} \end{cases}
\]

The soft overlap is:
\[
O_i^{\mathrm{soft}}(k) = \frac{1}{k} \sum_j \min(w_{ij}^H, w_{ij}^M).
\]

Global tie-aware preservation is:
\[
Q_{NX}^{\mathrm{soft}}(k) = \frac{1}{N} \sum_{i=1}^N O_i^{\mathrm{soft}}(k).
\]

Chance overlap is $Q_{\mathrm{chance}} = \frac{k}{N-1}$. Excess-over-chance ratio comparing model to human reliability is:
\[
\text{Excess Ratio} = \frac{Q_{\mathrm{model}} - Q_{\mathrm{chance}}}{Q_{\mathrm{human}} - Q_{\mathrm{chance}}}.
\]

### 5.4 Two-Level Graph Representation

- **Level 1 (Opinion-Profile Graph)**: Graph constructed over unique count vectors ($1,604$ unique nodes for ChaosNLI-S/M), weighted by item frequency. Evaluates opinion geometry without duplicate-item tie artifacts.
- **Level 2 (Items within Profiles)**: Evaluates text embedding, taxonomy, and prediction dispersion among items sharing identical opinion profiles.

### 5.5 Model edge consensus

For models \(m=1,\dots,M\),

\[
c_{ij}
=
\frac1M\sum_m
\mathbf{1}[j\in N_m(i;k)].
\]

Compare \(c_{ij}\) with human support \(s_{ij}\):

- high/high: shared relation;
- high consensus/low support: model-consensus failure;
- low consensus/high support: human relation missed by most models;
- intermediate: unresolved/model-specific.

---

## 6. Mismatch constructs

Define before using in UI.

### 6.1 Majority reversal

\[
\arg\max_jq_{ij}\ne\arg\max_j\hat p_{ij}.
\]

### 6.2 Uncertainty collapse

Model entropy is below human entropy for most posterior draws.

### 6.3 Spurious uncertainty

Model entropy is above human entropy for most posterior draws.

### 6.4 Unsupported label mass

Example preregistered rule:

\[
q_{ij}\ge0.25
\quad\text{and}\quad
P(\theta_{ij}<0.10\mid\mathbf{x}_i)\ge0.95.
\]

Use continuous measures for primary analysis; threshold labels help case selection.

### 6.5 Correct majority, wrong shape

Argmax agrees but JSD is large. This is central to ChaosNLI and should not be counted as ordinary success.

### 6.6 Neighborhood substitution

The model preserves a similar number of close neighbors but replaces human-supported edges with model-only edges.

---

## 7. Keep three reliability questions separate

1. **Human–model mismatch**  
   Difference between human and model distribution spaces.

2. **Metric/representation dependence**  
   Difference from Hellinger, JSD, Euclidean, or smoothed Aitchison.

3. **Projection distortion**  
   Difference between source neighbors and displayed-view neighbors.

For raw three-class distributions, the ternary view is exact. Every diagnostic should state:

- source representation;
- source metric;
- display coordinates;
- whether display distance is analytically meaningful;
- \(k\);
- posterior treatment;
- model/calibration version.

---

## 8. Construct-validity warnings

- Entropy measures amount, not cause, of disagreement.
- Softmax uncertainty is not automatically human uncertainty.
- Calibration is not identical to human-opinion recovery.
- Similar distributions can arise from different reasoning.
- Different labels can arise from similar reasoning.
- Taxonomy categories are heterogeneous and potentially multi-label.
- Human counts can contain valid variation and error.
- Text embeddings are model-derived, not ground-truth semantics.
- A stable model relation can be unsupported by humans.
- Projection stability cannot establish a causal linguistic explanation.
- High graph overlap does not imply identical calibration.
- Low graph overlap can reflect finite human sampling.
- ChaosNLI-S/M is a selected evaluation population; condition claims on scope.

---

## 9. Key references

1. Nie, Y., Zhou, X., & Bansal, M. (2020). *What Can We Learn from Collective Human Opinions on Natural Language Inference Data?* EMNLP. DOI: 10.18653/v1/2020.emnlp-main.734.
2. Pavlick, E., & Kwiatkowski, T. (2019). *Inherent Disagreements in Human Textual Inferences.* TACL 7, 677–694. DOI: 10.1162/tacl_a_00293.
3. Zhou, X., Nie, Y., & Bansal, M. (2022). *Distributed NLI: Learning to Predict Human Opinion Distributions for Language Reasoning.* Findings of ACL. DOI: 10.18653/v1/2022.findings-acl.79.
4. Wang, Y. et al. (2022). *Capture Human Disagreement Distributions by Calibrated Networks for Natural Language Inference.* Findings of ACL. DOI: 10.18653/v1/2022.findings-acl.120.
5. Jiang, N.-J., & de Marneffe, M.-C. (2022). *Investigating Reasons for Disagreement in Natural Language Inference.* TACL 10, 1357–1374. DOI: 10.1162/tacl_a_00523.
6. Weber-Genzel, L. et al. (2024). *VariErr NLI: Separating Annotation Error from Human Label Variation.* ACL. DOI: 10.18653/v1/2024.acl-long.123.
7. Jiang, N.-J., Tan, C., & de Marneffe, M.-C. (2023). *Ecologically Valid Explanations for Label Variation in NLI.* arXiv:2310.13850.
8. Hong, P. et al. (2025). *LiTEx: A Linguistic Taxonomy of Explanations for Understanding Within-Label Variation in NLI.* EMNLP. DOI: 10.18653/v1/2025.emnlp-main.1728.
9. Hong, P. et al. (2026). *Agree, Disagree, Explain.* Findings of ACL. DOI: 10.18653/v1/2026.findings-acl.1342.
10. Lee, N., An, N. M., & Thorne, J. (2023). *Can Large Language Models Capture Dissenting Human Voices?* EMNLP. DOI: 10.18653/v1/2023.emnlp-main.278.
11. Lee, J. A., & Verleysen, M. (2009). *Quality assessment of dimensionality reduction: Rank-based criteria.* Neurocomputing 72, 1431–1443.
12. Aitchison, J. (1983). *Principal component analysis of compositional data.* Biometrika 70(1), 57–65.
13. Choi, H. (2026). *How Much Human Label Variation Does Formal Semantic Structure Explain?* arXiv:2607.15870. **Recent preprint; not peer reviewed at package creation.**
