# The Geometry Models Impose on Human Disagreement

## Close-of-Night Research Synthesis and Open-Thread Memorandum

**Date:** August 3, 2026  
**Repository:** `admiralorbiter/shadowspace`  
**Core branch:** `chaosnli`  
**Exploratory branch:** `research/geometry-sandbox`  
**Latest audited exploratory commit:** `abf81ec97822ffd0862694bbe3814faa6ed50fb1`

---

## 1. Executive synthesis

This research program begins from a simple observation: a distribution of human labels contains more information than a majority label, but even a label distribution does not fully describe how human judgments are organized across cases.

The project therefore treats human disagreement as a **relational geometry**.

Each ChaosNLI item is represented by a three-class human judgment distribution. Posterior draws over those distributions define uncertainty-aware neighborhoods among items. A model is evaluated not only on whether its probability vector matches the human vector for each item, but also on whether it places items near the same cases that humans place them near.

The emerging central claim is:

> Human disagreement forms a compressible but nontrivial relational geometry. Individual classifiers recover only low-resolution versions of it. Calibration can substantially improve a selected pointwise probability score while leaving the model’s ambiguity direction and relational organization almost unchanged. Diverse model ensembles recover complementary structure and increase prototype-equivalent relational resolution.

The project currently contains four different evidentiary levels:

| Status | Meaning |
|---|---|
| **Established theorem** | A mathematical identity independent of the dataset, subject to stated definitions. |
| **Frozen / publication-grade result** | Coherent out-of-fold evaluation with appropriate nulls, uncertainty, and provenance. |
| **Audited pilot** | Methodologically serious result on a pilot or exploratory subset; not yet full confirmatory evidence. |
| **Exploratory signal** | Interesting result whose estimator, uncertainty, or interpretation still needs refinement. |
| **Pending redesign / invalid current artifact** | Current output should not be used as evidence for its advertised claim. |

The most defensible state of the work tonight is:

1. **The Calibration Ray Theorem is established.**
2. **E002 is publication-grade evidence that temperature scaling improves NLL far more than relational recovery.**
3. **E007 and E008 provide audited pilot evidence that ensembles recover complementary relational structure and roughly double prototype-equivalent state resolution relative to the best single classifier.**
4. **E014–E018 produce promising evidence of boundary collapse, angular mismatch, and limited post-hoc correction, but several geometric estimators remain exploratory.**
5. **The current E019 implementation is not a calibration-complexity map and should be replaced.**
6. **The current E018 pointwise cross-validation is informative, but its relational graph must be rebuilt fold-coherently and its non-scalar tiers should be reformulated in gauge-invariant ILR coordinates.**

---

## 2. Research question

The conventional questions are:

- Is the prediction correct?
- Is its confidence calibrated?
- Does the model match the average human label distribution?

This project asks a deeper set:

1. Does the model preserve the **type** of human ambiguity, not merely its amount?
2. Does the model recover the same **neighborhoods and analogies** among ambiguous cases?
3. How much human relational structure remains after conditioning on coarse information such as majority label and entropy?
4. How many effective human-distribution prototypes are needed to match the model’s relational performance?
5. Do different model families recover complementary parts of the geometry?
6. Which changes are reachable through post-hoc calibration, and which require a different representation or model?
7. Can ensemble relational structure be distilled into one efficient model?

---

## 3. Data and core representation

### 3.1 Dataset

The present core dataset contains:

- 3,113 ChaosNLI items;
- 1,514 SNLI items;
- 1,599 MNLI items;
- 100 human labels per item;
- three response classes: entailment, neutral, contradiction.

ChaosNLI was introduced to preserve and analyze collective human label distributions rather than collapse them to majority labels [1]. Prior work found that both conventional NLI systems and later LLMs struggle increasingly as human disagreement rises [1, 2].

### 3.2 Human probability vectors

For item \(i\),

\[
p_i=(p_{i,E},p_{i,N},p_{i,C}),\qquad \sum_c p_{i,c}=1.
\]

These vectors lie on a two-dimensional probability simplex.

A Dirichlet posterior represents finite-vote uncertainty:

\[
\theta_i\mid v_i\sim\operatorname{Dirichlet}(v_i+\alpha).
\]

### 3.3 Human relational target

For every posterior draw, a tie-aware \(k\)-nearest-neighbor graph is formed. The expected fuzzy human support matrix is

\[
S_{ij}=\mathbb E[W^{H}_{ij}].
\]

For model graph \(W^M\),

\[
Q_{\mathrm{support}}(M)
=
\frac{1}{Nk}\sum_{i,j}W^M_{ij}S_{ij}.
\]

A stratified identity null removes support available from coarse grouping or chance. A split-half human statistic supplies a reliability reference. The normalized recovery statistic is

\[
R_M=
\frac{Q_M-Q_{\mathrm{null},M}}
     {Q_{HH}-Q_{\mathrm{null},M}}.
\]

This converts a raw graph match into the fraction of reliable human relational excess recovered by the model.

---

## 4. Result ledger

## 4.1 E001 — Individual classifier relational recovery

**Status:** Foundational / frozen benchmark.

Nine NLI classifiers exceed a stratified null but recover only a minority of split-half human relational structure. Stronger systems generally recover more than compact systems, but no individual classifier approaches the human reference.

The exact numerical table should be pulled from the frozen canonical E001 artifact when drafting a paper. The current narrative-level result is:

> Standard classifiers are not merely imperfect at matching human probabilities; they organize ambiguous items at substantially lower relational resolution than human replicate cohorts.

This extends the original ChaosNLI finding from pointwise distribution recovery to uncertainty-aware neighborhood recovery [1].

---

## 4.2 E002 — Calibration improves NLL, barely relational recovery

**Status:** Publication-grade and frozen.  
**Commit:** `db95f37`.

Across nine classifiers, fold-specific temperature scaling was fitted on training data, applied coherently to all items within a fold, and scored only on held-out focal rows.

Key results:

- NLL gap closure: approximately **24.8%–56.6%**;
- relational topology gap closure: **at most about 0.70%**;
- Jensen–Shannon divergence worsened for all nine models;
- graph turnover under NLL-selected temperatures was approximately **13.4%–31.1%**;
- paired stratified bootstrap intervals for the difference between NLL and topology gap closure excluded zero for every model.

Interpretation:

> Temperature scaling can substantially improve forward cross-entropy while changing many edges and recovering almost none of the missing human-specific relational structure.

This sharpens the role of temperature scaling established by Guo et al. [3]. Temperature scaling is an effective one-parameter calibration method for selected probability objectives, but the present work shows that a better pointwise score does not imply recovery of a human relational geometry.

Important qualification:

- “Calibration” must name the selected objective.
- In this experiment, NLL improves while symmetric JSD worsens.
- The result does not imply that every calibration method is relationally ineffective.

---

## 4.3 E005 — Conditional-resolution ladder

**Status:** Audited pilot; full-data confirmatory run pending.

The nested null ladder conditions progressively on:

1. global identity;
2. source dataset;
3. majority label;
4. entropy;
5. top-two label identity and margin;
6. exact vote profile.

On the 600-item audited pilot, stronger models retained a small residual after the \(N_4\) null, while compact models often retained effectively none.

Representative pilot residual fractions after the \(N_4\) conditional null:

- BART-Large: approximately **9.7%**;
- XLNet-Large: approximately **7.7%**;
- RoBERTa-Large: approximately **4.8%**;
- RoBERTa-Base: approximately **0%**.

Frozen primary contrast:

\[
D_{\mathrm{size}}
=
\frac13\sum_{f\in\{\mathrm{BERT,RoBERTa,XLNet}\}}
\left(F_{f,\mathrm{large},N_4}-F_{f,\mathrm{base},N_4}\right).
\]

The full-data result must use common 30-stratum item bootstraps and report both residual fractions and absolute residual support.

Interpretation if replicated:

> Larger models preserve slightly finer item-specific relational structure beyond majority label, uncertainty magnitude, and ambiguity type.

This remains a hypothesis until the full confirmatory run is complete.

---

## 4.4 E007 — Exact ensemble coalition census

**Status:** Audited pilot; full census and held-out coalition selection pending.

All \(2^9-1=511\) nonempty model coalitions were enumerated in the pilot. Exact Shapley values attribute relational contribution across every coalition context.

Pilot normalized recovery:

- best single, BART-Large: approximately **25.5%**;
- best pair, BART + RoBERTa: approximately **34.5%**;
- best triplet, BART + RoBERTa + ALBERT: approximately **40.3%**;
- all nine classifiers: approximately **49.3%**.

All nine models received positive pilot Shapley contribution.

Interpretation:

> Model errors are not wholly redundant. Different architectures contribute partially complementary relational information.

This aligns with pluralistic multi-model work such as Modular Pluralism, which combines specialized models to represent multiple communities and pluralism objectives [10]. The present project contributes a different capability: an explicit relational value function and exact coalition attribution.

Required next steps:

- validate analytic nulls against literal stratified permutations;
- complete full-data 511-coalition census;
- perform training-only coalition selection and held-out focal-row evaluation;
- analyze contribution by ambiguity region, not only globally.

---

## 4.5 E008 — Prototype-equivalent relational resolution

**Status:** Audited pilot; full-data curve pending.  
**Commit:** `b62c392`.

Human distributions are compressed through cross-fitted Hellinger-space prototypes. The rate–distortion curve maps each model or ensemble to an external prototype-equivalent resolution.

Corrected pilot values:

| Condition | Normalized relational recovery | Prototype-equivalent bits | Effective states |
|---|---:|---:|---:|
| DistilBERT | 9.93% | 1.42 | 2.68 |
| BERT-Base | 10.81% | 1.51 | 2.85 |
| RoBERTa-Base | 12.29% | 1.64 | 3.12 |
| XLNet-Base | 13.60% | 1.76 | 3.39 |
| ALBERT-xxLarge | 16.37% | 1.99 | 3.97 |
| XLNet-Large | 20.54% | 2.24 | 4.72 |
| RoBERTa-Large | 21.38% | 2.29 | 4.89 |
| BART-Large | 25.45% | 2.54 | 5.82 |
| BART + RoBERTa | 34.46% | 3.02 | 8.11 |
| Best triplet | 40.30% | 3.29 | 9.78 |
| All nine models | 49.29% | 3.63 | 12.38 |

Interpretation:

> BART-Large’s relational recovery is comparable to a cross-fitted quantizer with roughly six human-distribution prototypes. The nine-model coalition raises this equivalence to roughly twelve states, approximately one additional prototype-equivalent bit.

Claim boundary:

- these are not internal model bits;
- they are not mutual information;
- they do not prove literal ambiguity concepts;
- they are conditional on the dataset, distance, \(k\), target support, prototype family, and interpolation procedure.

---

## 4.6 E014 — Disagreement flow fields and boundary collapse

**Status:** Exploratory, with corrected denominator.

For each item,

\[
v_i=q_i-p_i.
\]

The current analysis separates:

- increased Euclidean distance from the simplex center;
- projection toward the empirical human-majority vertex;
- transition from a declared human-interior region to a declared model-boundary region.

With:

\[
\min_c p_{ic}\ge0.05
\]

as the human-interior definition and

\[
\min_c q_{ic}<0.02
\]

as the model-boundary definition, 1,022 items are human-interior.

Reported collapse rates among these interior items range from:

- approximately **34.5%** for DistilBERT;
- up to approximately **75.5%** for RoBERTa-Large.

The stronger models also show larger average center-to-boundary sharpening.

However, mean projection toward the empirical majority vertex is negative for all nine models. This means the average displacement should not be described as simple majority-corner attraction.

More defensible interpretation:

> Model predictions often move away from diffuse human interiors toward lower-dimensional boundary regions, but the direction is not reducible to movement toward the empirical majority vertex.

Required analysis:

- threshold sweep;
- bootstrap intervals;
- which label is removed;
- local maps by ambiguity type;
- density-aware flow estimation.

---

## 4.7 E016 — Calibration Ray Theorem

**Status:** Established theorem plus exploratory empirical measurements.  
**Commit:** `bc5f074`.

Let model logits be \(z\), and let

\[
q(T)=\operatorname{softmax}(z/T).
\]

The centered log-ratio transform is

\[
\operatorname{clr}(q)
=
\log q-\frac1C\sum_c\log q_c.
\]

Then:

\[
\boxed{
\operatorname{clr}(q(T))
=
\frac1T\operatorname{clr}(q(1))
}
\]

for every \(T>0\).

Therefore, temperature scaling moves the prediction along a positive ray from the uniform distribution in CLR space. It changes radial magnitude but not direction.

For human target \(p_i\), define ambiguity angle

\[
\theta_i=
\arccos
\frac{
\langle \operatorname{clr}(p_i),\operatorname{clr}(q_i)\rangle
}{
\|\operatorname{clr}(p_i)\|
\|\operatorname{clr}(q_i)\|
}.
\]

Then:

\[
\boxed{\theta_i(T)=\theta_i(1)}
\]

whenever the angle is defined.

Empirical exploratory results across the nine classifiers:

- mean ambiguity angle: approximately **29.0°–46.2°**;
- mean ratio of human CLR norm orthogonal to the model calibration ray: approximately **41.5%–55.4%**.

The true theorem-level claim is:

> Scalar temperature is exactly incapable of rotating a model’s ambiguity direction in CLR geometry.

This provides a geometric explanation for E002. Temperature can improve sharpness-sensitive objectives while remaining unable to correct label-ratio direction.

The theorem sits naturally within compositional-data geometry, where probability vectors are treated as points in a simplex and log-ratio coordinates provide a principled Euclidean representation [5].

---

## 4.8 E018 — Reachable-set ladder

**Status:** Held-out pointwise result is promising; relational and non-scalar geometry require redesign.  
**Latest commit:** `abf81ec`.

Current tiers:

- Tier 0: raw;
- Tier 1: scalar temperature;
- Tier 2: diagonal logit scaling;
- Tier 3: affine softmax map;
- Tier 4: two-layer MLP.

The current five-fold pointwise results for three models are:

| Model | Raw Hellinger | Scalar | Diagonal | Affine | MLP |
|---|---:|---:|---:|---:|---:|
| ALBERT-xxLarge | 0.2923 | 0.2137 | 0.2098 | 0.1922 | 0.3054 |
| BART-Large | 0.2848 | 0.2092 | 0.2063 | 0.1871 | 0.3030 |
| RoBERTa-Large | 0.2829 | 0.2013 | 0.1972 | 0.1823 | 0.3013 |

Held-out ambiguity-angle reduction:

- scalar: exactly **0°**, as required by the theorem;
- diagonal: approximately **0.08°–0.40°**;
- affine: approximately **3.05°–4.49°**;
- current MLP: inconsistent and generally poor.

Defensible pointwise interpretation:

> More flexible global post-hoc transformations can improve held-out distributional fit and rotate ambiguity direction modestly. The particular affine family outperforms scalar and diagonal scaling on the three tested classifiers. The current MLP specification generalizes poorly.

Current limitations:

### A. Incoherent relational graph

The code concatenates predictions generated by five different fold-specific maps and then constructs one graph. Distances between nodes in different folds therefore compare outputs created under different coordinate transformations.

Required fix:

- for each fold, apply the fold-trained transformation to all items;
- build one coherent graph;
- score only held-out focal rows;
- aggregate held-out contributions.

### B. Logit-gauge dependence

Non-scalar transformations act on raw logits. Adding an item-specific constant to every logit leaves original probabilities unchanged but can alter diagonal, affine, or MLP outputs.

Required fix:

- convert predictions to two-dimensional ILR coordinates;
- define all tiers in that gauge-invariant space;
- map back through inverse ILR.

A clean nested ladder is:

\[
x'=\alpha x
\]

for scalar radial scaling,

\[
x'=Dx
\]

for positive diagonal ILR scaling,

\[
x'=Ax+b
\]

for affine ILR mapping, and

\[
x'=f_\theta(x)
\]

for a nonlinear map.

### C. Nonlinear conclusion

The current MLP predicts probability vectors and then treats them as logits before softmax. It is one poorly specified baseline, not evidence that nonlinear calibration generally overfits.

The literature already contains more principled multiclass calibration families, including Dirichlet calibration, which applies a linear layer to log probabilities followed by softmax [4]. This is a more natural comparison tier.

---

## 4.9 E019 — Minimal calibration complexity map

**Status:** Current implementation invalid for its advertised claim; replace artifact.

The intended definition is:

\[
c_i(\epsilon)
=
\min\{t:H(p_i,q_i^{(t)})\le\epsilon\}.
\]

The current script does not use actual per-tier predictions. It assigns classes by raw-distance intervals.

Therefore, its current output is not a minimal calibration-complexity map.

Required rewrite:

1. save per-item out-of-fold predictions from every corrected E018 tier;
2. calculate actual per-tier Hellinger distances;
3. assign the first successful tier;
4. use “unreached by tested maps,” not “unreachable”;
5. sweep \(\epsilon\in\{0.05,0.10,0.15,0.20\}\);
6. store per-item IDs, folds, coordinates, entropy, label structure, and linguistic features.

This future E019 could become one of the best visual artifacts:

> Some cases require only a radial confidence correction; others require a rotation or nonlinear remapping of ambiguity geometry.

---

## 5. Relationship to prior research

## 5.1 Collective human opinions in NLI

ChaosNLI established that disagreement is common, that models fail to reproduce human label distributions, and that model performance degrades sharply on low-agreement items [1].

Lee, An, and Thorne later evaluated LLM distributions using log-probability and Monte Carlo estimation and found similarly limited recovery of dissenting human distributions, especially under high disagreement [2].

The present project extends this line by asking not only whether each output distribution is close to humans, but whether the **relations among cases** are recovered.

## 5.2 Calibration

Temperature scaling became a standard baseline because a single fitted temperature often improves probability calibration on neural classifiers [3].

Dirichlet calibration generalizes beyond one scalar through a natively multiclass map on log probabilities [4].

The present work contributes:

1. a proof that scalar temperature is a positive CLR ray;
2. ambiguity-angle invariance;
3. evidence that pointwise improvement can coexist with negligible relational recovery;
4. a planned gauge-invariant hierarchy of reachable sets.

## 5.3 Probability-simplex geometry

Aitchison’s foundational work argues that compositions live in a simplex and require geometry respecting the fixed-sum constraint [5].

The present project primarily uses Hellinger geometry for posterior graphs. Exploratory metric audits show:

- Hellinger and categorical Fisher–Rao induce identical rankings and \(k\)-NN graphs because both are monotone transformations of the Bhattacharyya coefficient;
- Hellinger and JSD produce nearly identical local ChaosNLI neighborhoods;
- Aitchison/CLR geometry changes a material minority of neighborhoods and deserves boundary-policy analysis.

The next calibration ladder should use ILR coordinates to preserve the geometry and remove logit-gauge arbitrariness.

## 5.4 Perspectivist and pluralistic evaluation

Perspectivist research argues that disagreement should not automatically be deleted through majority aggregation [6, 7]. More recent evaluation frameworks, including PERSEVAL, emphasize user- and annotator-level evaluation rather than one aggregated gold label [11].

OpinionQA measures which demographic-group opinions language models reflect and finds substantial group misalignment [9]. Modular Pluralism combines specialized community models to support distributional, steerable, and Overton pluralism [10].

The present work is complementary:

> It supplies a relational-resolution axis: which neighborhoods, ambiguity types, and effective prototype states a model or coalition preserves.

ChaosNLI does not include the identity metadata necessary to make demographic-minority claims. It measures statistical minority interpretations, not demographic group representation.

## 5.5 Relational distillation

Relational Knowledge Distillation transfers pairwise distances and angles rather than only per-example teacher outputs [8]. Similarity-preserving distillation likewise trains students to preserve teacher pairwise similarity structure [12].

This directly motivates a future method:

\[
L
=
L_{\mathrm{pointwise}}
+
\lambda_1 L_{\mathrm{distance}}
+
\lambda_2 L_{\mathrm{edge}}
+
\lambda_3 L_{\mathrm{prototype}}.
\]

The novel objective would not simply preserve a teacher’s hidden geometry. It would preserve **human-supported disagreement geometry** recovered by an ensemble.

## 5.6 Structural comparison

Gromov–Wasserstein methods compare relational structures and can infer correspondences across spaces [13]. They are promising for comparing group- or model-specific judgment geometries when one does not want to assume a fixed identity alignment.

However, GW should come after cleaner identity-aligned analyses. Free correspondence can hide item-level errors by rematching nodes, and structural similarity does not imply shared semantics.

---

## 6. What appears novel

The individual ingredients have precedents:

- label distributions;
- perspectivist ground truth;
- calibration;
- probability geometry;
- relational distillation;
- ensemble attribution;
- graph matching.

The likely novelty lies in their combination.

### Candidate contribution 1: posterior relational evaluation

An uncertainty-aware human support graph evaluates which analogies among ambiguous cases a model recovers.

### Candidate contribution 2: conditional relational resolution

A nested null ladder identifies how much relational alignment remains beyond label, entropy, ambiguity type, and margin.

### Candidate contribution 3: prototype-equivalent resolution

Models and ensembles are mapped to a common human-distribution compression scale.

### Candidate contribution 4: calibration-ray theorem

Scalar temperature scaling is characterized as an exact positive ray in CLR space, proving ambiguity-angle invariance.

### Candidate contribution 5: calibration versus relational topology

Large pointwise gains can coexist with negligible human-specific relational gains.

### Candidate contribution 6: exact ensemble complementarity

Exact coalition census and Shapley attribution quantify nonredundant relational information across models.

### Candidate contribution 7: geometry-preserving optimization

The framework suggests new calibration and distillation objectives that preserve human-supported structure rather than only average probability fit.

---

## 7. Claim boundaries

The following distinctions must stay explicit.

### Supported

- ChaosNLI item-level vote distributions have measurable geometry.
- The evaluated classifiers recover limited human relational structure.
- Temperature scaling substantially improves held-out NLL.
- Scalar temperature cannot rotate CLR ambiguity direction.
- The audited pilot ensemble recovers more relational structure than any single classifier.
- Prototype-equivalent resolution increases under ensembling.

### Not yet supported

- demographic minority erasure in ChaosNLI;
- stable individual belief or opinion twins;
- semantic equivalence of all distributionally close items;
- causal mechanisms behind human disagreement;
- a valid E019 item complexity map;
- coherent held-out E018 relational recovery;
- generic nonlinear calibration failure;
- proof that the residual affine angle is mathematically unreachable;
- proof that current local Jacobian estimates represent true manifold collapse.

### Terminology to prefer

Use:

- statistical minority interpretation;
- aggregate vote-distribution geometry;
- posterior-supported human neighborhood;
- prototype-equivalent relational resolution;
- unreached by tested maps;
- lexically matched distribution twins;
- external performance equivalence.

Avoid:

- demographic minority erasure without group metadata;
- true internal bits;
- semantic twins without semantic adjudication;
- unreachable unless supported by a theorem;
- proves flattening when based on unstable local regression.

---

## 8. Open thread ledger

## Thread A — Complete the frozen core portfolio

### E004: local modern LLM bridge

**State:** Running or awaiting final integration.

Required outputs:

- Gemma log-probability estimate;
- calibrated LPE;
- Monte Carlo estimate;
- consistent fold-specific graph evaluation;
- mapping to prototype-equivalent resolution;
- comparison with classifiers and ensembles.

### E005: full conditional ladder

**Next action:**

- run all 3,113 items;
- use common 30-stratum bootstrap;
- report matched-family primary contrast;
- report absolute residuals and denominator diagnostics.

### E007: full exact coalition census

**Next action:**

- validate analytic nulls;
- run all 511 coalitions;
- held-out coalition selection;
- region-specific Shapley maps.

### E008: full rate–distortion curve

**Next action:**

- run full-data persisted folds;
- bootstrap effective bits;
- remap all models, coalitions, and Gemma conditions.

---

## Thread B — Repair and freeze E018

1. Reuse the persisted 30-stratum folds.
2. Assert object ordering and hashes.
3. Transform model probabilities into ILR coordinates.
4. Implement gauge-invariant nested tiers.
5. Add Dirichlet calibration as a principled flexible baseline.
6. Fit each fold on training items.
7. Apply each fold map to all items.
8. Build a coherent full graph for that fold.
9. Score only held-out focal rows.
10. Use frozen posterior support \(S\), coherent nulls, and normalized \(R\).
11. Store per-item OOF predictions and pointwise outcomes.
12. Add paired stratified bootstrap intervals.
13. Report optimizer convergence and fold stability.

---

## Thread C — Rebuild E019 correctly

Per item and tier, store:

```text
object_id
fold_id
human_distribution
raw_prediction
tier_1_prediction
tier_2_prediction
tier_3_prediction
tier_4_prediction
raw_distance
tier_1_distance
tier_2_distance
tier_3_distance
tier_4_distance
minimal_successful_tier
simplex_coordinates
entropy
majority_label
top_two_pair
dataset
```

Then:

- sweep tolerance;
- map classes on the simplex;
- identify regressions where later tiers hurt;
- test associations with linguistic features;
- compare model families;
- create a clickable “calibration obstruction” visualizer.

---

## Thread D — Stabilize E017 differential belief maps

Current local Jacobian maps are exploratory.

Required:

- ILR-space local regression;
- minimum effective local sample size;
- bootstrap intervals;
- multiple bandwidths or grids;
- occupancy weighting;
- residual rather than raw output dispersion;
- condition-number diagnostics;
- preregistered ambiguity regions.

Potential scientific quantities:

- local area change;
- anisotropy ratio;
- rotation angle;
- residual content-dependent dispersion.

---

## Thread E — Geometry-preserving distillation

Start with a cheap proof-of-concept student.

Compare:

1. standard soft-label distillation;
2. pairwise Hellinger-distance preservation;
3. human-edge support loss;
4. prototype assignment preservation;
5. mixed objective.

Primary outcome:

> Fraction of ensemble prototype-equivalent resolution retained by one student model.

This is likely the strongest next method paper after the evaluation framework.

---

## Thread F — Annotation-budget geometry

Ask how many votes are needed to stabilize:

- pointwise distributions;
- nearest-neighbor graph;
- model rankings;
- prototypes;
- effective bits;
- conditional residuals.

This can become a practical annotation-allocation system.

---

## Thread G — Public-dataset pluralism

Use cross-dataset triangulation rather than concatenation.

Candidate datasets:

- OpinionQA for demographic-group distributions [9];
- PRISM for participant-level and cross-cultural preferences;
- rationale datasets for label-versus-reason geometry;
- moral or safety datasets with repeated judgments;
- argument and viewpoint graphs.

Questions:

- Does relational under-resolution generalize?
- Do models miss group-specific regions?
- Is calibration relationally weak across domains?
- Are disagreement prototypes transferable?

---

## 9. Recommended paper architecture

### Paper A — Core methods and classifier results

**Working title:**

> From Label Distributions to Relational Pluralism: Measuring the Geometry Models Recover from Human Disagreement

Sections:

1. human posterior relational target;
2. individual classifier recovery;
3. calibration versus topology;
4. conditional null ladder;
5. exact ensemble complementarity;
6. prototype-equivalent resolution.

### Paper B — Calibration geometry

**Working title:**

> Better Calibrated, Same Ambiguity Direction: The Geometry and Limits of Post-Hoc Calibration

Sections:

1. Calibration Ray Theorem;
2. ambiguity angles;
3. E002 pointwise/topological disconnect;
4. gauge-invariant reachable-set ladder;
5. item-level calibration complexity map;
6. relationally constrained calibration.

### Paper C — Optimization

**Working title:**

> Distilling Pluralistic Resolution: Preserving Human Disagreement Geometry in a Single Model

---

## 10. Draft abstract

Human disagreement is usually evaluated through majority labels or per-item label distributions, obscuring how judgments are organized across cases. We introduce an uncertainty-aware relational framework that represents each item by a posterior distribution over human labels and evaluates whether model-induced neighborhoods align with posterior-supported human neighborhoods. Across nine natural language inference classifiers, individual systems recover only a limited fraction of reliable human relational structure. Fold-specific temperature scaling substantially improves soft-label cross-entropy but produces less than one percent relational gap closure, despite appreciable graph turnover. A nested conditional-null analysis suggests that stronger models preserve a small amount of item-specific structure beyond label and ambiguity summaries. Exact coalition enumeration shows that model families contribute complementary relational information, and a cross-fitted rate–distortion analysis maps this gain to an interpretable prototype-equivalent scale: the strongest single classifier matches roughly six human-distribution states, while a nine-model coalition matches roughly twelve. We further prove that scalar temperature scaling traces a positive ray in centered log-ratio space and therefore cannot rotate a model’s ambiguity direction. Together, these results motivate relational pluralism as a distinct evaluation and optimization target: models should be assessed not only by how well they fit average human probabilities, but by which structures and alternatives in collective human judgment they preserve.

**Note:** This abstract combines frozen results and audited pilots. Before submission, update E005, E007, and E008 with full-data confirmatory results and decide whether the calibration theorem belongs in the same paper or a companion paper.

---

## 11. Immediate restart note

When returning to the project, begin here:

1. Check the status and artifacts of E004.
2. Do not rely on current E019 outputs.
3. Preserve E016 as theorem-level.
4. Preserve current E018 pointwise CV as exploratory evidence.
5. Redesign E018 in ILR coordinates with fold-coherent graph scoring.
6. Run the full frozen E005/E007/E008 sequence before expanding the portfolio further.
7. Keep `research/geometry-sandbox` separate from confirmatory branches.
8. Convert any result promoted from sandbox into a registry entry with:
   - estimand;
   - folds;
   - null;
   - uncertainty;
   - artifact hashes;
   - claim boundary.

---

## 12. References

[1] Nie, Y., Zhou, X., & Bansal, M. (2020). *What Can We Learn from Collective Human Opinions on Natural Language Inference Data?* EMNLP 2020. DOI: 10.18653/v1/2020.emnlp-main.734.  
https://aclanthology.org/2020.emnlp-main.734/

[2] Lee, N., An, N. M., & Thorne, J. (2023). *Can Large Language Models Capture Dissenting Human Voices?* EMNLP 2023. DOI: 10.18653/v1/2023.emnlp-main.278.  
https://aclanthology.org/2023.emnlp-main.278/

[3] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML 2017, PMLR 70.  
https://proceedings.mlr.press/v70/guo17a.html

[4] Kull, M., Perello-Nieto, M., Kängsepp, M., Silva Filho, T., Song, H., & Flach, P. (2019). *Beyond Temperature Scaling: Obtaining Well-Calibrated Multi-Class Probabilities with Dirichlet Calibration.* NeurIPS 2019.  
https://proceedings.neurips.cc/paper_files/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html

[5] Aitchison, J. (1982). *The Statistical Analysis of Compositional Data.* Journal of the Royal Statistical Society, Series B, 44(2), 139–160. DOI: 10.1111/j.2517-6161.1982.tb01195.x.  
https://academic.oup.com/jrsssb/article/44/2/139/7027742

[6] Cabitza, F., Campagner, A., & Basile, V. (2023). *Toward a Perspectivist Turn in Ground Truthing for Predictive Computing.* AAAI 2023. DOI: 10.1609/aaai.v37i6.25840.  
https://ojs.aaai.org/index.php/AAAI/article/view/25840

[7] Fleisig, E., Blodgett, S. L., Klein, D., & Talat, Z. (2024). *The Perspectivist Paradigm Shift: Assumptions and Challenges of Capturing Human Labels.* NAACL 2024. DOI: 10.18653/v1/2024.naacl-long.126.  
https://aclanthology.org/2024.naacl-long.126/

[8] Park, W., Kim, D., Lu, Y., & Cho, M. (2019). *Relational Knowledge Distillation.* CVPR 2019.  
https://arxiv.org/abs/1904.05068

[9] Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P., & Hashimoto, T. (2023). *Whose Opinions Do Language Models Reflect?* ICML 2023, PMLR 202.  
https://proceedings.mlr.press/v202/santurkar23a.html

[10] Feng, S., Sorensen, T., Liu, Y., Fisher, J., Park, C. Y., Choi, Y., & Tsvetkov, Y. (2024). *Modular Pluralism: Pluralistic Alignment via Multi-LLM Collaboration.* EMNLP 2024. DOI: 10.18653/v1/2024.emnlp-main.240.  
https://aclanthology.org/2024.emnlp-main.240/

[11] Lo, S. M., Casola, S., Sezerer, E., Basile, V., Sansonetti, F., Uva, A., & Bernardi, D. (2025). *PERSEVAL: A Framework for Perspectivist Classification Evaluation.* EMNLP 2025. DOI: 10.18653/v1/2025.emnlp-main.1137.  
https://aclanthology.org/2025.emnlp-main.1137/

[12] Tung, F., & Mori, G. (2019). *Similarity-Preserving Knowledge Distillation.* ICCV 2019.  
https://arxiv.org/abs/1907.09682

[13] Xu, H., Luo, D., Zha, H., & Carin, L. (2019). *Gromov-Wasserstein Learning for Graph Matching and Node Embedding.* ICML 2019, PMLR 97.  
https://proceedings.mlr.press/v97/xu19b.html

---

## 13. Internal artifact provenance

| Experiment / package | Status | Commit or branch reference |
|---|---|---|
| E002 publication-grade calibration | Frozen | `db95f37` |
| E007 exact coalition pilot | Audited pilot | `f9dce21` and subsequent audit commits |
| E008 prototype resolution pilot | Audited pilot | `b62c392` |
| Persistent geometry sandbox | Exploratory | `d73409f` |
| E016/E017 refinement | Exploratory / theorem | `bc5f074` |
| Correction patch and initial E018 | Exploratory | `ee8b2d0` |
| E018 CV and current E019 | Exploratory; E019 invalid | `abf81ec97822ffd0862694bbe3814faa6ed50fb1` |

Primary current paths:

```text
research/chaosnli/lab/exploratory/
results/exploratory/
docs/studies/chaosnli/PERSISTENT_GEOMETRY_AND_METRIC_AUDIT.md
docs/viz/chaosnli/geometry_lens.html
```
