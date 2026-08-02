# ChaosNLI Hypotheses and Analysis Plan

- **Document type:** analysis plan
- **Status:** planning document; not a completed preregistration

## 1. Scope and inferential population

Primary population:

> The 3,113 ChaosNLI-SNLI and ChaosNLI-MNLI items selected and reannotated by the ChaosNLI study.

Do not generalize without qualification to all SNLI/MNLI, arbitrary NLI tasks, other languages, experts, individuals, or abductive NLI.

The 100 labels within an item estimate its collective distribution. They are not 100 independent dataset items. The unit of generalization is the NLI item.

---

## 2. Analysis partitions

Use deterministic stratified partitions, for example:

```text
engineering: 300
exploratory: 1,969
confirmatory: 844
```

Stratify by:

- source dataset;
- human majority label;
- entropy quintile.

Exact counts can change before lock.

No confirmatory item may be used to choose thresholds, \(k\), models, zero policy, packet rules, or calibration.

External taxonomy subsets should preserve original development/validation distinctions when available.

---

## 3. Primary estimands

### 3.1 Pointwise alignment

For item \(i\), model \(m\):

\[
D_{im}=d_{JS}(\hat p_i,q_{im}).
\]

Primary summary:

\[
\bar D_m=\frac1N\sum_iD_{im}.
\]

Also report median and quantiles.

### 3.2 Neighborhood recovery

For \(k=10\):

\[
O_{im}(k)
=
\frac{|N_H(i;k)\cap N_m(i;k)|}{k}.
\]

\[
Q_{NX,m}(k)=\frac1N\sum_iO_{im}(k).
\]

\[
LCMC_m(k)=Q_{NX,m}(k)-\frac{k}{N-1}.
\]

### 3.3 Human reliability

For split \(b\), \(Q^{HH,(b)}_{NX}(k)\) compares two 50-label human halves.

Report median, 2.5%/97.5% quantiles, and model recovery as a proportion of the median. This normalization is descriptive, not an absolute upper-bound theorem.

### 3.4 Edge support

Human posterior support \(s_{ij}\): fraction of posterior graphs containing directed edge \(i\to j\).

Model consensus \(c_{ij}\): fraction of model graphs containing it.

---

## 4. Confirmatory hypotheses

### H1 — Model graphs do not reach human split-half reliability

#### Statement

For each preregistered model,

\[
Q^{HM}_{NX}(10)
<
median_b[Q^{HH,(b)}_{NX}(10)].
\]

#### Test

Item-level difference:

\[
\Delta_i=
O_i^{HM}(10)-mean_b[O_i^{HH,(b)}(10)].
\]

Use paired item bootstrap stratified by dataset and majority label.

#### Report

- mean difference;
- bootstrap 95% CI;
- effect relative to split-half variability;
- separate SNLI/MNLI;
- full \(k\)-curve.

#### Null interpretation

An interval around zero means the model recovers local structure about as well as two 50-vote human estimates recover each other at this scale. It does not mean the model reproduces human reasoning.

---

### H2 — Pointwise calibration gains exceed topology gains

#### Statement

Temperature scaling reduces average JSD more strongly than it improves neighborhood recovery:

\[
\Delta JSD_m=
\bar D_{m,cal}-\bar D_{m,raw}<0,
\]

while

\[
\Delta Q_m=
Q_{m,cal}(10)-Q_{m,raw}(10)
\]

has smaller standardized magnitude.

#### Test

- paired item bootstrap for JSD change;
- paired item bootstrap for local-overlap change;
- compare standardized changes through bootstrap distributions.

#### Control

Calibration fitted outside locked confirmatory items.

#### Meaning

Tests whether marginal probability alignment is sufficient to recover relational human structure.

---

### H3 — Human disagreement moderates graph recovery

#### Statement

Model–human overlap decreases as human entropy rises.

#### Model

\[
shared_{im}\sim Binomial(k,\pi_{im}),
\]

\[
logit(\pi_{im})
=
\beta_0+\beta_1H_i+\beta_2Dataset_i+\beta_3Model_m+
\beta_4MajorityLabel_i+u_i.
\]

Because edges are dependent, use this for effect estimation and pair it with stratified item bootstrap.

Primary parameter:

\[
\beta_1<0.
\]

Robustness:

- entropy quintiles;
- posterior mean entropy;
- maximum-vote share;
- JSD geometry;
- separate SNLI/MNLI.

---

### H4 — Calibration does not eliminate mismatch asymmetry

#### Statement

After calibration, uncertainty collapse remains more common than spurious uncertainty on high-disagreement items.

Collapse probability:

\[
P(H(q_i)<H(\theta_i)\mid x_i).
\]

Spurious probability:

\[
P(H(q_i)>H(\theta_i)\mid x_i).
\]

Classify only above posterior probability 0.95; otherwise unresolved.

#### Test

Within high-entropy items, compare categories using paired item bootstrap or multinomial modeling.

Report every model separately.

---

### H5 — Model consensus is not sufficient for human support

#### Statement

There are recurring edges with high model consensus and low human support.

Primary prevalence:

\[
P(c_{ij}\ge0.80,\;s_{ij}\le0.10)
\]

among the union of model top-\(k\) edges.

#### Test

Descriptive prevalence with item-clustered bootstrap intervals.

Null: permute model distributions across item IDs within dataset, majority-label, and entropy strata.

#### Meaning

Demonstrates model-family relations not explained by human judgment similarity. It does not identify their cause.

---

### H6 — Externally validated disagreement types have relational signal

#### Statement

On externally annotated items, human-opinion neighborhoods show above-null agreement in high-level disagreement categories after controlling for majority and entropy.

For taxonomy sets \(T_i\):

\[
A_i(k)
=
\frac1k\sum_{j\in N_H(i;k)}
Jaccard(T_i,T_j).
\]

#### Null

Permute taxonomy sets within majority-label, entropy, and dataset/genre strata where sample size permits.

#### Test

Permutation distribution of mean \(A_i(k)\).

#### Limitation

The taxonomy was developed on a subset of these data. Prefer held-out annotation rounds or VariErr overlap for confirmatory validation.

---

### H7 — Joint opinion-and-text neighborhoods better predict shared reasons

#### Statement

Edges supported in human-opinion and text-semantic space have greater taxonomy/explanation agreement than either alone.

Groups:

```text
joint
human_only
text_only
matched_neither
```

#### Test

Compare taxonomy-set Jaccard or explanation similarity with item-clustered bootstrap and degree-preserving permutation where practical.

#### Meaning

Supports multi-view investigation, not ground-truth status for the text encoder.

---

### H8 — Reliability-aware visualization improves diagnostic accuracy

Later human-subject study.

#### Conditions

A. text + table + ternary  
B. static comparison + neighbor list  
C. Shadowspace reliability-aware comparison

#### Task labels

- human-supported;
- model-specific;
- human variation collapsed;
- spurious model uncertainty;
- projection/metric-sensitive;
- unresolved.

#### Primary outcome

Correct/incorrect against a frozen rubric.

#### Secondary

- confidence calibration;
- time;
- source-text inspection;
- use of unresolved;
- claim-qualification score.

#### Analysis

Mixed-effects logistic regression with participant and item random intercepts, plus paired nonparametric sensitivity.

Formative sessions are not confirmatory.

---

## 5. Exploratory questions

- Which label transitions dominate residuals?
- Are neutral–entailment and neutral–contradiction neighborhoods different?
- Do genre or length modify recovery?
- Are human-only edges enriched for implicature/coreference/lexical cases?
- Which models share unsupported edges?
- Do text neighbors explain model-only edges?
- Are correct-majority/wrong-shape cases model-family-specific?
- Do LLM sampling and classifier softmax fail differently?
- Are model distributions closer to valid VariErr label sets than raw ChaosNLI counts?
- How often do posterior-uncertain edges look visually stable?
- Which mismatches disappear after controlling for annotation error?

---

## 6. Statistical principles

### Item-level resampling

Resample items, not individual pairwise distances. For edge statistics, resample focal nodes with outgoing edges or use graph-aware permutation.

### Stratification

Maintain dataset and majority-label composition in bootstrap/permutation.

### Multiple comparisons

For H1–H7 define one primary metric, \(k\), and model comparison/hierarchical model. Use Holm correction across the confirmatory family when emphasizing p-values. Prefer estimates and intervals.

### Missing external annotations

“Not annotated” is not “no category.” Report eligible, matched, excluded, and reasons.

### Thresholds

Mismatch labels are for selection/communication. Primary analysis uses continuous values.

### Negative results

Taxonomy categories may not cluster in opinion space. That would be informative because similar distributions may arise from heterogeneous reasons.

---

## 7. Robustness matrix

| Dimension | Values |
|---|---|
| Dataset | SNLI, MNLI, pooled |
| \(k\) | 5, 10, 20, 50 |
| Geometry | Hellinger, JSD, TVD, Euclidean |
| Human estimate | empirical, posterior mean, posterior draws |
| Prior | Dirichlet 0.5, Dirichlet 1.0 |
| Model | each separately |
| Calibration | raw, calibrated |
| Edge direction | directed, mutual kNN |
| Taxonomy | fine, high-level |
| Item set | exploratory, confirmatory |

Aitchison also varies zero policy.

---

## 8. Reporting tables

### Dataset audit

```text
dataset
items
majority E/N/C
mean entropy
zero-count prevalence
external annotation overlap
```

### Pointwise alignment

```text
model
calibration
mean JSD
median JSD
multinomial log score
majority accuracy
```

### Graph recovery

```text
model
metric
k
QNX
LCMC
human split-half median
fraction of human reliability
```

### Mismatch prevalence

```text
model
collapse
spurious uncertainty
majority reversal
correct-majority/wrong-shape
unsupported mass
```

### Taxonomy validation

```text
taxonomy level
metric
k
observed homophily
null mean
effect
permutation p
```

### Consensus failures

```text
model consensus threshold
human support threshold
edge count
item count
prevalence
bootstrap CI
```

---

## 9. Release criteria

- [ ] Study and analysis manifests locked.
- [ ] Raw hashes match.
- [ ] Original baselines reproduced.
- [ ] Confirmatory items not used for tuning.
- [ ] Model label maps verified.
- [ ] Calibration leakage checks pass.
- [ ] Primary estimates regenerated by CLI.
- [ ] Robustness complete.
- [ ] Exclusions documented.
- [ ] Coding reliability reported.
- [ ] Every figure contains release/state ID.
- [ ] Shadowspace reproduces analysis coordinates and diagnostics.
