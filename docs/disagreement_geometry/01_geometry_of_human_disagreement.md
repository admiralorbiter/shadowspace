# The Geometry of Human Disagreement

## Executive thesis

Human disagreement is usually collapsed into one of four summaries:

- majority label;
- inter-rater agreement;
- entropy;
- average preference.

Those quantities are useful, but none captures the full relational organization of judgments.

Two items may have the same entropy while reflecting different ambiguity types. Two populations may have the same average response while containing different camps. A model may match every item’s marginal confidence reasonably well while placing the items into the wrong neighborhoods. Conversely, a model may reproduce the broad shape of a human judgment space while misassigning individual items.

A geometric view asks:

1. **Where do judgment distributions lie?**
2. **Which items or people are near one another?**
3. **Which regions or clusters exist?**
4. **Which structures persist across samples, models, groups, or time?**
5. **What resolution is needed to preserve the structure?**
6. **What does aggregation destroy?**

## 1. The basic spaces

### 1.1 Item-level probability simplex

For an item with \(C\) labels, the empirical human distribution is

\[
p_i=(p_{i1},\ldots,p_{iC}),\qquad p_{ic}\ge 0,\qquad \sum_c p_{ic}=1.
\]

For three-label NLI, the space is a triangle.

- A **corner** represents near-consensus on one label.
- An **edge** represents ambiguity concentrated between two labels.
- The **center** represents diffuse three-way uncertainty.
- A **shoulder** near one corner represents a dominant label with a structured alternative.
- Equal entropy contours cut across these qualitatively different regions.

The point is simple but important:

> “How much disagreement?” and “what kind of disagreement?” are different questions.

### 1.2 Annotator–item response space

If repeated annotator identities are available, define an annotator–item matrix \(Y\).

This permits a different geometry:

- distances between annotators;
- latent viewpoints;
- item discrimination;
- stable personal response patterns;
- cross-group or longitudinal movement.

ChaosNLI does not expose the repeated identity structure needed for a true individual-level geometry. It supplies rich item-level collective distributions.

### 1.3 Relational graph

Given item distributions, build a graph whose nodes are items and whose edges encode similarity or posterior-supported neighborhood membership.

This changes the target from:

> Did the model predict the correct distribution for item \(i\)?

to:

> Did the model organize item \(i\) near the same cases that humans organize it near?

That is the central shift in the current project.

### 1.4 Collective twin

A collective twin is not a population mean. It is a model of a group’s full judgment organization:

- item distributions;
- uncertainty;
- prototypes;
- relational graph;
- subgroups;
- missing regions;
- potentially temporal dynamics.

Two collective twins can overlap in their average judgment while differing radically in internal structure.

## 2. Canonical disagreement shapes

### 2.1 Consensus

Most mass lies near one category.

Possible interpretations:

- genuinely clear item;
- shared convention;
- annotation shortcut;
- common artifact.

Consensus is not automatically truth.

### 2.2 Binary ambiguity

Mass lies primarily between two categories.

In NLI this can distinguish:

- entailment–neutral ambiguity;
- neutral–contradiction ambiguity;
- entailment–contradiction conflict.

Entropy may be identical across these cases while the semantic alternatives differ.

### 2.3 Diffuse ambiguity

Mass is spread across all labels.

Possible causes include:

- underspecification;
- multiple plausible readings;
- poor task instructions;
- missing context;
- mixed annotation strategies.

### 2.4 Population polarization

A population may split into camps even when the aggregate vote vector resembles ordinary uncertainty.

A single item-level distribution cannot tell whether a 50/50 split arose from:

- two stable camps;
- independently uncertain individuals;
- repeated noisy judgments;
- two demographic groups;
- two interpretations of the prompt.

Detecting polarization requires annotator-level or group-level structure.

### 2.5 Fragmentation

More than two stable clusters or viewpoints appear.

This is central in Q methodology, opinion dynamics, and pluralistic-alignment research. Majority vote is particularly lossy under fragmentation.

### 2.6 Nested consensus

Groups may disagree globally while showing strong within-group agreement.

Examples:

- political parties;
- professional specialties;
- cultural groups;
- novice versus expert populations.

The geometry must distinguish:

\[
\text{within-group cohesion}
\quad\text{from}\quad
\text{between-group separation}.
\]

### 2.7 Cross-cutting structure

A single left–right axis may fail because different issues create different coalitions.

This is a major lesson from ideal-point research and multidimensional opinion models: the geometry may be low-dimensional overall but issue-dependent locally.

## 3. What older research learned

### 3.1 Reliability theory

Classical agreement metrics ask whether measurements are dependable.

Generalizability theory goes further by decomposing variance into facets such as:

- persons;
- items;
- raters;
- occasions;
- interactions among them.

**Lesson for this project:** disagreement should be decomposed by source rather than treated as one undifferentiated residual.

**Experiment implied:** estimate how much graph instability comes from finite human votes, item sampling, model variation, and metric choices.

### 3.2 Cultural consensus theory

Cultural consensus models infer shared answers and respondent competence from patterns of agreement when the “correct” answer is not directly observed.

**Lesson:** agreement matrices contain latent information about shared culture and respondent structure.

**Caution:** classic consensus models often assume one cultural truth. Modern pluralistic settings may require multiple-consensus or mixture extensions.

### 3.3 Q methodology

Q methodology is designed to reveal structured subjectivity. People rank statements, and factor analysis groups people who arrange them similarly.

**Lesson:** viewpoints are patterns across items, not isolated responses.

**Experiment implied:** with repeated identities, factor or cluster annotators first, then compare the geometry of group-specific item judgments.

### 3.4 Ideal-point and item-response models

These models place respondents and items in latent spaces. A response is explained by their relative positions and the item’s discrimination or difficulty.

**Lesson:** people and questions can be embedded jointly.

**Experiment implied:** fit a latent-space model to repeated human judgments and compare model-generated “respondents” against human respondent regions.

### 3.5 Judgment aggregation and social choice

Impossibility results show that individually rational judgments cannot always be aggregated into one collectively rational set while satisfying seemingly mild principles.

**Lesson:** the loss created by majority aggregation is not merely an engineering mistake. Some aggregation tensions are structural.

**Experiment implied:** quantify which logical or relational structures disappear under majority-label compression.

### 3.6 Opinion dynamics

DeGroot, bounded-confidence, homophily, and related models study when interaction produces:

- consensus;
- polarization;
- fragmentation;
- persistent minorities.

**Lesson:** shape is dynamic, and averaging does not universally produce polarization. The interaction rule matters.

**Experiment implied:** use longitudinal judgments to test whether AI mediation reduces disagreement, reorganizes it, or simply pushes minorities out of measured space.

### 3.7 Polarization measurement

Polarization is distinct from variance or inequality. It involves identification within groups and alienation between groups.

**Lesson:** entropy is not a polarization measure.

**Experiment implied:** construct group-aware measures that combine within-cluster cohesion and between-cluster separation.

### 3.8 Compositional and information geometry

Probability vectors live on a simplex, not unconstrained Euclidean space.

Useful geometries include:

- Hellinger / square-root geometry;
- Fisher–Rao information geometry;
- Jensen–Shannon divergence;
- Aitchison log-ratio geometry;
- optimal-transport geometry when labels have an external ground metric.

**Lesson:** the metric defines what “similar judgment” means.

**Experiment implied:** test which geometry yields the most stable human graph and the strongest out-of-sample interpretability.

### 3.9 Crowd disagreement and perspectivism

CrowdTruth and perspectivist AI argue that disagreement can represent legitimate ambiguity rather than worker error.

**Lesson:** preserving multiple labels is not enough; one must model ambiguity in the item, worker, and annotation process.

### 3.10 Pluralistic alignment

Recent work asks whether LLMs represent demographic, cultural, individual, distributional, or Overton pluralism.

**Lesson:** there are several pluralism objectives:

- **distributional pluralism:** match the population distribution;
- **Overton pluralism:** represent the range of reasonable positions;
- **steerable pluralism:** respond according to a selected viewpoint;
- **individual pluralism:** adapt to a person without stereotyping;
- **deliberative pluralism:** expose arguments and unresolved assumptions.

The current geometry contributes a missing axis:

> Does the system preserve the relational organization and resolution of plural judgments?

## 4. What is new in the current program

### 4.1 Relational recovery

The model graph is evaluated against posterior human neighborhood support rather than only pointwise distributions.

### 4.2 Conditional-resolution ladder

The null ladder asks how much relational alignment survives after controlling for:

1. dataset;
2. majority label;
3. entropy;
4. top-two label identity;
5. margin;
6. exact profile.

This converts “model alignment” into a hierarchy of recovered information.

### 4.3 Effective prototype equivalents

Human distributions are compressed into \(K\) cross-fitted prototypes. A model’s recovery is mapped to the prototype curve.

This produces an interpretable external scale:

> The model performs like a human-distribution quantizer with approximately \(K\) effective states.

It is not a claim about literal internal states or mutual information.

### 4.4 Ensemble complementarity

Exact coalition enumeration and Shapley attribution quantify which models add nonredundant relational information.

### 4.5 Calibration as relational intervention

Temperature scaling can improve a selected pointwise scoring rule while changing neighborhoods without improving human-supported topology.

This motivates a broader principle:

> Any post-hoc model intervention should be audited for what relational structure it preserves, gains, or destroys.

## 5. The collective-twin overlap problem

Suppose groups \(A\) and \(B\) answer the same items.

Several forms of overlap are possible.

### 5.1 Marginal overlap

Compare \(p_i^A\) and \(p_i^B\) item by item.

Useful but incomplete.

### 5.2 Neighborhood overlap

Compare group-specific graphs:

\[
W^A,\quad W^B.
\]

This asks whether the groups see the same cases as analogous.

### 5.3 Prototype overlap

Fit group-specific archetypes and ask:

- which prototypes are shared;
- which are group-specific;
- how much mass each group assigns to each region.

### 5.4 Structural overlap without fixed item identity

When groups answer different but related item sets, compare internal geometries through Gromov–Wasserstein or related graph distances.

### 5.5 Missing-region analysis

Define a region that is supported in group \(A\) but poorly represented in group \(B\) or in a model.

This is a stronger and more careful concept than saying a “minority opinion was erased.”

## 6. Major claim boundaries

### Current data can support

- aggregate disagreement shapes;
- item-level probability geometry;
- model relational recovery;
- conditional resolution;
- ensemble complementarity;
- cross-dataset replication;
- demographic overlap on datasets that expose group labels.

### Current data cannot fully support

- stable individual digital twins;
- causal explanations of disagreement;
- demographic attribution in ChaosNLI;
- true polarization without repeated identity or group structure;
- longitudinal dynamics without repeated time points;
- semantic reasons without rationales.

## 7. Research questions worth carrying forward

1. Is disagreement geometry compressible across domains?
2. Do stronger models recover finer conditional structure?
3. Does model scaling increase relational resolution?
4. Does calibration improve likelihood while reducing effective relational resolution?
5. Do ensembles add approximately independent “bits” of pluralistic structure?
6. Which group-specific regions are systematically underrepresented?
7. How many human annotations are required to stabilize geometry?
8. Can a student model preserve an ensemble’s relational geometry?
9. Are cross-domain disagreement archetypes transferable?
10. Can collective-twin overlap predict where a system will fail to represent a population?

## 8. One-sentence synthesis

> Human judgments occupy structured probability and relational spaces; the frontier question is no longer whether people disagree, but which parts of that structure an AI system preserves, compresses, misplaces, or fails to represent.
