# Ambiguity Doppelgängers: What Confidence and Entropy Cannot Tell Us About Human Disagreement in Natural Language Inference

**Anonymous Author(s)**
*Affiliation withheld for review*

## Abstract

Natural language inference systems and evaluation dashboards commonly summarize a distribution over entailment, neutrality, and contradiction using a majority label, the probability assigned to that label, and a scalar uncertainty measure such as Shannon entropy. We show that these summaries are structurally incomplete. Within each majority-label region of the three-class probability simplex, the mapping from a labeled probability distribution to majority label, confidence, and entropy is generically two-to-one: exchanging the two minority probabilities leaves all three summaries unchanged while reversing the direction of disagreement. We call distributions related by this transformation **ambiguity doppelgängers**.

We derive closed-form Hellinger, Fisher–Rao, Jensen–Shannon, and Aitchison distances between such mirror distributions and conduct an empirical census using 3,113 ChaosNLI examples. We identify 1,375 exact doppelgänger pairs involving 1,100 items, or 35.3% of the dataset. These pairs are identical under the selected summaries but have a median Hellinger distance of 0.324. A broader tolerance-based search identifies 1,505 tight and 10,449 loose approximate collisions. A corrected Dirichlet posterior audit finds that these are exact properties of the observed vote counts but are generally uncertain as claims about latent population distributions. Finally, across three frozen NLI models and five calibration conditions, models collapse 10.3%–15.2% and invert 4.5%–6.5% of human minority-orientation contrasts. Calibration changes these rates non-monotonically and does not consistently recover the lost relational structure.

These results establish that common scalar summaries measure the **amount** of disagreement without preserving what the disagreement is **about**. We release a reproducible analysis pipeline and interactive simplex atlas for inspecting these hidden distinctions.

## 1. Introduction

Natural language inference (NLI) asks whether a hypothesis is entailed by, neutral with respect to, or contradicted by a premise. Although these labels are conventionally presented as mutually exclusive ground truth categories, human judgments frequently form distributions rather than unanimous decisions. Previous research has shown that NLI disagreement often persists as more annotations or context are collected and cannot always be dismissed as annotator error. Models that perform well against majority labels may nevertheless fail to reproduce the distribution of plausible human judgments (Pavlick and Kwiatkowski, 2019; Nie et al., 2020).

ChaosNLI made this problem measurable at scale by collecting 100 judgments per example for thousands of items drawn from established NLI benchmarks. Its results demonstrated substantial human disagreement and showed that model errors concentrate among examples with low human agreement. The dataset has consequently supported work on soft-label learning, calibration under disagreement, and distribution-aware evaluation.

Most disagreement-aware evaluation still relies on compressed descriptions of a probability vector. A system may display the majority class, maximum probability, and entropy; a model may be evaluated using confidence calibration, divergence from the human distribution, or agreement with the human ranking. These quantities are useful, but they are not complete representations of labeled disagreement.

Consider the distributions

\[
p=(0.60,0.30,0.10)
\]

and

\[
q=(0.60,0.10,0.30),
\]

where the coordinates correspond to entailment, neutrality, and contradiction. Both have the same majority label, the same confidence, the same entropy, and the same sorted probability vector. Yet they express different collective judgments. In the first, dissent primarily favors neutrality; in the second, it primarily favors contradiction.

This distinction matters because neutrality and contradiction are not interchangeable uncertainty bins. They correspond to different interpretations of the relationship between a premise and hypothesis. Scalar summaries can therefore report that two examples are equivalently ambiguous while concealing that their ambiguity points in opposite semantic directions.

This paper formalizes and measures that failure. We make four contributions:

1. **A summary-collision theorem.** We prove that, within a fixed majority-label region of the three-class simplex, majority label, maximum probability, and entropy generically determine the two minority probabilities only up to permutation.

2. **A geometric characterization.** We derive exact expressions for the Hellinger, Fisher–Rao, Jensen–Shannon, and Aitchison distances between distributions that collide under these summaries.

3. **An empirical census.** We identify exact and approximate ambiguity doppelgängers in the 3,113-example SNLI/MNLI portion of ChaosNLI.

4. **A model-retention audit.** Using held-out predictions from three models under raw and four post-hoc calibration conditions, we measure whether models preserve, attenuate, collapse, amplify, or invert the hidden direction of human disagreement.

The central conclusion is that confidence and entropy can characterize the degree of concentration in a judgment distribution without preserving its labeled orientation. This creates an important difference between **pointwise calibration** and **relational alignment**: a model may improve its probability-level fit while failing to preserve how examples differ from one another in human judgment space.

## 2. Related Work

### 2.1 Human disagreement in NLP

Traditional supervised-learning pipelines frequently aggregate multiple annotations into a single target through majority voting, averaging, or adjudication. This practice can remove systematic variation caused by ambiguity, differing interpretations, or socially situated perspectives. Work on human label variation therefore argues that disagreement affects dataset construction, modeling, and evaluation rather than representing a preprocessing nuisance alone (Plank, 2022).

Pavlick and Kwiatkowski (2019) showed that disagreements about textual inference often persist under additional annotations and contextual information. They further found a mismatch between model uncertainty and human uncertainty and argued that NLI evaluation should assess full human-judgment distributions. ChaosNLI extended this agenda by collecting dense annotations for SNLI, MNLI, and abductive NLI items and showing that models struggle to recover collective human opinions.

Subsequent research has examined the causes of NLI disagreement. Jiang and de Marneffe (2022) developed a taxonomy including uncertainty about sentence meaning, annotator tendencies, and task artifacts. Their findings reinforce the idea that a distribution over labels may encode multiple qualitatively different processes and interpretations.

More broadly, disagreement-aware and perspectivist approaches seek to preserve either population-level label distributions or annotator-specific perspectives. Multi-annotator and soft-label approaches can outperform or complement majority-label learning, while recent perspectivist work cautions that the interpretation of disagreement depends on assumptions about annotators, tasks, and legitimate perspectives.

Our work focuses on population-level label distributions rather than individual annotator identities. It asks a prior representational question: even when the complete population distribution is available, what information disappears when it is reduced to familiar summaries?

### 2.2 Calibration and human judgment distributions

Calibration conventionally asks whether predictions issued with confidence $c$ are correct approximately $c$ of the time. Temperature scaling and related post-hoc transformations have become standard techniques for improving neural-network calibration (Guo et al., 2017).

The extension of calibration to inherently disputed labels is less straightforward. Wang et al. (2022) argued that calibrated NLI models can approximate human disagreement distributions according to pointwise divergence and accuracy measures. Baan et al. (2022), however, showed that calibration against a human majority class is theoretically problematic when humans disagree and proposed distribution-sensitive alternatives involving class frequencies, rankings, and entropy.

Our analysis complements this debate by distinguishing pointwise distribution fit from pairwise relational preservation. Entropy and confidence may be correctly matched while label-specific orientation remains absent or reversed. Similarly, a calibration transformation may improve one distributional objective without preserving contrasts among examples.

### 2.3 Geometry of probability distributions

Probability vectors lie on a simplex rather than in an unconstrained Euclidean space. Aitchison’s work on compositional data established that proportions should often be analyzed through log-ratio geometry because the fixed-sum constraint creates dependencies among coordinates.

We use several complementary distances. Hellinger and Fisher–Rao geometry arise naturally through square-root embeddings of categorical distributions. Jensen–Shannon distance provides a bounded, symmetric information-theoretic metric; the square root of Jensen–Shannon divergence is a metric on probability distributions (Endres and Schindelin, 2003). Aitchison distance captures log-ratio separation in the simplex interior.

Rather than arguing that one metric is uniquely correct, we show that the collision is visible under multiple geometries. The information loss originates in the summary map, not in a particular choice of distance.

## 3. The Ambiguity Doppelgänger Theorem

### 3.1 Parameterization

Let

\[
p=(p_1,p_2,p_3)\in\Delta^2
\]

be a three-class probability distribution. Fix a designated majority class with probability

\[
m=\max_i p_i,\qquad m\geq \frac13.
\]

The total probability allocated to the other two classes is $1-m$. We parameterize the relative allocation between those minority classes using

\[
\delta=
\frac{p_A-p_B}{p_A+p_B},
\]

where $A$ and $B$ are the two non-majority classes in a fixed label order.

This gives the mirror distributions

\[
p^{+}(m,\delta)=
\left(
m,
\frac{(1-m)(1+\delta)}{2},
\frac{(1-m)(1-\delta)}{2}
\right)
\]

and

\[
p^{-}(m,\delta)=
\left(
m,
\frac{(1-m)(1-\delta)}{2},
\frac{(1-m)(1+\delta)}{2}
\right).
\]

The sign of $\delta$ records which minority label receives more probability. Its magnitude records the strength of the imbalance.

For $m\geq 1/2$, every $|\delta|\leq 1$ leaves the designated class as a majority. For $1/3\leq m < 1/2$, validity additionally requires

\[
|\delta|
\leq
\frac{3m-1}{1-m}.
\]

### 3.2 Entropy invariance

Let

\[
h_2(x)=-x\log_2x-(1-x)\log_2(1-x)
\]

denote binary entropy. The entropy of either mirror distribution is

\[
H(m,\delta)
=
h_2(m)
+
(1-m)
h_2\left(\frac{1+\delta}{2}\right).
\]

Because binary entropy is symmetric around $1/2$,

\[
h_2\left(\frac{1+\delta}{2}\right)
=
h_2\left(\frac{1-\delta}{2}\right),
\]

and therefore

\[
H(m,\delta)=H(m,-\delta).
\]

The maximum probability and sorted probability vector are also invariant under $\delta\mapsto -\delta$.

### 3.3 Minority-Swap Collision Theorem

**Theorem 1 — Minority-Swap Collision.**
Within a fixed majority-label sector of the three-class simplex, the summary map

\[
\mathcal{S}(p)=
\left(
\operatorname{argmax}(p),
\max(p),
H(p)
\right)
\]

is generically two-to-one on interior distributions with unequal minority probabilities. In particular,

\[
\mathcal{S}\left(p^{+}(m,\delta)\right)
=
\mathcal{S}\left(p^{-}(m,\delta)\right)
\]

for every valid $\delta$, while

\[
p^{+}(m,\delta)\neq p^{-}(m,\delta)
\]

whenever $\delta\neq 0$.

**Proof sketch.** Fixing the majority label and its probability $m$ fixes the total minority mass $1-m$. Entropy then determines the unordered pair of minority probabilities but is invariant to their permutation. For unequal minority probabilities, there are exactly two assignments of this pair to the two labeled minority coordinates. These assignments correspond to $\delta$ and $-\delta$. The equality collapses to one point at $\delta=0$. $\blacksquare$

The result should not be interpreted as a deficiency of entropy. Entropy correctly measures concentration while being deliberately invariant to label permutations. The problem occurs when permutation-invariant uncertainty is used as though it fully described a labeled judgment distribution.

### 3.4 Exact geometric separation

The Bhattacharyya coefficient of a mirror pair is

\[
BC
=
m+(1-m)\sqrt{1-\delta^2}.
\]

The corresponding Hellinger distance is

\[
d_H
=
\sqrt{
1-m-(1-m)\sqrt{1-\delta^2}
}.
\]

Under the factor-two Fisher–Rao convention used in this study,

\[
d_{FR}
=
2\arccos
\left[
m+(1-m)\sqrt{1-\delta^2}
\right].
\]

The Jensen–Shannon divergence in bits is

\[
JS
=
(1-m)
\left[
1-
h_2\left(\frac{1+\delta}{2}\right)
\right],
\]

with Jensen–Shannon distance

\[
d_{JS}=\sqrt{JS}.
\]

For interior distributions, Aitchison distance is

\[
d_A
=
\sqrt{2}
\left|
\log
\frac{1+\delta}{1-\delta}
\right|.
\]

All four distances equal zero at $\delta=0$ and increase as the minority split becomes more asymmetric. A dashboard collision can therefore conceal arbitrarily pronounced minority orientation up to the restrictions imposed by $m$.

The implementation validated these identities numerically over a 6,565-point $(m,\delta)$ surface.

## 4. Data and Methods

### 4.1 Human judgment data

We analyze the 3,113 SNLI and MNLI examples in ChaosNLI, each represented by vote counts and normalized probabilities for entailment, neutrality, and contradiction. ChaosNLI was created to study collective human opinions through dense annotation rather than majority labels alone.

The frozen canonical table contains one row per item, with a unique identifier, source dataset, premise and hypothesis text, three vote counts, three probabilities, Shannon entropy, and the observed majority label. Preflight checks enforce finite probabilities in $[0,1]$, unit probability sums, nonnegative integer counts, majority-label consistency, and agreement between stored and recomputed entropy. The final input contains 3,113 unique items.

### 4.2 Exact collision census

For each item, we record:

* The majority label and majority vote count.
* The lower and higher minority vote counts after sorting.
* The label receiving the higher minority count.
* The signed minority orientation $\delta$.

Two items form a **strict doppelgänger pair** when they have:

1. The same majority label.
2. The same majority count.
3. The same unordered pair of minority counts.
4. Unequal minority counts.
5. Different labels receiving the higher minority count.

This definition detects exact empirical instances of the theorem at the vote-count level. Items with equal minority counts are excluded because $\delta=0$ and the mirror distributions coincide.

We report pair counts, unique participating items, majority-class composition, source composition, and geometric distances. Source categories—within SNLI, within MNLI, and cross-source—are mutually exclusive.

### 4.3 Approximate collisions

Exact count permutations may underestimate the practical prevalence of summary collisions. We therefore search all same-majority item pairs with opposite orientation signs under three nested tolerance regimes:

\[
\begin{array}{lll}
\text{Tight:}
&
|\Delta \operatorname{conf}|\leq 0.005,
&
|\Delta H|\leq 0.01,
\\
\text{Standard:}
&
|\Delta \operatorname{conf}|\leq 0.01,
&
|\Delta H|\leq 0.02,
\\
\text{Loose:}
&
|\Delta \operatorname{conf}|\leq 0.02,
&
|\Delta H|\leq 0.05.
\end{array}
\]

For each candidate pair we compute

\[
d_{\mathrm{summary}}
=
\sqrt{
(\Delta\operatorname{conf})^2+
(\Delta H)^2
}
\]

and the full-distribution distances. We additionally identify a Pareto frontier in which a pair is nondominated if no other pair has both a smaller summary discrepancy and a larger Hellinger distance.

### 4.4 Posterior uncertainty audit

The exact census treats observed vote proportions as the target distributions. To assess uncertainty arising from finite annotation samples, we place an independent Jeffreys-style Dirichlet posterior on each endpoint:

\[
\theta_i
\sim
\operatorname{Dirichlet}(c_i+0.5),
\]

where $c_i$ is the vector of observed label counts. We draw 2,000 samples per endpoint using deterministic SHA-256-derived pair seeds.

For each paired posterior draw, we evaluate a strict joint event requiring:

1. Both endpoints retain the original observed majority class ($M_0$).
2. Their minority orientations, computed relative to the fixed $M_0$ coordinate system, remain opposite.
3. Their majority probabilities differ by no more than 0.01.
4. Their entropies differ by no more than 0.02 bits.

The joint probability is

\[
P_{\mathrm{joint}}
=
P(
\text{both retain }M_0
\land
\text{opposite orientation}
\land
\text{tight summary}
).
\]

Pairs are assigned descriptive stability categories using a frozen rubric: robust for $P_{\mathrm{joint}}\geq 0.70$, probable for $P_{\mathrm{joint}}\geq 0.40$, uncertain under weaker evidence, and point-estimate-only otherwise. These labels are descriptive sensitivity categories rather than null-hypothesis significance tests.

### 4.5 Frozen model-retention audit

We use 9,339 held-out prediction records: 3,113 items for each of three models—RoBERTa-large, BART-large, and ALBERT-xxlarge. Each record contains raw probabilities and four calibrated variants:

* T1: scalar temperature scaling.
* T2: diagonal mapping in isometric log-ratio coordinates.
* T3: affine isometric log-ratio mapping.
* T4: nonlinear isometric log-ratio mapping.

The preflight report confirms complete coverage of all 3,113 items for all three models.

For each strict human pair, model, and tier, we compute model orientation relative to the **human pair’s majority class**. Let

\[
\Delta_H
=
\delta_{H,A}-\delta_{H,B}
\]

be the human orientation contrast and

\[
\Delta_M
=
\delta_{M,A}-\delta_{M,B}
\]

be the corresponding model contrast. We define retention ratio

\[
R=\frac{\Delta_M}{\Delta_H}.
\]

The descriptive categories are:

\[
\begin{array}{ll}
R < -0.10 & \text{inverted},\\
|R|\leq 0.10 & \text{collapsed},\\
0.10 < R < 0.50 & \text{attenuated},\\
0.50\leq R\leq 1.50 & \text{preserved},\\
R > 1.50 & \text{amplified}.
\end{array}
\]

This produces

\[
1{,}375
\times
3
\times
5
=
20{,}625
\]

pair-model-tier observations.

## 5. Results

### 5.1 Exact collisions are widespread in the observed distributions

We identify 257 strict collision groups containing 1,375 doppelgänger pairs. These pairs involve 1,100 unique items, representing 35.3% of the 3,113-item dataset.

| Statistic | Result |
| :--- | ---: |
| Collision groups | 257 |
| Exact pairs | 1,375 |
| Participating items | 1,100 |
| Participating-item rate | 35.3% |
| Median Hellinger distance | 0.324 |
| Mean Hellinger distance | 0.342 |
| Maximum Hellinger distance | 0.700 |

The median distance of 0.324 shows that the collisions are not dominated by nearly symmetric distributions. Many pairs are meaningfully separated in the full simplex despite being identical under majority label, confidence, and entropy.

Neutral-majority examples account for 1,156 pairs, or 84.1%. Entailment-majority examples account for 181 pairs, and contradiction-majority examples for 38. This concentration suggests that hidden orientation is particularly important when neutrality is the dominant collective judgment. In such examples, the discarded distinction often concerns whether residual belief leans toward entailment or contradiction. Establishing the linguistic causes of that pattern requires targeted textual analysis and is left for future work.

The corrected source breakdown is:

| Pair source | Count | Percentage |
| :--- | ---: | ---: |
| Within SNLI | 780 | 56.7% |
| Within MNLI | 181 | 13.2% |
| Cross-source | 414 | 30.1% |
| **Total** | **1,375** | **100.0%** |

The phenomenon therefore occurs in both source datasets. Cross-source pairs demonstrate recurrence of the same distributional pattern; they do not imply that the paired sentences are semantically equivalent.

### 5.2 Approximate collisions form a larger continuum

The approximate search identifies:

| Tolerance | Candidate pairs |
| :--- | ---: |
| Tight | 1,505 |
| Standard | 2,022 |
| Loose | 10,449 |

These sets are nested and include exact collisions. The increase from 1,375 exact pairs to 1,505 tight pairs indicates that near-indistinguishable summary states occur beyond exact count permutations. Under the loose tolerance, more than ten thousand pairs have the same majority label and opposite minority orientation while remaining close in confidence and entropy.

The Pareto frontier isolates especially striking cases: pairs for which no alternative simultaneously produces a smaller dashboard discrepancy and a larger Hellinger separation. These cases provide natural candidates for qualitative analysis and interactive explanation.

### 5.3 The posterior-stability hypothesis is not supported

The posterior analysis materially changes how the prevalence result should be interpreted.

Under the strict joint persistence criterion:

| Stability category | Count | Percentage |
| :--- | ---: | ---: |
| Robust | 0 | 0.0% |
| Probable | 0 | 0.0% |
| Uncertain | 1,305 | 94.9% |
| Point-estimate-only | 70 | 5.1% |

Thus, the initial hypothesis that exact collisions would remain robust under annotation uncertainty is **not supported**. The empirical vote tables contain exact doppelgängers, but none of the pairs reaches the frozen probable or robust threshold when majority retention, opposite orientation, and tight summary proximity must occur simultaneously.

This negative result does not weaken the mathematical theorem. Nor does it erase the descriptive finding that observed summaries collide. Instead, it imposes an important distinction between three claims:

1. **Theoretical claim:** the summary map necessarily discards minority orientation.
2. **Observed-data claim:** exact and approximate collisions are widespread in the recorded ChaosNLI votes.
3. **Population claim:** the latent human judgment distributions form exact doppelgänger pairs.

The first two are supported. The third is not established at the pair level under the present annotation budget and strict posterior criterion.

### 5.4 Models partially preserve hidden orientation

Across the 15 model-tier combinations, orientation-sign accuracy ranges from approximately 88.0% to 89.7%. Models therefore usually point in the same pairwise direction as the observed human contrast.

However, correct sign alone does not imply faithful preservation. Across tiers:

* Collapse rates range from 10.3% to 15.2%.
* Inversion rates range from 4.5% to 6.5%.
* Preservation rates range from 51.2% to 67.4%.
* Spearman correlations between human and model pairwise Hellinger distances range from approximately 0.59 to 0.69.

These results indicate partial relational alignment rather than either complete success or complete failure.

Table 4 compares raw and T4 behavior:

| Model | Tier | Collapsed | Inverted | Preserved |
| :--- | :--- | ---: | ---: | ---: |
| RoBERTa-large | Raw | 12.2% | 5.0% | 66.0% |
| RoBERTa-large | T4 | 10.9% | 5.4% | 63.8% |
| BART-large | Raw | 11.6% | 6.0% | 67.4% |
| BART-large | T4 | 10.3% | 6.4% | 65.9% |
| ALBERT-xxlarge | Raw | 12.1% | 6.5% | 62.4% |
| ALBERT-xxlarge | T4 | 12.7% | 5.3% | 61.0% |

Raw outputs exhibit the highest preservation rate for each of the three models. T1–T3 frequently reduce retention, while T4 partially recovers it. For example, RoBERTa falls from 66.0% preservation under raw predictions to 54.0% at T2 before recovering to 63.8% at T4. BART falls from 67.4% to 57.2% at T2 and recovers to 65.9%. ALBERT falls from 62.4% to 51.2% and recovers to 61.0%.

Calibration therefore has a **non-monotonic** relationship with disagreement retention. More flexible mappings recover some of the relational structure lost by simpler mappings, but no tier eliminates collapse or inversion.

## 6. Discussion

### 6.1 Entropy measures how much, not which way

The theorem identifies a precise limitation of entropy-based ambiguity summaries. Entropy is invariant to class-label permutations. That property is desirable when the goal is to measure concentration without privileging coordinates. It becomes a liability when the coordinates have distinct meanings and the summary is treated as a complete account of disagreement.

In NLI, the two minority classes are semantically consequential. A neutral-majority item with residual entailment support differs from one with residual contradiction support, even when both have identical entropy. The hidden sign of $\delta$ records this distinction.

A practical interface need not abandon entropy. It can supplement it with the complete vector or a signed minority-orientation statistic:

\[
\delta
=
\frac{p_A-p_B}{p_A+p_B}.
\]

Together, $m$, $H$, and the sign of $\delta$ distinguish the two mirror branches. Displaying all three class probabilities remains the most direct solution.

### 6.2 Pointwise calibration is not relational alignment

Calibration is generally evaluated point by point or through aggregates of pointwise errors. Our results show why this may be insufficient for disagreement-aware applications.

Suppose a calibration transformation improves the average divergence between model and human distributions. It may still move two related examples toward one another, reverse their ordering along a labeled direction, or distort their distance. These failures are relational: they concern how differences between examples are preserved.

This helps reconcile apparently conflicting findings in the calibration literature. A calibrated model can become more similar to human distributions under conventional divergences while still failing to preserve all structure encoded in those distributions. Likewise, the persistence of collapse and inversion does not prove that calibration is ineffective; it shows that relational preservation is an additional objective that should be measured directly. This complements both work proposing calibrated networks for human disagreement and work questioning majority-based calibration under disagreement.

### 6.3 The corrected posterior result strengthens the paper

The absence of robust posterior pairs might initially appear disappointing. We argue that it improves the scientific contribution.

A less careful analysis could report that more than one third of ChaosNLI has stable population-level doppelgängers. The corrected audit demonstrates that this conclusion is not warranted. Exact equality in empirical summaries is partly enabled by discretized annotation counts, and strict pairwise persistence is difficult to establish even with dense annotation.

The appropriate claim is therefore structural and descriptive rather than population-prevalence based:

> The summary representation is provably non-identifying, and the observed ChaosNLI annotations frequently occupy its collision classes.

This claim is both meaningful and supported. The posterior result also motivates annotation-budget research: how many annotations are required to distinguish two nearby summary states while resolving the direction of their minority mass?

### 6.4 Toward disagreement-aware evaluation

Recent perspectivist evaluation work emphasizes that no single aggregate metric captures all relevant relationships among items, annotators, users, and perspectives. Shared tasks on learning with disagreement have likewise begun moving beyond ordinary cross-entropy toward complementary population- and annotator-level evaluation.

The present work adds a geometric diagnostic at the population-distribution level. A disagreement-aware evaluation suite could include:

* Pointwise divergence from the human distribution.
* Majority-label accuracy.
* Entropy or concentration error.
* Orientation-sign accuracy.
* Pairwise orientation-retention error.
* Neighborhood preservation in the simplex.
* Collapse and inversion rates.
* Calibration measures defined against distributions rather than majority labels.

Such a suite would separate several questions that are currently conflated: whether a model predicts the correct winner, the correct amount of uncertainty, the correct alternative interpretation, and the correct relationships among examples.

## 7. Limitations

First, the exact two-to-one theorem is developed for three classes. For larger label sets, the fibers of permutation-invariant summaries can contain more than two labeled distributions and may be higher-dimensional. Extending the analysis to $K>3$ is a promising direction but is not proved here.

Second, this study analyzes one benchmark domain. ChaosNLI is unusually well suited to the question because it contains dense human judgments, but the observed prevalence of collisions should not be generalized to other NLP tasks without replication.

Third, pair observations are not independent. The 1,375 pairs involve 1,100 items, and individual items may participate in multiple pairs within a collision group. Reported percentages are descriptive census quantities, not independent-sample estimates. Future inferential comparisons should resample at the collision-group or connected-component level.

Fourth, the posterior categories depend on frozen tolerances and descriptive thresholds. The result that no pair is robust or probable is specific to the simultaneous joint event and rubric used here. Alternative scientific questions—such as posterior probability of opposite orientation without requiring near-identical entropy in the same draw—would produce different quantities.

Fifth, the model audit includes three encoder-style NLI models and four previously fitted post-hoc mapping families. It does not establish a universal impossibility result for calibration, modern generative models, or models explicitly trained to preserve relational disagreement structure.

Sixth, ChaosNLI provides population-level vote distributions but does not, in the analysis used here, support demographic or longitudinal interpretation of the annotators. Label distributions should not be assumed to represent stable social groups or coherent individual perspectives.

Finally, cross-source doppelgängers share distributional summaries, not necessarily linguistic content or reasoning. Qualitative claims about why particular pairs disagree require manual analysis using established taxonomies of NLI disagreement.

## 8. Ethical Considerations

Preserving disagreement can improve transparency, but distributions should not be treated as neutral or exhaustive representations of a population. Annotation pools reflect recruitment practices, task framing, compensation, language background, and access. Aggregated distributions may hide minority annotators even when they preserve minority labels.

The orientation statistic introduced here describes the allocation of label mass. It does not explain why annotators selected those labels, whether the disagreement is caused by ambiguity or error, or whether each interpretation should be treated as equally legitimate.

The interactive atlas should therefore be used as a diagnostic and educational tool rather than as evidence about annotator identities or motivations. Textual examples may also contain sensitive or biased benchmark content inherited from SNLI and MNLI.

## 9. Reproducibility

The study is frozen on the `research/ambiguity-doppelganger-atlas` branch. The release includes:

* Canonical input hashes.
* Strict and approximate pair tables.
* The theory surface.
* Posterior-stability results.
* Model-retention records and summaries.
* An interactive standalone atlas.
* A SHA-256 manifest distinguishing Git blobs from external input files.
* A verifier that checks the bound Git commit and committed object bytes.

The final branch commit is `8408213654178f5e89c3bd758ff88c9d1d49a8a9`. The manifest records the analysis source commit, Python version, file source types, sizes, and SHA-256 values for 19 artifacts.

## 10. Conclusion

We introduced ambiguity doppelgängers: labeled probability distributions that share a majority class, confidence, and entropy while assigning disagreement to different alternatives. In the three-class simplex, these collisions are not accidental. They arise from a generically two-to-one summary map that discards the sign of minority orientation.

The observed ChaosNLI distributions contain 1,375 exact doppelgänger pairs involving 35.3% of items, along with thousands of approximate collisions. These results show that the information loss is practically visible, not merely a pathological edge case. At the same time, a corrected posterior audit demonstrates that exact pair-level persistence should not be overstated as a property of latent population distributions.

Frozen model predictions preserve the direction of human disagreement in most pairs but regularly attenuate, collapse, or invert it. Post-hoc calibration changes this behavior non-monotonically and does not consistently recover relational structure.

The broader lesson is straightforward:

> A model or dashboard may correctly report which class wins and how uncertain the judgment is while still failing to represent what the uncertainty is about.

Disagreement-aware NLP should therefore evaluate and visualize the full labeled geometry of judgment distributions rather than relying exclusively on majority labels, confidence, entropy, or other permutation-invariant summaries.

## References

J. Aitchison. 1982. The Statistical Analysis of Compositional Data. *Journal of the Royal Statistical Society: Series B*, 44(2):139–160. DOI: 10.1111/j.2517-6161.1982.tb01195.x.

Joris Baan, Wilker Aziz, Barbara Plank, and Raquel Fernandez. 2022. Stop Measuring Calibration When Humans Disagree. In *Proceedings of EMNLP 2022*, pages 1892–1915. DOI: 10.18653/v1/2022.emnlp-main.124.

Aida Mostafazadeh Davani, Mark Díaz, and Vinodkumar Prabhakaran. 2022. Dealing with Disagreements: Looking Beyond the Majority Vote in Subjective Annotations. *Transactions of the Association for Computational Linguistics*, 10:92–110. DOI: 10.1162/tacl_a_00449.

Dominik M. Endres and Johannes E. Schindelin. 2003. A New Metric for Probability Distributions. *IEEE Transactions on Information Theory*, 49(7):1858–1860. DOI: 10.1109/TIT.2003.813506.

Eve Fleisig, Su Lin Blodgett, Dan Klein, and Zeerak Talat. 2024. The Perspectivist Paradigm Shift: Assumptions and Challenges of Capturing Human Labels. In *Proceedings of NAACL 2024*, pages 2279–2292. DOI: 10.18653/v1/2024.naacl-long.126.

Tommaso Fornaciari, Alexandra Uma, Silviu Paun, Barbara Plank, Dirk Hovy, and Massimo Poesio. 2021. Beyond Black & White: Leveraging Annotator Disagreement via Soft-Label Multi-Task Learning. In *Proceedings of NAACL 2021*, pages 2591–2597. DOI: 10.18653/v1/2021.naacl-main.204.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On Calibration of Modern Neural Networks. In *Proceedings of ICML 2017*, pages 1321–1330.

Nan-Jiang Jiang and Marie-Catherine de Marneffe. 2022. Investigating Reasons for Disagreement in Natural Language Inference. *Transactions of the Association for Computational Linguistics*, 10:1357–1374. DOI: 10.1162/tacl_a_00523.

Elisa Leonardelli, Silvia Casola, Siyao Peng, Giulia Rizzi, Valerio Basile, Elisabetta Fersini, Diego Frassinelli, Hyewon Jang, Maja Pavlovic, Barbara Plank, and Massimo Poesio. 2025. LeWiDi-2025 at NLPerspectives: Third Edition of the Learning with Disagreements Shared Task. In *Proceedings of the Fourth Workshop on Perspectivist Approaches to NLP*. DOI: 10.18653/v1/2025.nlperspectives-1.16.

Soda Marem Lo, Silvia Casola, Erhan Sezerer, Valerio Basile, Franco Sansonetti, Antonio Uva, and Davide Bernardi. 2025. PERSEVAL: A Framework for Perspectivist Classification Evaluation. In *Proceedings of EMNLP 2025*, pages 22334–22359. DOI: 10.18653/v1/2025.emnlp-main.1137.

Yixin Nie, Xiang Zhou, and Mohit Bansal. 2020. What Can We Learn from Collective Human Opinions on Natural Language Inference Data? In *Proceedings of EMNLP 2020*, pages 9131–9143. DOI: 10.18653/v1/2020.emnlp-main.734.

Ellie Pavlick and Tom Kwiatkowski. 2019. Inherent Disagreements in Human Textual Inferences. *Transactions of the Association for Computational Linguistics*, 7:677–694. DOI: 10.1162/tacl_a_00293.

Barbara Plank. 2022. The “Problem” of Human Label Variation: On Ground Truth in Data, Modeling and Evaluation. In *Proceedings of EMNLP 2022*, pages 10671–10682. DOI: 10.18653/v1/2022.emnlp-main.731.

Yuxia Wang, Minghan Wang, Yimeng Chen, Shimin Tao, Jiaxin Guo, Chang Su, Min Zhang, and Hao Yang. 2022. Capture Human Disagreement Distributions by Calibrated Networks for Natural Language Inference. In *Findings of ACL 2022*, pages 1524–1535. DOI: 10.18653/v1/2022.findings-acl.120.

Tharindu Cyril Weerasooriya, Alexander Ororbia, Raj Bhensadadia, Ashiqur KhudaBukhsh, and Christopher M. Homan. 2023. Disagreement Matters: Preserving Label Diversity by Jointly Modeling Item and Annotator Label Distributions with DisCo. In *Findings of ACL 2023*, pages 4679–4695. DOI: 10.18653/v1/2023.findings-acl.287.
