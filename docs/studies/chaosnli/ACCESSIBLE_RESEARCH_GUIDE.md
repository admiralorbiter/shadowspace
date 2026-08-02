# An Accessible Guide to the Research: Collective Opinion as a Relational Space

> **One-Sentence Summary**: This research asks whether AI language models organize ambiguous language examples in the same way people do—and develops a new, mathematically rigorous way to compare them when human opinion patterns contain distance ties or uncertainty.

---

## Table of Contents
1. [What Problem Is This Research Trying to Solve?](#1-what-problem-is-this-research-trying-to-solve)
2. [What Dataset Does the Study Use? (Real Text Examples)](#2-what-dataset-does-the-study-use-real-text-examples)
3. [What Does "Relational Space" Mean?](#3-what-does-relational-space-mean)
4. [The Unexpected Problem: Nearest Neighbors Are Not Uniquely Defined](#4-the-unexpected-problem-nearest-neighbors-are-not-uniquely-defined)
5. [The Proposed Solution: Tie-Aware Neighborhoods (Real Calculation Example)](#5-the-proposed-solution-tie-aware-neighborhoods-real-calculation-example)
6. [Three Ways to Compare Tied Neighborhoods](#6-three-ways-to-compare-tied-neighborhoods)
7. [Human Judgment Is Uncertain Too](#7-human-judgment-is-uncertain-too)
8. [The AI Model Experiment and Paired Results](#8-the-ai-model-experiment-and-paired-results)
9. [How Annotation Depth Changes the Map](#9-how-annotation-depth-changes-the-map)
10. [The Phase-Diagram Experiment](#10-the-phase-diagram-experiment)
11. [The Level-1 and Level-2 Distinction](#11-the-level-1-and-level-2-distinction)
12. [The VariErr External Analysis](#12-the-varierr-external-analysis)
13. [Exploratory Text-Space Work](#13-exploratory-text-space-work)
14. [What Is the Main Scientific Contribution?](#14-what-is-the-main-scientific-contribution)
15. [Why Does This Matter for Language Models?](#15-why-does-this-matter-for-language-models)
16. [Practical Downstream Uses](#16-practical-downstream-uses)
17. [Applications Outside Language](#17-applications-outside-language)
18. [What the Research Does Not Prove](#18-what-the-research-does-not-prove)
19. [Important Limitations](#19-important-limitations)
20. [Most Important Future Questions](#20-most-important-future-questions)
21. [Plain-Language Conclusion](#21-plain-language-conclusion)
* [Appendix: Design Evolution](#appendix-how-the-experimental-design-evolved)

---

## Quick Summary (5-minute read)

**What we asked**: Do AI language models organize ambiguous language examples into the same relational patterns that people do?

**What we found**: Human disagreement about ambiguous sentence pairs produces a structured—but noisy and scale-dependent—map of similarity. The index-resolved nearest-neighbor implementation tested here distorts this map silently, because tied distances occur for 72.4% of items and the implementation resolves ties using array storage order (different libraries may use different tie policies). Our tie-aware framework is provably invariant to row order. Under a fully paired comparison against the same simulated human cohorts, the nine evaluated NLI models recover roughly 17–21% of human replicate overlap—substantially below the posterior-predictive human reference of ~0.0755.

**Why it matters**: Evaluation methods that ignore distance ties are non-reproducible across data orderings. Models that match majority labels can still organize the opinion space very differently from human populations.

**What we did not prove**: We do not identify which annotator is correct, why specific items are disputed, or that preserving human opinion neighborhood structure is required for all engineering applications.

---

## 1. What Problem Is This Research Trying to Solve?

Most language model evaluations evaluate **one example at a time**.

In a standard **Natural Language Inference (NLI)** task, a model receives a **Premise** and a **Hypothesis** and must choose between three possible label categories:
- **Entailment**: The premise guarantees the hypothesis is true.
- **Neutral**: The premise does not settle whether the hypothesis is true.
- **Contradiction**: The premise conflicts directly with the hypothesis.

Standard benchmark evaluation reduces human judgments to a single majority label:

```
Premise:    "Two young children in blue jerseys..."
Hypothesis: "Two kids at a ballgame wash their hands."
Single Label: NEUTRAL (Majority vote)
```

However, when 100 people evaluate this exact sentence pair, their judgments look like this:

| Label Category | Human Votes | Percentage |
|---|---|---|
| **Entailment** | 30 votes | 30% |
| **Neutral** | 70 votes | 70% |
| **Contradiction** | 0 votes | 0% |

Reducing this to a single label ("Neutral") discards the fact that **30% of human evaluators saw an entailment relationship**. That variation may reflect genuine ambiguity, implicit context, differing interpretations, annotation error, or a mixture of these.

### The Research Question
Instead of asking *"Does the AI get the majority label right?"*, this project asks:

$$\mathbf{\text{Do AI models preserve the relationships among examples implied by human judgment distributions?}}$$

In plain language: **When humans judge two disputed items similarly, does the AI model also treat those items as near neighbors in the space formed by its output probabilities?**

---

## 2. What Dataset Does the Study Use? (Real Text Examples)

The study uses the **ChaosNLI** dataset (Nie et al., 2020), which re-annotates low-agreement NLI items with 100 human judgments per item across 3,113 total NLI items (1,514 from SNLI, 1,599 from MultiNLI).

Each item $i$ is represented as a probability vector $\mathbf{p}_i = (p_E, p_N, p_C)$ on the 3-class simplex:

$$\mathbf{p}_i = \left( \frac{\text{votes}_E}{100}, \frac{\text{votes}_N}{100}, \frac{\text{votes}_C}{100} \right)$$

### Real Examples Drawn Directly from the ChaosNLI Dataset

#### 1. High-Agreement Clear Consensus Item (`chaosnli_snli_3980085662`)
- **Premise**: *"Two young boys of opposing teams play football, while wearing full protection uniforms and helmets."*
- **Hypothesis**: *"Boys play football."*
- **Human Votes**: **98 Entailment, 2 Neutral, 0 Contradiction** $\rightarrow \mathbf{p} = (0.98, 0.02, 0.00)$
- *Interpretation*: Near-unanimous consensus. Almost everyone agrees that playing football implies boys play football.

#### 2. Two-Way Disagreement Item (`chaosnli_snli_2407214681`)
- **Premise**: *"Two young children in blue jerseys, one with the number 9 and one with the number 2 are standing on wooden steps in a bathroom and washing their hands in a sink."*
- **Hypothesis**: *"Two kids at a ballgame wash their hands."*
- **Human Votes**: **30 Entailment, 70 Neutral, 0 Contradiction** $\rightarrow \mathbf{p} = (0.30, 0.70, 0.00)$
- *Interpretation*: 70% feel "blue jerseys" doesn't guarantee they are at a ballgame (Neutral), while 30% infer from the jerseys that they are kids at a ballgame (Entailment).

#### 3. Near-Even Three-Way Disagreement Item (`chaosnli_snli_3271178748`)
- **Premise**: *"Number 13 kicks a soccer ball towards the goal during children's soccer game."*
- **Hypothesis**: *"A player passing the ball in a soccer game."*
- **Human Votes**: **36 Entailment, 33 Neutral, 31 Contradiction** $\rightarrow \mathbf{p} = (0.36, 0.33, 0.31)$
- *Possible interpretation*: A near-even three-way split. Some may see kicking towards the goal as consistent with passing (Entailment), some may find the action ambiguous (Neutral), and others may read shooting on goal as mutually exclusive with passing (Contradiction).

> [!IMPORTANT]
> **Sample Scope**: ChaosNLI specifically targets items with low initial annotator agreement. The results describe a selected population of disputed NLI examples and should not be assumed to apply uniformly to easy, clear-cut language items.

---

## 3. What Does "Relational Space" Mean?

Imagine plotting all 3,113 items on a map based on distance between vote distributions.

We measure distance between two items $i$ and $j$ using **Hellinger distance**:

$$d_H(\mathbf{p}_i, \mathbf{p}_j) = \sqrt{\frac{1}{2} \sum_{c \in \{E,N,C\}} \left( \sqrt{p_{i,c}} - \sqrt{p_{j,c}} \right)^2}$$

- Item A $(0.30, 0.70, 0.00)$ and Item B $(0.33, 0.67, 0.00)$ have a tiny distance ($d_H = 0.0228$). They are **nearest neighbors** on the human opinion map.
- Item C $(0.98, 0.02, 0.00)$ is far away from Item A and B ($d_H \approx 0.583$).

> *Calculation check for Item C*: $d_H = \frac{1}{\sqrt{2}}\sqrt{(\sqrt{0.30}-\sqrt{0.98})^2 + (\sqrt{0.70}-\sqrt{0.02})^2 + 0^2} \approx \frac{1}{\sqrt{2}}\sqrt{0.1295 + 0.5222} \approx \frac{1}{\sqrt{2}}\sqrt{0.6517} \approx 0.583$

```
[ Human Opinion Map ]
   (Item A: 30E/70N) ----- d_H = 0.0228 ----- (Item B: 33E/67N)
          \                                         /
           \                                       /
            \--- Far --- (Item C: 98E/2N) --- Far
```

When an AI language model predicts logits $z_m$ for these items, its softmax outputs $q_m = \text{softmax}(z_m)$ create a parallel **Model Opinion Map**. 

The research evaluates whether the **neighbors** on the human map remain neighbors on the model map.

---

## 4. The Unexpected Problem: Nearest Neighbors Are Not Uniquely Defined

Because 100 votes are split across 3 categories, all possible vote distributions lie on a finite 100-vote grid. The number of possible 3-class vote profiles is:

$$\binom{100 + 3 - 1}{3 - 1} = \binom{102}{2} = 5,151 \text{ possible profiles}$$

Across 3,113 items in ChaosNLI:
- **1,604 unique profiles** are occupied.
- **70.4% of items** (2,193 items) share their exact vote counts with at least one other item.
- **72.4% of items** (2,254 items) have an **exact distance tie** at the boundary of their top-$k=10$ nearest neighbors!

---

## 5. The Proposed Solution: Tie-Aware Neighborhoods (Real Calculation Example)

Let's look at an **exact real calculation** directly from the ChaosNLI dataset for **Item 0** (`chaosnli_snli_2407214681`):

- **Focal Item 0**: *"Two young children in blue jerseys... wash their hands in a sink."* vs. *"Two kids at a ballgame wash their hands."*  
  **Votes**: $(30\text{ E}, 70\text{ N}, 0\text{ C})$

We want to find the top $k=10$ nearest neighbors for Focal Item 0.

### Step-by-Step Distance Audit from ChaosNLI:
1. **9 candidate items** in the dataset have Hellinger distance strictly less than $0.02284$ ($d_{ij} < 0.02284$).
2. **7 candidate items** are **tied exactly** at Hellinger distance $d_{ij} = 0.02284$ (all 7 having vote counts $33\text{ E}, 67\text{ N}, 0\text{ C}$)!

Here are 4 of those 7 tied real candidate sentence pairs from ChaosNLI:

```
Tied Candidate 62:
Premise:    "Two female workers sit on some steps during work."
Hypothesis: "Two friends sitting on step at their job."
Votes:      (33 E, 67 N, 0 C)  -->  d_H = 0.02284

Tied Candidate 74:
Premise:    "A man with a large power drill standing next to his daughter..."
Hypothesis: "The man and girl are doing some home maintenance."
Votes:      (33 E, 67 N, 0 C)  -->  d_H = 0.02284

Tied Candidate 194:
Premise:    "A man in an orange shirt looking at his cellphone."
Hypothesis: "A man stands around as he looks at his phone."
Votes:      (33 E, 67 N, 0 C)  -->  d_H = 0.02284

Tied Candidate 769:
Premise:    "A mother and her two children sit down to rest."
Hypothesis: "A mother and her daughters are resting."
Votes:      (33 E, 67 N, 0 C)  -->  d_H = 0.02284
```

### The Arbitrary Storage-Order Failure
We need **10 neighbors total**. Since 9 items are strictly closer, we only have **1 remaining slot** ($r_i = 10 - 9 = 1$). But we have **7 tied candidates** ($|B_i| = 7$).

- **The index-resolved NumPy implementation tested here** (`np.argsort` without explicit `kind`): In this run, the implementation selected **Candidate 62**. Because NumPy's default sort is not guaranteed to preserve the original row order among tied values, reordering the rows of the dataset can cause another equally valid tied candidate (Candidate 74, 194, etc.) to be selected instead.
- **Our Tie-Aware Solution**: Assigns each of the 7 tied candidates an exact, reproducible fractional weight:

$$w_{ij} = \frac{r_i}{|B_i|} = \frac{1}{7} \approx 0.14286$$

### Formal Weight Formula
For focal item $i$ and rank $k$:
- $A_i = \{j \neq i : d_{ij} < d_i(k)\}$ (strictly closer items, $w_{ij} = 1.0$)
- $B_i = \{j \neq i : |d_{ij} - d_i(k)| \le \text{atol}\}$ (tied boundary items, $w_{ij} = r_i / |B_i|$)
- $r_i = k - |A_i|$ (remaining slots)

$$w_{ij} = \begin{cases} 1.0, & j \in A_i \\ \frac{r_i}{|B_i|}, & j \in B_i \\ 0.0, & \text{otherwise} \end{cases}$$

This guarantees $\sum_{j \neq i} w_{ij} = k$ strictly holds for every node, and fractional soft overlap $Q_{NX}^{\text{soft}}(k)$ is **100% row-permutation invariant** ($1.0000 \pm 0.0000$).

---

## 6. Three Ways to Compare Tied Neighborhoods

We formalize a three-quantity tie-aware framework ($Q_{\text{strict}} \le Q_{\text{expected}} \le Q_{\text{fuzzy}} \le 1.0$):

```
0.0 ------------ Q_strict ------------ Q_expected ------------ Q_fuzzy ------------ 1.0
                  (Core Only)         (Random Ties)         (Fuzzy Partial)
```

| Formulation | Mathematical Definition | Scientific Interpretation |
|---|---|---|
| **$Q_{\text{strict}}$ (Strict Bound)** | $\frac{1}{Nk} \sum_i \sum_{j \neq i} \mathbf{1}(w_{ij}^A=1)\mathbf{1}(w_{ij}^B=1)$ | Counts only guaranteed common neighbors (excludes boundary ties). |
| **$Q_{\text{expected}}$ (Random Resolution)** | $\frac{1}{Nk} \sum_i \sum_{j \neq i} w_{ij}^A w_{ij}^B$ | Expected collision probability under independent random tie choices. |
| **$Q_{\text{fuzzy}}$ (Fuzzy Partial Membership)** | $\frac{1}{Nk} \sum_i \sum_{j \neq i} \min(w_{ij}^A, w_{ij}^B)$ | Partial set membership overlap (treats weights as fuzzy sets). |

---

## 7. Human Judgment Is Uncertain Too

Even 100 human votes are a finite sample from a latent population distribution $\boldsymbol{\theta}_i$.

To model human variability, we sample latent probabilities $\boldsymbol{\theta}_i \sim \text{Dirichlet}(\mathbf{x}_i + \boldsymbol{\alpha})$ (Jeffreys prior $\alpha=0.5$), then draw **two independent 100-vote human cohorts** $G_{H1}$ and $G_{H2}$.

Across 500 simulated human-human pairs, average fuzzy overlap is:

$$E[Q_{\text{fuzzy}}(G_{H1}, G_{H2})] = 0.07549 \quad (95\% \text{ CI: } [0.07000, 0.08099])$$

> [!NOTE]
> **Why is 0.07549 not 1.0?** Exact 10-nearest-neighbor recovery across 3,113 items is a demanding criterion. Small sampling fluctuations in vote counts shift ranks at the boundary. The value **0.07549 is the posterior-predictive human replicate reference** — the average fuzzy overlap between two independent cohorts of 100 simulated human votes — under this dataset, prior, and distance metric.

---

## 8. The AI Model Experiment and Paired Results

We evaluated 9 benchmark NLI models using our **fully paired estimand**:

$$M_{m,b} = \frac{1}{2} \left[ Q_{\text{fuzzy}}(G_m, G_{H1}^{(s)}) + Q_{\text{fuzzy}}(G_m, G_{H2}^{(s)}) \right], \quad s = b \bmod 500$$
$$\Delta_{m,b} = H_b - M_{m,b}$$

Both model and human scores are evaluated **symmetrically against the exact same posterior human cohorts**.

### Complete Benchmark Results ($k=10$, Hellinger Distance)

| Model Architecture | Paired Score $M_{m,b}$ | Model Gap $\Delta_m$ (vs. Human Replicate Reference) | 95% Joint Bootstrap CI | Replicates $\Delta_m > 0$ | Fixed-ref. focal-bootstrap mean |
|---|---|---|---|---|---|
| **BART-Large** | **0.01572** | **0.05977** | [0.05431, 0.06539] | 1,000 / 1,000 | 0.01867 |
| **RoBERTa-Large** | **0.01415** | **0.06135** | [0.05557, 0.06685] | 1,000 / 1,000 | 0.01821 |
| **XLNet-Large** | **0.01285** | **0.06264** | [0.05711, 0.06846] | 1,000 / 1,000 | 0.01319 |
| **ALBERT-xxLarge** | **0.01124** | **0.06426** | [0.05896, 0.06997] | 1,000 / 1,000 | 0.01074 |
| **BERT-Large** | **0.01029** | **0.06520** | [0.05966, 0.07076] | 1,000 / 1,000 | 0.01059 |
| **RoBERTa-Base** | **0.01007** | **0.06543** | [0.05979, 0.07106] | 1,000 / 1,000 | 0.01129 |
| **XLNet-Base** | **0.00927** | **0.06623** | [0.06069, 0.07175] | 1,000 / 1,000 | 0.00893 |
| **DistilBERT** | **0.00854** | **0.06695** | [0.06124, 0.07261] | 1,000 / 1,000 | 0.00854 |
| **BERT-Base** | **0.00768** | **0.06782** | [0.06235, 0.07356] | 1,000 / 1,000 | 0.00865 |
| **HH100 Reference (Human Replicate Reference)** | **0.07549** | — | **[0.07000, 0.08099]** | — | — |

```
Human Replicate Reference (0.07549)  ========================================|
                                                                           | Gap = 0.05977
BART-Large Paired Score (0.01572)  ========|                               |
BERT-Base Paired Score  (0.00768)  ====|                                   |
Random Chance Null      (0.00354)  =|                                      |
```

### Key Takeaway
Top-performing models (BART-Large at $0.01572$) achieve a raw paired ratio of approximately **20.8%** relative to the posterior-predictive human reference ($0.01572 / 0.07549 \approx 0.208$), or a chance-adjusted ratio of approximately **16.9%** after subtracting the empirical stratified null ($[0.01572 - 0.00354] / [0.07549 - 0.00354] \approx 0.169$). Models are substantially closer to the empirical stratified null ($0.00354$) than to human collective opinion alignment.

---

## Appendix: How the Experimental Design Evolved

> **Note for lay readers**: This appendix documents the historical development of the paired estimand. It is internal documentation useful for understanding the revision history. Most readers can proceed directly to Section 9.

In earlier revisions of this research, model performance was evaluated asymmetrically:
- $H_b = Q(G_{H1}^{(s)}, G_{H2}^{(s)})$ (human vs. posterior cohort, $s = b \bmod 500$)
- $M_{m,b} = Q(G_m, G_{100}^{\text{obs}})$ (model vs. fixed observed graph)

Our current baseline resolves this asymmetry by adopting the **fully paired construction**:

$$M_{m,b} = \frac{1}{2}\left[Q(G_m, G_{H1}^{(s)}) + Q(G_m, G_{H2}^{(s)})\right]$$

This supports a cleaner, directly matched scientific comparison:
> *"Given the exact same simulated human cohorts, the nine evaluated AI language models resemble those cohorts substantially less than the cohorts resemble one another."*


---

## 9. How Annotation Depth Changes the Map

We evaluate plug-in empirical reference similarity $R_{\text{reference}}(n, k) = Q(G_n^{\text{rep}}, G_{100}^{\text{obs}})$ across vote depths $n \in \{3..100\}$ and scales $k \in \{5..100\}$ using a 50-seed simulation (Rust/Rayon, confirmed monotone):

| Votes ($n$) | $k=5$ | $k=10$ | $k=20$ | $k=50$ | $k=100$ |
|---|---|---|---|---|---|
| **3 votes** | $0.0060 \pm 0.0001$ | $0.0109 \pm 0.0002$ | $0.0206 \pm 0.0004$ | $0.0490 \pm 0.0008$ | $0.0940 \pm 0.0014$ |
| **10 votes** | $0.0135 \pm 0.0005$ | $0.0242 \pm 0.0007$ | $0.0449 \pm 0.0009$ | $0.0999 \pm 0.0019$ | $0.1793 \pm 0.0032$ |
| **50 votes** | $0.0474 \pm 0.0020$ | $0.0813 \pm 0.0018$ | $0.1424 \pm 0.0020$ | $0.2769 \pm 0.0033$ | $0.4136 \pm 0.0038$ |
| **100 votes** | $\mathbf{0.0807 \pm 0.0025}$ | $\mathbf{0.1391 \pm 0.0033}$ | $\mathbf{0.2341 \pm 0.0041}$ | $\mathbf{0.4080 \pm 0.0039}$ | $\mathbf{0.5448 \pm 0.0038}$ |

*Each entry is mean \u00b1 SD across 50 independent seeds. Monotonicity confirmed for both means and 95% normal-approximation simulation interval lower bounds ($\bar{x} - 1.96 \times \text{SD}$) across all five k-columns.*

### Two Architectural Regimes
- **Microstructure ($k=5, 10$)**: Highly sensitive to individual vote fluctuations.
- **Mesostructure ($k=50, 100$)**: Broad regional opinion clusters recover smoothly ($0.5448 \pm 0.0038$ at $n=100, k=100$, 50-seed simulation).

---

## 10. The Phase-Diagram Experiment

We simulate synthetic items across Dirichlet concentration regimes ($\boldsymbol{\theta}_i \sim \text{Dirichlet}(\alpha \mathbf{1}_C)$) to determine when boundary ties dominate.

Across 10,500 simulations (100 reps per cell):

| Votes ($n$) | Concentrated ($\alpha=0.1$) | Symmetric ($\alpha=0.5$) | Uniform ($\alpha=1.0$) | Empirical ChaosNLI |
|---|---|---|---|---|
| **100 votes ($k=10$)** | **73.3% ± 3.0%** | 16.6% ± 2.9% | 9.4% ± 2.2% | **72.4%** |

### Theoretical Occupancy
Out of $S = \binom{100+3-1}{3-1} = 5,151$ possible 3-class 100-vote profiles:
- Expected occupied profiles under uniform occupancy: $\approx 2,337$
- **Observed occupied profiles in ChaosNLI**: **1,604**

Human response patterns are highly concentrated in specific recurring regions of the probability simplex. This concentration is consistent with, and likely contributes to, the high observed tie prevalence ($72.4\%$); however, the simulations match only one summary statistic and do not uniquely identify a single generating distribution.

---

## 11. The Level-1 and Level-2 Distinction

The project maintains a strict conceptual separation:

```
[ Level-1: Vote Distribution ]
   50% Entailment, 50% Neutral  (Empirical outcome: WHAT happened)
                |
    +-----------+-----------+
    |                       |
[ Level-2 Cause A ]     [ Level-2 Cause B ]
Lexical Ambiguity       Implicit Context & Presupposition
(WHY it happened)       (WHY it happened)
```

- **Level 1 (Distributional Outcome)**: The empirical vote ratio $(0.50, 0.50, 0.00)$.
- **Level 2 (Underlying Driver)**: The cognitive or linguistic reason (polysemy, scope ambiguity, missing context, guideline vagueness, or annotator error).

Two items can share the exact same Level-1 vote profile for entirely different Level-2 reasons.

---

## 12. The VariErr External Analysis

We matched 500 ChaosNLI items with **VariErr NLI** (Weber-Genzel et al., ACL 2024), containing 7,732 human validity judgments evaluating whether disagreements represent valid variation or error:

- **Matched items**: 500 items across 52 multi-item profiles.
- **Observed within-profile SD**: $0.1060$
- **500,000-Permutation Null Mean SD**: $0.1150$ (executed natively in Rust in $190.5\text{ ms}$)
- **Empirical $p$-value**: **$p = 0.2045$** ($102,248 / 500,000$)

### Scientific Takeaway
The test shows a $7.8\%$ descriptive reduction in SD, but the result is **statistically inconclusive ($p = 0.2045$)**. We find no evidence that Level-1 vote profiles alone predict explanation validity. This reinforces the need for rationale-level data.

---

## 13. Exploratory Text-Space Work

We evaluated operational case-routing categories combining model predictions, text distance, and human support:

| Operational Category | Candidate Edges | Percentage | Recommended Review Action |
|---|---|---|---|
| **Unclassified / Intermediate** | 156,999 | 51.0% | Background candidate pool |
| **Model Artifact Candidate** | 69,838 | 22.7% | High model consensus, low human & text support |
| **Semantic Similarity Divergence** | 67,455 | 21.9% | High model consensus & text similarity, low human support |
| **Human Relation Missed by Models** | 6,835 | 2.2% | High human support & text similarity, low model consensus |
| **Same Opinion, Distinct Language** | 5,743 | 1.9% | High human support, low model & text support |
| **Broadly Shared Relation** | 792 | 0.3% | Consensus reference edge |

### Retrieval Benchmark
A text-space tie-breaker slightly improves heuristic taxonomy retrieval (MAP@10 $+0.00535$, $p \le 0.002$), but pure text embeddings discard opinion neighborhood structure ($Q_{NX}^{\text{soft}} = 0.0041$). Text similarity and opinion similarity are complementary, non-interchangeable structures.

---

## 14. What Is the Main Scientific Contribution?

1. **Diagnosing Storage-Order Instability**: Exposing that conventional fixed-$k$ implementations that resolve ties by index order silently distort neighbor sets for $62.1\%$ of items.
2. **Formalizing Tie-Aware Mathematics**: Proving $Q_{\text{strict}} \le Q_{\text{expected}} \le Q_{\text{fuzzy}} \le 1.0$ and establishing six core theoretical properties (fuzzy self-identity, row-permutation invariance, etc.).
3. **Paired Relational Model Evaluation**: Evaluating whether the nine evaluated models' **output probability distributions** preserve the *relational neighborhood structure of collective human judgment*, demonstrating a raw paired recovery of approximately $20.8\%$ (or $16.9\%$ chance-adjusted using the empirical stratified null) relative to the posterior-predictive human replicate reference.
4. **Multiscale Scale Dependence**: Disentangling volatile local microstructure ($k=5,10$) from stable regional mesostructure ($k=50,100$).

---

## 15. Why Does This Matter for Language Models?

A language model can achieve high majority-label accuracy while constructing a distorted map of human collective judgment.

For example, a model might:
- Predict correct probabilities for isolated items.
- Group examples together based on lexical overlap (e.g., matching words) rather than shared ambiguity profiles.
- Fail to distinguish between clear-cut items and disputed items in its output distributions.

Relational evaluation audits whether language models' **output probability distributions** align with the **collective judgment structure of human populations**.

---

## 16. Practical Downstream Uses

- **Model Evaluation**: Report relational human-opinion recovery alongside accuracy and Brier score.
- **Dataset Auditing**: Detect unstable item clusters, recurring disagreement profiles, and annotation guideline gaps.
- **Active Annotation Budgeting**: Direct new human votes to items whose relational graph position is most uncertain.
- **Human-Review Routing**: Flag model-only clusters for diagnostic review.
- **Relational Training Losses**: Train models using neighborhood-preservation loss terms alongside cross-entropy.

---

## 17. Applications Outside Language

The tie-aware relational methodology applies to any domain with multi-annotator categorical distributions:

- **Medical Imaging**: Radiologist opinion distributions over (Benign, Suspicious, Malignant).
- **Educational Grading**: Teacher scoring distributions over student essays or math solutions.
- **Content Moderation**: Moderator panel votes over (Acceptable, Borderline, Hateful, Severe).
- **Political Surveys**: Public opinion distributions across policy options.

---

## 18. What the Research Does Not Prove

- It does **not** identify which individual human annotator is "correct."
- It does **not** prove why annotators disagreed on specific items (Level-2 rationale drivers).
- It does **not** establish that preserving human opinion neighborhood structure is required for all engineering tasks.
- It measures specifically: **How reproducible are relationships among judgment distributions, and how well do AI models preserve those relationships?**

---

## 19. Important Limitations

1. **Selected Low-Agreement Population**: ChaosNLI targets disputed items; tie rates may be lower in standard corpora.
2. **Pre-2023 Model Set**: Benchmark models reflect BERT/RoBERTa/BART-era architectures.
3. **Plug-in Surface vs. Posterior-Predictive Surface**: Reference surface $R_{\text{reference}}(n, k)$ conditions on observed proportions $\hat{p}_i$ and does not include posterior uncertainty over latent human distributions.
4. **VariErr Power Constraints**: External test is restricted to 52 multi-item profiles ($p=0.2045$).

---

## 20. Most Important Future Questions

1. **Modern LLMs**: Do modern generative LLM ensembles (GPT-4, Claude 3.5, Llama 3) achieve higher relational recovery?
2. **Uncertain Graph Estimation**: Estimating edge-inclusion probabilities $P(j \in N_k(i) \mid \text{votes})$.
3. **Relational Active Learning**: Selecting the next item/vote to minimize graph uncertainty.
4. **Relational Training**: Fine-tuning models directly on $Q_{NX}^{\text{soft}}$ loss objectives.

---

## 21. Plain-Language Conclusion

Most AI evaluations ask whether a model gives the right answer for each example.

This study asks something broader:

$$\mathbf{\text{Does the model organize uncertain examples into the same relational patterns people do?}}$$

The research finds that human disagreement creates a structured—but noisy and scale-dependent—relational space. Conventional fixed-$k$ implementations that resolve ties by index order distort this space because distance ties are ubiquitous ($72.4\%$). Our tie-aware framework represents these ties explicitly and is invariant to file row order.

Under a fully paired experimental design, the nine evaluated benchmark language models recover approximately **17–21%** of human replicate overlap in the opinion relational space. Human disagreement is not merely noise—it may induce a rich neighbor-graph structure that remains a key challenge for AI systems. The tie-aware framework developed here provides a reproducible, storage-order-invariant foundation for studying these structures.
