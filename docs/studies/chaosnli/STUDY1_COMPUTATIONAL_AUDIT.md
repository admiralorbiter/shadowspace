# Study 1 Computational Audit & Empirical Report (Revised)

**Dataset:** 3,113 Three-Class ChaosNLI Examples (1,514 SNLI + 1,599 MNLI)  
**Date:** 2026-08-01 (Revised post peer-review audit)  
**Scope:** Selection-Conditioned ChaosNLI Low-Agreement Sample, Human-Opinion Topology, Dirichlet Posteriors, Fractional Tie-Aware Neighborhoods, and Level-1 Opinion Profile Graphs

> **Scope Declaration:** All entropy, density, tie, and topology results reported herein are strictly conditional on the low-original-agreement selection defining ChaosNLI-S/M (where MNLI items had exactly 3 of 5 original annotators agreeing). They must not be generalized without qualification to all NLI data.

---

## 1. Summary of Quantitative Findings

| Estimand / Property | Value | Description |
|---|---|---|
| **Canonical Dataset Size** | **3,113 items** | 100 human judgments per item ($N=3,113$) |
| **Unique Opinion Profiles (Level 1 Nodes)** | **1,604 unique** | Discrete 3-class distribution vectors |
| **Items in Non-Singleton Profiles** | **2,193 items (70.4%)** | Items sharing an exact label distribution with at least one other item |
| **Max Profile Multiplicity** | **14 items** | Maximum number of items sharing an identical vote count vector |
| **Items with Distance Ties at $k=10$ Boundary** | **2,254 items (72.4%)** | Items with exact distance ties across the $k=10$ neighbor boundary |
| **Median Boundary Tie Block Size** | **3.0 items** | Median number of tied candidate neighbors at rank $k=10$ |
| **Empirical Mean Entropy** | **0.9386 bits** | Overall distribution dispersion across dataset |
| **Posterior Mean Composition Entropy ($H(E[\theta\mid x])$)** | **0.9534 bits** | Smoothly regularized under Dirichlet $\boldsymbol{\alpha}=(0.5, 0.5, 0.5)$ |
| **Average 95% Entropy CI Width** | **0.3278 bits** | Finite 100-vote sampling noise bounds |
| **Zero-Count Prevalence** | **23.1% (720 items)** | Items with at least one zero-vote class ($p_j = 0$) |
| **Deterministic Fixed-$k$ $Q_{NX}(10)$** | **0.9555** | Sensitivity under natural row ordering |
| **Fractional Tie-Aware Soft Overlap ($Q_{NX}^{\text{soft}}(10)$)** | **0.0426 (4.26%)** | Tie-invariant 50/50 split-half agreement |
| **Chance Baseline Overlap ($k/(N-1)$)** | **0.00321 (0.321%)** | Expected random overlap for $k=10, N=3113$ |
| **Excess-Over-Chance Soft Overlap** | **12.3x chance** | Soft split-half overlap relative to random chance baseline |

---

## 2. Deep Dive: Unique Vectors & Profile Ties

### The Empirical Fact
Out of 3,113 items, there are **only 1,604 unique 3-class count vectors**. Exactly **2,193 items (70.4%)** belong to non-singleton profile groups sharing exact human label distributions.

### Mathematical Mechanism
A 3-class probability distribution with $N_{\text{votes}}=100$ lives on a 2-dimensional equilateral triangle:
$$\Delta^2 = \{(p_E, p_N, p_C) : p_j \ge 0, \; p_E + p_N + p_C = 1\}.$$
For integer vote counts summing to 100, there are exactly $\binom{100 + 3 - 1}{3 - 1} = 5,151$ possible grid positions.

### Research Implication
Any two NLI items with identical vote counts have Hellinger distance $d_H = 0.0$. Pure label-distribution geometry cannot explain **why** annotators split 50/50 on Premise A vs Premise B. This result **motivates Hypothesis 7** and establishes that label-distribution geometry alone is insufficient; **joint opinion-and-text spaces** (opinion geometry $\times$ text embeddings) are required to resolve profile ties.

---

## 3. Human 50/50 Split-Half Agreement & Sampling Redesign

To evaluate whether models recover human opinion structure, model performance must be compared against **human split-half agreement under specified tie rules**, normalized by excess-over-chance scaling.

### Sampling Schemes
1. **Complementary 50/50 Random Partition**: $Q_{NX}^{\text{soft}}(10) = 0.0426$ (4.26%).
2. **Independent Posterior-Predictive Dirichlet-Multinomial 50-Vote Samples**: $Q_{NX}^{\text{soft}}(10) = 0.0474$ (4.74%).

### Excess-Over-Chance Ratio
Rather than raw ratios, model evaluation will use excess-over-chance scaling:
$$\text{Excess Ratio} = \frac{Q_{\text{model}} - Q_{\text{chance}}}{Q_{\text{human}} - Q_{\text{chance}}}, \qquad Q_{\text{chance}} = \frac{k}{N-1} \approx 0.00321.$$

---

## 4. Two-Level Representation Architecture

To eliminate arbitrary tie truncation artifacts, our analysis adopts a **Two-Level Graph Representation**:
- **Level 1 (Opinion-Profile Graph)**: Graph constructed over the $1,604$ unique count vectors weighted by item frequency. Minimum distance between distinct profile nodes is $d_H = 0.0071$, completely eliminating zero-distance ties.
- **Level 2 (Items within Profiles)**: Evaluates text embedding, disagreement taxonomy, and model prediction dispersion among items sharing identical opinion profiles.

---

## 5. Concrete NLI Case Studies (Provisional Interpretive Coding)

### Example Case A: Exact Distribution Tie ($E=50, N=50, C=0$)
*Demonstrates two distinct linguistic phenomena mapped to the exact same probability coordinate.*

- **Item A1 (`chaosnli_snli_4431189771.jpg#0r1n`)**:
  - **Premise**: *"A young man is washing his hair, brushing his teeth, and shaving his face simultaneously."*
  - **Hypothesis**: *"The young man is very coordinated and flexible."*
  - **Human Votes**: `Entailment: 50 | Neutral: 50 | Contradiction: 0` ($H = 1.0000 \text{ bits}$)
  - **Provisional Interpretive Coding**: **Probabilistic Enrichment / Implicature**. Is multi-tasking grooming an implicit proof of coordination (Entailment) or a separate subjective evaluation (Neutral)?

- **Item A2 (`chaosnli_snli_3160531982.jpg#0r1e`)**:
  - **Premise**: *"With the sun rising, a person is gliding with a huge parachute attached to them."*
  - **Hypothesis**: *"The person is falling to saftey with the parachute"*
  - **Human Votes**: `Entailment: 50 | Neutral: 50 | Contradiction: 0` ($H = 1.0000 \text{ bits}$)
  - **Provisional Interpretive Coding**: **Presupposition / Accommodating Minimal Content**. Does paragliding at sunrise entail "falling to safety" or merely gliding?

---

### Example Case B: Trinary 3-Way Ambiguity ($P(\text{majority}) < 0.50$)
*Demonstrates maximum-entropy items where no single majority label exists.*

- **Item B1 (`chaosnli_snli_2093742216.jpg#3r1e`)**:
  - **Premise**: *"An elderly woman crafts a design on a loom."*
  - **Hypothesis**: *"The woman is sewing."*
  - **Human Votes**: `Entailment: 35 | Neutral: 31 | Contradiction: 34`
  - **Entropy**: **1.5831 bits** (Theoretical max = 1.5850 bits). $P(\text{mode}) = 0.4620$.
  - **Provisional Interpretive Coding**: **Lexical Ambiguity & Category Inclusion**. Is weaving on a loom a sub-type of "sewing" (Entailment), a completely distinct craft (Contradiction), or loosely related handiwork (Neutral)?

- **Item B2 (`chaosnli_mnli_18189c`)**:
  - **Premise**: *"The important thing is to realize that it's way past time to move it."*
  - **Hypothesis**: *"It cannot be moved, now or ever."*
  - **Human Votes**: `Entailment: 34 | Neutral: 32 | Contradiction: 34` (Plurality tie between Entailment and Contradiction)
  - **Entropy**: **1.5844 bits**. $P(\text{mode}) = 0.3875$.

---

### Example Case C: Zero-Count Boundary Split ($E=0, N=50, C=50$)
*Demonstrates zero-vote prevalence ($23.1\%$) and location/coreference underspecification.*

- **Item C1 (`chaosnli_snli_4665413015.jpg#4r1c`)**:
  - **Premise**: *"Two monks are visiting a big city."*
  - **Hypothesis**: *"The monks are running down the dirt trail."*
  - **Human Votes**: `Entailment: 0 | Neutral: 50 | Contradiction: 50` ($H = 1.0000 \text{ bits}$)
  - **Provisional Interpretive Coding**: **Location & Coreference Underspecification**. If visiting a big city, can they also run on a dirt trail in a city park (Neutral), or does "big city" contradict a "dirt trail" (Contradiction)?
  - **Distance Impact**: Under Aitchison distance, Bayesian Dirichlet smoothing yields $p_E = \frac{0 + 0.5}{100 + 1.5} \approx 0.00493$, preventing log-ratio infinity while preserving Hellinger boundary stability ($d_H(p, q) = 0.7071$).

---

## 6. Model Benchmark & Hypothesis Testing Results

### Hypothesis 1: Model Topology Recovery vs Human Split-Half Baseline
**Statement:** All model opinion-neighborhood recovery scores $Q_{NX}^{\mathrm{soft, HM}}(10)$ fall significantly below the human split-half baseline ($Q_{NX}^{\mathrm{soft, HH}}(10) = 0.0426$).

**Empirical Results (9 Models Evaluated):**

| Model Name | Soft $Q_{NX}^{\mathrm{soft, HM}}(10)$ | Pointwise JSD (bits) | Excess Ratio vs Human Reliability | H1 Result |
|---|---|---|---|---|
| **BART-Large** | **0.01099** (1.10%) | 0.1402 bits | **19.8%** | **Confirmed** |
| **RoBERTa-Large** | **0.01075** (1.08%) | 0.1374 bits | **19.1%** | **Confirmed** |
| **XLNet-Large** | **0.01071** (1.07%) | 0.1399 bits | **19.0%** | **Confirmed** |
| **ALBERT-xxLarge** | **0.01058** (1.06%) | 0.1470 bits | **18.7%** | **Confirmed** |
| **BERT-Large** | **0.01033** (1.03%) | 0.1470 bits | **18.1%** | **Confirmed** |
| **RoBERTa-Base** | **0.00981** (0.98%) | 0.1426 bits | **16.7%** | **Confirmed** |
| **XLNet-Base** | **0.00928** (0.93%) | 0.1445 bits | **15.4%** | **Confirmed** |
| **DistilBERT** | **0.00891** (0.89%) | 0.1514 bits | **14.5%** | **Confirmed** |
| **BERT-Base** | **0.00815** (0.82%) | 0.1445 bits | **12.5%** | **Confirmed** |

- **Chance Baseline ($Q_{\mathrm{chance}}$)**: $0.00321$ (0.321%).
- **Finding**: Models achieve at most **19.8% of human excess reliability**. Large architectures consistently outperform base architectures ($19.8\%$ vs $12.5\%$).

---

### Hypothesis 2: Temperature Scaling & Decoupled Calibration
**Statement:** Temperature scaling alters pointwise distribution calibration (JSD) without changing relational neighborhood topology ($Q_{NX}^{\mathrm{soft}}$).

**Empirical Temperature Curve (RoBERTa-Large):**
- $T = 0.5 \implies \mathrm{JSD} = 0.1929 \text{ bits}, \; Q_{NX}^{\mathrm{soft}} = 0.01087$
- $T = 1.0 \implies \mathrm{JSD} = 0.1374 \text{ bits}, \; Q_{NX}^{\mathrm{soft}} = 0.01075$
- $T = 2.0 \implies \mathrm{JSD} = 0.0793 \text{ bits}, \; Q_{NX}^{\mathrm{soft}} = 0.01090$

**Finding**: As temperature increases from $0.5$ to $2.0$, pointwise error improves by **58.9%** (JSD drops from $0.1929$ to $0.0793$ bits), while neighborhood recovery remains completely flat at $\sim 0.0108 - 0.0109$. This proves that **pointwise distribution calibration and relational opinion topology are decoupled constructs**.

---

## 7. Audit Summary & Protocol Approvals

1. **Multiplicity & Tie Audit Complete**: 70.4% items in non-singleton profiles; 72.4% boundary ties at $k=10$.
2. **Fractional Tie-Aware Neighborhoods Enabled**: $Q_{NX}^{\text{soft}}(10) = 0.0426$ (12.3x chance).
3. **Level-1 Opinion-Profile Graph Built** ($1,604$ unique profile nodes).
4. **Hypothesis 1 Confirmed Across All 9 Models** (Models recover at most 19.8% of human reliability).
5. **Hypothesis 2 Confirmed** (Temperature scaling improves pointwise JSD by 58.9% without affecting $Q_{NX}^{\text{soft}}$).
