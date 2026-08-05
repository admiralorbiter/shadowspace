# Phase EDU-2a Offline Analysis: Automated Text Audit & Counterfactual Difference Atlas

**Branch**: `research/education-automated-text-audit`  
**Parent Scientific Commit**: [`3064ab0`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy) (Phase EDU-2a-R1.2c Human Review Activation)  

---

## 1. Overview & Research Motivation

While rule-based screening rubrics assigned uniform scalar scores across initial pilot prompts, the generated recommendation letters exhibit rich variations in sentence structure, lexical framing, accomplishment coverage, and prompt adherence.

The **Automated Text Audit** provides an offline, fine-grained quantitative evaluation framework for the 60 frozen Gemma recommendation letters (`results/education_audit/edu_2a/generations.jsonl`) without running additional GPU inference or disturbing the frozen human-review pipeline.

---

## 2. Key Objectives & Metrics

### A. Structural & Compliance Metrics
* **Word / Sentence / Paragraph Count**: Quantifies compliance with the requested length target (180–220 words).
* **Opening Endorsement Strength**: Evaluates sentence-initial strength markers.
* **Superlative & Hedging Density**: Measures high-distinction vocabulary vs. doubt-raising hedge phrases.

### B. Published Lexical Dimensions
* **Agentic vs. Communal Language**: Quantifies active vs. relational descriptions (based on LABE/LAC and EMNLP 2023 reference-letter literature).
* **Core Competency Categories**: Measures ability, standout, grindstone/effort, leadership, competence, warmth, doubt raisers, and future potential.

### C. Fact Coverage & Unsupported Specificity
* **Verified Fact Retention**: Maps profile accomplishments to explicit/implicit inclusions in generated text.
* **Unsupported Specificity**: Automatically flags invented team sizes, institutional titles, grant amounts, and award names.

### D. Paired Counterfactual Divergence
Computes fine-grained differences within fixed `(profile, prompt, seed)` tuples across identity pairs (`pronoun_masc` $\leftrightarrow$ `pronoun_fem`, `name_masc` $\leftrightarrow$ `name_fem`, `condition` $\leftrightarrow$ `anonymous`):
* **Edit Distance & Alignment**: Token and sentence-level edit distances.
* **Maximum Local Sentence Divergence**: Identifies localized heavy-tailed variations.
* **Metric Deltas**: Difference in agentic, communal, warmth, and leadership scores.

---

## 3. Privacy Safeguard & Rater Blinding

> [!IMPORTANT]
> **Rater Protection Rule**: All condition-linked metrics, heatmap data, and the interactive HTML Counterfactual Difference Atlas will be exported strictly to `private_analysis/automated_text_audit/` (git-ignored) until manual human ratings (Pass 1 & Pass 2) are completed and locked. Sanitized summaries will be published only after review closure.

---

## 4. Component Directory Map

```text
research/education_audit/automated_text_audit/
├── feature_registry.py           # Lexical categories, structure, & fact coverage extractors
├── external_replication.py       # Calibration adapter for public reference-letter datasets
├── paired_difference_analysis.py # Paired counterfactual edit distance & divergence engine
├── visualize_counterfactuals.py   # HTML Counterfactual Difference Atlas dashboard generator
└── sensitivity_simulator.py      # Minimum detectable paired effect size calculator
```
