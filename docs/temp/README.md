# Geometry of Human Disagreement — Knowledge Base

**Status:** Working research knowledge base  
**Prepared:** August 3, 2026  
**Scope:** Human disagreement, collective-judgment geometry, model evaluation, pluralistic alignment, and moonshot applications.

## Purpose

This collection turns the current research program into a reusable intellectual and experimental map.

The central idea is:

> Human disagreement is not merely an error rate or an entropy value. It has shape, neighborhood structure, resolution, and potentially group-specific organization. AI systems can preserve, distort, compress, or erase parts of that structure.

The documents distinguish three kinds of content:

1. **Established research traditions** — reliability theory, psychometrics, cultural consensus, Q methodology, social choice, opinion dynamics, compositional geometry, crowdsourcing, and pluralistic AI.
2. **Current-project evidence** — findings from ChaosNLI classifier, calibration, ensemble, prototype-compression, and local-LLM experiments.
3. **Proposed research directions** — experiments and applications that remain hypotheses until tested.

## Reading order

1. [01 — Geometry of Human Disagreement](01_geometry_of_human_disagreement.md)  
   The conceptual foundation: what disagreement shapes are, what older fields learned, and what is new here.

2. [02 — Mathematical Toolkit and Experiment Ledger](02_math_and_experiment_ledger.md)  
   Definitions, metrics, overlap measures, nulls, uncertainty, and a ledger connecting math to experiments and claim boundaries.

3. [03 — Dataset Landscape and Triangulation Plan](03_dataset_landscape.md)  
   What can be answered using ChaosNLI and public datasets now, what requires new data, and how to combine datasets without pretending they are one population.

4. [04 — Moonshot Application Portfolio](04_moonshot_portfolio.md)  
   Platforms, products, methods, and long-horizon applications ranked by scientific depth, visual impact, feasibility, and relevance to frontier AI labs.

5. [05 — Annotated Research Lineage](05_annotated_research_lineage.md)  
   A curated bibliography organized by intellectual tradition, with the main lesson and the experiment each source suggests.

6. [06 — High-Attention Hooks and Demonstrations](06_high_attention_hooks.md)  
   Flashy-but-deep titles, visual demos, launch artifacts, and claim-safe ways to attract serious attention.

## Current evidence snapshot

The project’s current evidence supports a coherent chain:

- **Calibration versus relational structure:** Across nine NLI classifiers, temperature scaling closed approximately 24.8%–56.6% of the NLL gap while relational gap closure remained below roughly 0.7%. Jensen–Shannon divergence worsened, and graph turnover was approximately 13.4%–31.1%.
- **Conditional-resolution ladder:** On the audited 600-item pilot, stronger classifiers retained small residual relational excess after conditioning on dataset, majority label, entropy, top-two labels, and margin; compact models often retained effectively none.
- **Ensemble complementarity:** The full nine-classifier coalition recovered substantially more pilot relational structure than any single model, with nonzero Shapley contributions from all nine systems.
- **Prototype-equivalent resolution:** On the audited pilot scale, BART-Large was comparable to approximately 5.82 prototype-equivalent states, the best pair to approximately 8.11, the best triplet to approximately 9.78, and all nine models to approximately 12.38.
- **Modern local LLM bridge:** E004 extends the framework to Gemma through log-probability and Monte Carlo estimation.

These findings do **not** yet establish demographic minority erasure. ChaosNLI measures statistical minority interpretations without annotator identity metadata.

## Core claim boundary

This knowledge base supports the study of:

- aggregate vote-distribution geometry;
- item-to-item relational structure;
- model and ensemble recovery;
- conditional resolution;
- group overlap when group labels are available.

It does not, by itself, identify:

- why a person chose a label;
- stable individual belief systems;
- demographic causes of disagreement;
- causal social dynamics;
- literal internal “bits” stored by a model.

## Recommended flagship arc

A focused high-impact program would combine:

1. **PluralityBench / Resolution Cards** — a general evaluation primitive.
2. **Resolution-preserving distillation** — a trainable method that compresses ensemble pluralism into one deployable model.
3. **Collective-geometry overlap** — cross-group or cross-domain comparison using public datasets.
4. **One applied demonstrator** — education misconception geometry is especially promising.
