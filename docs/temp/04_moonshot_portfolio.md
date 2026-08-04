# Moonshot Application Portfolio

## Selection rule

A serious moonshot should have four properties:

1. **Immediate hook:** understandable in one sentence.
2. **Deep primitive:** introduces a reusable scientific or engineering object.
3. **Current feasibility:** can be prototyped using existing data and local models.
4. **Expansion path:** becomes more powerful with new domains or data.

## Tier 1 — Closest and strongest

## 1. PluralityBench and Resolution Cards

### Hook

> Accuracy tells you whether the model chose the answer. Resolution tells you how much of the human disagreement space it can actually see.

### Scientific core

Evaluate models on:

- relational recovery;
- conditional-null survival;
- prototype-equivalent states;
- effective bits;
- calibration-induced structure change;
- ensemble contribution;
- missing-region recall.

### Flashy artifact

An interactive leaderboard where a model’s “vision” of human disagreement visibly expands from 3 states to 6 to 12.

### Current feasibility

Very high. The Rust engine and core metrics already exist.

### Frontier-lab relevance

High. This is a new evaluation primitive for pluralistic alignment, reward-model auditing, and post-training evaluation.

### Main risk

The scale is conditional on dataset and geometry. It must not be marketed as a universal IQ-like number.

---

## 2. Better Calibrated, Less Human

### Hook

> We improved the model’s probability score—and made its organization of human ambiguity no better.

### Scientific core

Compare post-hoc interventions on:

- NLL;
- JSD;
- graph turnover;
- relational recovery;
- effective bits;
- missing-region recall.

### Flashy artifact

A before/after animation:

- points soften;
- many graph edges move;
- human-supported regions remain unrecovered or shrink.

### Current evidence

The full classifier experiment already shows large NLL gap closure with less than roughly one percent relational gap closure.

### Moonshot extension

A calibration algorithm with a constraint:

\[
b_{\mathrm{eff}}(\text{calibrated})
\ge
b_{\mathrm{eff}}(\text{raw})-\epsilon.
\]

### Main risk

“Less human” is a hook, not a universal conclusion. The paper must specify the selected metric and relational target.

---

## 3. Resolution-Preserving Distillation

### Hook

> Nine models see twice as many human disagreement states as one. Can one small model learn to keep the extra view?

### Scientific core

Train a student with losses for:

- labels;
- distributions;
- pairwise distances;
- human-supported edges;
- ensemble-only edges;
- prototype assignments.

A generic objective:

\[
L
=
L_{\mathrm{task}}
+\lambda_1L_{\mathrm{distribution}}
+\lambda_2L_{\mathrm{distance}}
+\lambda_3L_{\mathrm{edge}}
+\lambda_4L_{\mathrm{prototype}}.
\]

### Flashy artifact

A single small model that preserves most of the nine-model ensemble’s prototype-equivalent resolution.

### Current feasibility

High. Requires training, but not paid APIs.

### Frontier-lab relevance

Very high. It connects pluralistic alignment to efficiency and deployment.

### Main risk

The student may imitate the ensemble’s biases as well as its useful complementarity.

---

## 4. AI Portfolio Optimizer

### Hook

> Choose the cheapest set of models that preserves ten states of human judgment.

### Scientific core

Optimize:

\[
\max_A
\left[
b_{\mathrm{eff}}(A)
-\lambda_1\operatorname{cost}(A)
-\lambda_2\operatorname{latency}(A)
-\lambda_3\operatorname{energy}(A)
\right].
\]

### Flashy artifact

A slider:

- budget;
- latency;
- required pluralistic resolution.

The selected coalition changes in real time.

### Extension

Learn an item-level router so only hard or pluralistic cases invoke a larger coalition.

### Main risk

Coalition value is dataset-dependent; routing requires held-out validation.

---

## 5. Annotation-Budget Operating System

### Hook

> Stop collecting labels when the geometry is stable—not when an arbitrary count is reached.

### Scientific core

For every item or stratum, estimate expected improvement in:

- posterior distribution;
- neighbor stability;
- prototype assignment;
- model ranking;
- group-overlap estimates.

### Flashy artifact

A live annotation dashboard:

```text
Item A: geometry stable at 5 votes.
Item B: needs 17 more votes.
Item C: more labels will not help; collect rationales.
```

### Current feasibility

High after E006.

### Frontier-lab relevance

High for data curation and human-feedback pipelines.

### Main risk

Stopping criteria must match the intended downstream use.

## Tier 2 — High-impact alignment applications

## 6. Open Reward-Model Pluralism Audit

### Hook

> A scalar reward can improve average preference while deleting a coherent human region.

### Scientific core

Use public preference datasets and open reward models.

Measure:

- group-specific score geometry;
- pairwise preference distributions;
- missing-region recall;
- effective resolution before and after scalarization;
- whether reward-model ensembles restore structure.

### Flashy artifact

Show two response clusters that humans distinguish but a scalar reward collapses onto one line.

### Current feasibility

Moderate to high with public reward models and datasets.

### Frontier-lab relevance

Extremely high because scalar reward is central to RLHF and preference optimization.

### Main risk

Preference datasets often have selection bias and sparse repeated users.

---

## 7. Pluralistic AI Jury

### Hook

> Do not ask one model for the answer. Assemble the smallest jury that covers the reasonable human space.

### Scientific core

Select models or adapters by:

- Shapley contribution;
- domain competence;
- group coverage;
- argument diversity;
- cost.

Output:

- consensus;
- alternatives;
- unresolved assumptions;
- minority-supported regions;
- provenance of each perspective.

### Current feasibility

Moderate using local models and debate/public-opinion datasets.

### Main risk

A model jury can create synthetic diversity that does not correspond to real people.

---

## 8. Minority-View Early Warning

### Hook

> This model update improved average preference—but removed a stable 12% viewpoint region.

### Scientific core

Compare pre/post systems on group- or viewpoint-specific regions.

Trigger when:

- a coherent region’s recall drops;
- group cross-support declines;
- effective bits fall;
- a prototype disappears;
- a minority-supported edge family is lost.

### Current feasibility

Requires datasets with group labels or argument/viewpoint structure. OpinionQA and PRISM are promising.

### Main risk

Use “minority view” rather than demographic minority unless demographic evidence exists.

---

## 9. Collective-Twin Overlap Engine

### Hook

> Two communities have the same average answer—but almost none of the same internal geometry.

### Scientific core

Represent each group through:

- item distributions;
- graph;
- prototypes;
- uncertainty;
- missing regions.

Compare them through:

- pointwise overlap;
- cross-support;
- prototype matching;
- GW/FGW;
- issue-specific overlap.

### Flashy artifact

Two translucent geometric structures that appear identical from one angle and separate when rotated into the full relational space.

### Current feasibility

Moderate with OpinionQA and PRISM.

### Main risk

A collective twin must never be presented as a substitute for actual participation.

---

## 10. Cross-Cultural Alignment Certification

### Hook

> The assistant preserves twelve effective states for one population and four for another.

### Scientific core

Audit models separately by culture or country:

- within-group recovery;
- cross-group transfer;
- shared and unique prototypes;
- missing-region alerts;
- steering fidelity.

### Current feasibility

Moderate with PRISM and global opinion datasets.

### Main risk

Sampling and measurement equivalence are difficult. Country is not culture, and demographic categories do not define homogeneous communities.

## Tier 3 — Applied domain moonshots

## 11. Education Misconception Atlas

### Hook

> Two students gave different answers for the same underlying reason. Two others gave the same wrong answer for completely different reasons.

### Scientific core

Collect:

- student answers;
- confidence;
- reasoning traces;
- teacher misconception labels;
- intervention outcomes.

Build:

- misconception prototypes;
- student–item geometry;
- intervention transitions;
- tutor retrieval based on reasoning neighborhoods.

### Flashy artifact

A live map showing a student move through misconception space after a targeted intervention.

### Current feasibility

Moderate. A small domain-specific pilot could be collected locally.

### Why it is strategically strong

It is a domain where relational analogy is directly useful and the social risks are lower than medicine or law.

---

## 12. Moderation Precedent Engine

### Hook

> Retrieve the cases reviewers considered analogous—not merely the texts an embedding model thinks are similar.

### Scientific core

Use distributions of reviewer decisions and rationales to map:

- policy ambiguity;
- reviewer clusters;
- analogous precedents;
- unstable boundaries;
- missing perspectives.

### Main risk

Policy decisions are normative and institution-specific. The system should surface structure, not automate final judgment.

---

## 13. Medical Differential-Geometry Assistant

### Hook

> The leading diagnosis is only one region; show the full expert differential and the similar cases supporting it.

### Scientific core

Multi-expert diagnosis distributions plus rationales and specialties.

### Main risk

High stakes, privacy, regulation, and prospective validation. This is a partnership moonshot, not a first deployment.

---

## 14. Scientific Controversy Map

### Hook

> Map not just what scientists believe, but which hypotheses, assumptions, and evidence patterns form the disagreement.

### Scientific core

Expert distributions across claims, evidence quality, and causal explanations.

### Extension

Choose the experiment with the highest expected reduction in disagreement geometry.

### Main risk

Expert sampling and rapidly changing evidence.

## Tier 4 — Long-horizon research programs

## 15. Plurality Compiler

### Hook

> Compile a human judgment space into the cheapest AI system that preserves it.

### Inputs

- repeated human judgments;
- optional groups and rationales;
- model pool;
- deployment constraints.

### Pipeline

1. estimate human geometry;
2. set required resolution;
3. audit candidate models;
4. select or route modules;
5. calibrate under preservation constraints;
6. distill;
7. certify;
8. monitor drift.

### Why it matters

This converts pluralistic alignment from a vague objective into a systems design problem.

---

## 16. Universal Disagreement Genome

### Hook

> Consensus, ambiguity, polarization, and fragmentation may have reusable geometric signatures across domains.

### Scientific core

Learn cross-domain archetypes from:

- language judgments;
- public opinion;
- safety preferences;
- education;
- expert decisions.

### Research questions

- Which structures recur?
- Which are domain-specific?
- Which models capture each archetype?
- Can annotation and routing policies transfer?

### Main risk

Cross-domain structural similarity can be semantically empty. Transfer must be validated, not assumed.

---

## 17. Collective-Judgment Digital Twins

### Hook

> Simulate a community’s full landscape of reasonable judgments—not its average opinion.

### Potential uses

- policy consultation;
- product decisions;
- organizational deliberation;
- scenario analysis.

### Required future data

- repeated panel respondents;
- many shared items;
- rationales;
- longitudinal observations;
- participation and governance.

### Main risk

Severe misuse potential. A twin cannot replace the people it models.

## Portfolio ranking

Scores are qualitative: 1 low, 5 high.

| Moonshot | Flash | Scientific depth | Feasible now | Frontier-lab relevance |
|---|---:|---:|---:|---:|
| Better Calibrated, Less Human | 5 | 5 | 5 | 5 |
| Resolution-preserving distillation | 5 | 5 | 4 | 5 |
| PluralityBench / Resolution Cards | 4 | 5 | 5 | 5 |
| Reward-model pluralism audit | 5 | 5 | 4 | 5 |
| AI portfolio optimizer | 4 | 4 | 5 | 5 |
| Collective-twin overlap | 5 | 5 | 3 | 5 |
| Annotation-budget OS | 4 | 5 | 4 | 4 |
| Education Misconception Atlas | 5 | 5 | 3 | 4 |
| Minority-view early warning | 5 | 5 | 3 | 5 |
| Plurality compiler | 5 | 5 | 2 | 5 |
| Universal disagreement genome | 5 | 5 | 2 | 4 |

## Recommended three-project strategy

### Platform

**PluralityBench**

Release the metrics, adapters, model cards, and reproducible Rust/Python interface.

### Method

**Resolution-preserving distillation**

Demonstrate that relational pluralism can be optimized and compressed.

### Demonstrator

**Education Misconception Atlas** or **open reward-model audit**

Choose education for a distinctive real-world application; choose reward models for maximum frontier-lab relevance.

## The deepest unifying question

> Can AI systems be designed not merely to average human judgments, but to preserve the resolution, structure, and legitimate alternatives contained in them?
