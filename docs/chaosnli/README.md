# Shadowspace × ChaosNLI Research Package

**Working theme:** reliable, source-grounded comparison of human and model disagreement in natural-language inference.

**Primary data scope:** 3,113 three-label ChaosNLI examples:

- 1,514 ChaosNLI-SNLI examples;
- 1,599 ChaosNLI-MNLI examples;
- 100 human judgments per example;
- canonical label order: entailment, neutral, contradiction.

Treat the 1,532 binary abductive-NLI examples as a later extension because they have a different task and geometry.

## 1. Research identity

Prior work already shows that ordinary NLI model softmax outputs often fail to recover collective human judgment distributions, and later research has tested calibration, ensembles, distribution distillation, and LLM sampling. The stronger Shadowspace question is relational:

> **Which relationships among NLI examples are supported by human judgment distributions, semantic content, and validated explanations—and where do model predictions preserve, collapse, substitute, or invent those relationships?**

The intended contribution has three levels:

1. **Computational methodology**
   - uncertainty-aware human neighborhood graphs;
   - model–human neighborhood recovery;
   - multi-scale and multi-geometry sensitivity;
   - source-grounded case selection.

2. **Empirical NLI analysis**
   - recurring model–human mismatch patterns;
   - whether disagreement taxonomies correspond to local geometry;
   - valid variation versus likely annotation error;
   - cases where model consensus is not human alignment.

3. **Human-centered visualization**
   - whether reliability-aware views improve qualified diagnosis;
   - comparison with simpler ternary and table-based baselines.

The interface is not evidence for the computational claims. Computational validation comes first; the interface is evaluated afterward.

## 2. Recommended study sequence

### Study 0 — Reproduction and data audit

Reproduce ChaosNLI statistics and supplied-model baselines.

Required outputs:

- record counts and label-count checks;
- entropy recomputation;
- majority-label consistency;
- original JSD/KL scores;
- file hashes and source manifest;
- exact label-order validation.

**Exit gate:** every selected published baseline matches within a declared tolerance.

### Study 1 — Human opinion neighborhood recovery

Construct a human neighborhood graph and compare every model graph against it.

> How much local relational structure among human opinion distributions is recovered by model prediction distributions?

Compare model–human agreement with a **human split-half reliability distribution**, rather than assuming the empirical 100-vote distribution is exact.

### Study 2 — Meaning of neighborhoods

Use external annotations where IDs overlap:

- Jiang–de Marneffe disagreement-source taxonomy;
- VariErr validity/error annotations;
- LiTEx or explanation-derived reasoning annotations;
- optionally LiveNLI or new blinded coding.

> Do human-opinion neighborhoods correspond to common disagreement causes, and does combining opinion geometry with text-semantic similarity improve that correspondence?

### Study 3 — Model mismatch taxonomy

Automatically identify and review:

- human disagreement collapsed by a model;
- spurious model uncertainty on human-consensus examples;
- correct majority label but wrong distribution shape;
- model-neighbor substitution;
- cross-model consensus unsupported by humans;
- geometry-sensitive conclusions;
- conclusions unsupported after human posterior uncertainty.

### Study 4 — Shadowspace evaluation

Only after Studies 0–3 are frozen:

> Does Shadowspace help users distinguish human-consistent, model-consistent, projection-sensitive, taxonomy-supported, and unresolved relationships?

## 3. Default decisions to preregister

| Decision | Default |
|---|---|
| Primary items | ChaosNLI-SNLI + ChaosNLI-MNLI |
| Unit of analysis | NLI item |
| Human distribution | 100-vote empirical distribution + Dirichlet posterior |
| Primary pointwise alignment | Jensen–Shannon distance |
| Primary neighborhood geometry | Hellinger distance |
| Primary neighborhood size | \(k=10\) |
| Multi-scale sensitivity | \(k\in\{5,10,20,50\}\) |
| Primary graph measure | \(Q_{NX}(k)\), adjusted with LCMC |
| Human reliability baseline | repeated 50/50 split-half graphs |
| Initial model baseline | official ChaosNLI supplied logits |
| Calibration | raw vs temperature-scaled, fit outside locked test |
| Taxonomy validation | external annotations only |
| Aitchison | sensitivity analysis with declared zero policy |
| AlphaNLI | excluded from primary three-class analysis |

## 4. What should count as a result

A defensible statement:

> At \(k=10\) under Hellinger geometry, Model A recovers 61% of human neighborhood edges, compared with a 79% median human split-half value. Recovery falls to 48% for high-entropy items. The result is stable under Jensen–Shannon distance but weaker under smoothed Aitchison geometry. On externally coded items, human-only edges are enriched for lexical and implicature cases, although intervals overlap for low-prevalence categories.

Avoid:

- “The plot shows a cluster.”
- “These are the true disagreement types.”
- “The model understands ambiguity.”
- “The model is calibrated, therefore it captures human disagreement.”
- “A stable projection proves a linguistic relation.”
- “Model consensus means human alignment.”

## 5. Documentation map

1. **LITERATURE_AND_MATH.md**
   - research landscape;
   - exact simplex geometry;
   - probability distances;
   - finite-annotation uncertainty;
   - graph and projection diagnostics;
   - construct-validity warnings.

2. **DATA_PIPELINE_AND_AUTOMATION.md**
   - acquisition and schemas;
   - deterministic pipeline;
   - model inference and calibration;
   - graph construction;
   - automated case selection;
   - image/review-packet generation;
   - Shadowspace bundle integration.

3. **HYPOTHESES_AND_ANALYSIS_PLAN.md**
   - confirmatory hypotheses;
   - estimands;
   - statistical tests;
   - robustness;
   - exploratory analyses;
   - reporting requirements.

4. **CODING_AND_REVIEW_PROTOCOL.md**
   - taxonomy use;
   - blinding;
   - packet design;
   - adjudication;
   - LLM-assistance rules;
   - inter-coder analysis.

## 6. Proposed repository placement

```text
docs/studies/chaosnli/
├── README.md
├── LITERATURE_AND_MATH.md
├── DATA_PIPELINE_AND_AUTOMATION.md
├── HYPOTHESES_AND_ANALYSIS_PLAN.md
└── CODING_AND_REVIEW_PROTOCOL.md

research/chaosnli/
├── configs/
│   ├── study.yaml
│   ├── models.yaml
│   ├── metrics.yaml
│   └── review_packets.yaml
├── src/shadowspace_chaosnli/
├── tests/
├── notebooks/
├── manifests/
└── artifacts/                 # ignored except frozen releases

data/chaosnli/
├── raw/                       # ignored; never silently modified
├── external/
├── interim/
└── processed/
```

Do not commit third-party data unless redistribution terms clearly permit it. Commit acquisition scripts, manifests, checksums, and schemas.

## 7. Immediate implementation backlog

### Required before analysis

- [ ] Study manifest schema.
- [ ] Deterministic ChaosNLI acquisition and checksums.
- [ ] Label normalization to `[entailment, neutral, contradiction]`.
- [ ] Reproduction of original entropy and baseline scores.
- [ ] Dirichlet posterior draws.
- [ ] Hellinger, JSD, TVD, Euclidean, and optional Aitchison distances.
- [ ] Split-half reliability.
- [ ] \(Q_{NX}\), LCMC, local overlap, edge support, and rank comparison.
- [ ] Official supplied model predictions.
- [ ] Temperature calibration without test leakage.
- [ ] Shadowspace bundle generation from frozen tables.
- [ ] Deterministic HTML/PNG review packets.
- [ ] Annotation import and adjudication workflow.

### Required before confirmatory claims

- [ ] Lock hypotheses, \(k\), metrics, and exclusions.
- [ ] Lock exploratory/confirmatory partitions.
- [ ] Freeze model versions and inference settings.
- [ ] Verify external annotation ID overlap.
- [ ] Define missing-data handling.
- [ ] Define multiple-comparison handling.
- [ ] Run from an empty cache.
- [ ] Produce an artifact manifest with hashes.
