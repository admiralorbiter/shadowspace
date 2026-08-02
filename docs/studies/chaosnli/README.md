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

## 2. Research program and current status

### Study 0 — Reproduction and data audit

**Status:** data acquisition, checksums, and current computational audits are implemented;
reproduction of every selected published baseline is not separately documented.

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

**Status:** canonical computational audit completed on 2026-08-02. See the
[Study 1 report](STUDY1_COMPUTATIONAL_AUDIT.md).

Construct a human neighborhood graph and compare every model graph against it.

> How much local relational structure among human opinion distributions is recovered by model prediction distributions?

Compare model–human agreement with a **posterior-predictive human-replicate distribution**, rather
than assuming the empirical 100-vote distribution is exact. Retain 50/50 split-half analysis as a
secondary annotation-depth sensitivity check.

### Study 2 — Meaning of neighborhoods

**Status:** exploratory work completed in two parts:

- [Study 2A](STUDY2A_EXPLORATORY_TEXT_SPACE.md) evaluates text-space and heuristic
  taxonomy signals;
- [Study 2B](STUDY2B_VARIERR_EXTERNAL_VALIDATION.md) evaluates matched VariErr
  annotations and reports an inconclusive permutation test ($p=0.2045$).

Use external annotations where IDs overlap:

- Jiang–de Marneffe disagreement-source taxonomy;
- VariErr validity/error annotations;
- LiTEx or explanation-derived reasoning annotations;
- optionally LiveNLI or new blinded coding.

> Do human-opinion neighborhoods correspond to common disagreement causes, and does combining opinion geometry with text-semantic similarity improve that correspondence?

### Study 3 — Model mismatch taxonomy

**Status:** planned; operational candidate categories exist, but blinded human review is not
reported as complete.

Automatically identify and review:

- human disagreement collapsed by a model;
- spurious model uncertainty on human-consensus examples;
- correct majority label but wrong distribution shape;
- model-neighbor substitution;
- cross-model consensus unsupported by humans;
- geometry-sensitive conclusions;
- conclusions unsupported after human posterior uncertainty.

### Study 4 — Shadowspace evaluation

**Status:** planned.

Only after Studies 0–3 are frozen:

> Does Shadowspace help users distinguish human-consistent, model-consistent, projection-sensitive, taxonomy-supported, and unresolved relationships?

## 3. Analytical conventions

The canonical Study 1 report and `results/canonical_results.json` are the source of truth for
completed quantitative analyses. The planning documents describe the broader intended program
and should not be read as evidence that every planned analysis is complete.

Recomputation scripts write ignored intermediates beneath `research/chaosnli/artifacts/`.
Only the explicitly promoted canonical JSON is release-facing; see `results/README.md`.

| Decision | Default |
|---|---|
| Primary items | ChaosNLI-SNLI + ChaosNLI-MNLI |
| Unit of analysis | NLI item |
| Human distribution | 100-vote empirical distribution + Dirichlet posterior |
| Primary pointwise alignment | Jensen–Shannon distance |
| Primary neighborhood geometry | Hellinger distance |
| Primary neighborhood size | \(k=10\) |
| Multi-scale sensitivity | \(k\in\{5,10,20,50,100\}\) |
| Primary graph measure | fractional tie-aware fuzzy overlap, \(Q_{NX}^{\text{soft}}(k)\) |
| Human reliability reference | posterior-predictive 100-vote cohort pairs |
| Initial model baseline | nine benchmark models from pre-computed logits |
| Calibration | raw vs temperature-scaled, fit outside locked test; planned extension |
| Taxonomy validation | external annotations; exploratory evidence to date |
| Aitchison | sensitivity analysis with declared zero policy |
| AlphaNLI | excluded from primary three-class analysis |

## 4. What should count as a result

A defensible statement identifies the neighborhood definition, comparison reference, uncertainty
interval, robustness checks, and limits of the external annotations. For example:

> At the declared \(k\) under the declared geometry, Model A's tie-aware neighborhood overlap is
> compared with the matched human-replicate reference and a chance/null baseline. Report the
> interval for the paired difference, sensitivity across geometries and scales, and the available
> external-validation evidence without treating exploratory categories as population labels.

Avoid:

- “The plot shows a cluster.”
- “These are the true disagreement types.”
- “The model understands ambiguity.”
- “The model is calibrated, therefore it captures human disagreement.”
- “A stable projection proves a linguistic relation.”
- “Model consensus means human alignment.”

## 5. Documentation map

| Document | Role | Status |
|---|---|---|
| [Accessible research guide](ACCESSIBLE_RESEARCH_GUIDE.md) | Plain-language walkthrough of the motivation, methods, results, limits, and applications | Current summary |
| [Study 1 computational audit](STUDY1_COMPUTATIONAL_AUDIT.md) | Canonical computational results, tie audit, model benchmark, reference surface, and geometry sensitivity | Canonical report; locked 2026-08-02 |
| [Study 2A text-space analysis](STUDY2A_EXPLORATORY_TEXT_SPACE.md) | Text-space and heuristic-taxonomy analyses | Exploratory |
| [Study 2B VariErr validation](STUDY2B_VARIERR_EXTERNAL_VALIDATION.md) | External profile-homogeneity test on matched VariErr items | Exploratory; inconclusive |
| [Unified paper draft](UNIFIED_PAPER_DRAFT.md) | Integrated scholarly narrative and methods | Draft; provenance manifest still required |
| [Literature and mathematical foundations](LITERATURE_AND_MATH.md) | Research landscape, probability geometry, uncertainty, graph diagnostics, and construct validity | Reference |
| [Data pipeline and automation plan](DATA_PIPELINE_AND_AUTOMATION.md) | Acquisition, schemas, pipeline, inference, graph construction, review packets, and bundle integration | Design plan; some components implemented |
| [Hypotheses and analysis plan](HYPOTHESES_AND_ANALYSIS_PLAN.md) | Hypotheses, estimands, tests, robustness, and reporting requirements | Planning document; not a completed preregistration |
| [Coding and review protocol](CODING_AND_REVIEW_PROTOCOL.md) | Taxonomy use, blinding, adjudication, LLM assistance, and inter-coder analysis | Protocol; completion not reported |

## 6. Repository layout

```text
docs/studies/chaosnli/
├── README.md                         # package index and status
├── ACCESSIBLE_RESEARCH_GUIDE.md      # plain-language summary
├── STUDY1_COMPUTATIONAL_AUDIT.md     # canonical Study 1 report
├── STUDY2A_EXPLORATORY_TEXT_SPACE.md # exploratory Study 2A report
├── STUDY2B_VARIERR_EXTERNAL_VALIDATION.md
├── UNIFIED_PAPER_DRAFT.md
├── LITERATURE_AND_MATH.md
├── DATA_PIPELINE_AND_AUTOMATION.md
├── HYPOTHESES_AND_ANALYSIS_PLAN.md
├── CODING_AND_REVIEW_PROTOCOL.md
└── MANIFEST.sha256                   # checksums for this documentation set

research/chaosnli/
├── configs/                          # study configuration and source hashes
├── manifests/                        # audit and analysis entry points
├── rust_manifest/                    # accelerated canonical analyses
└── tests/                            # study-specific test notes

data/chaosnli/
├── raw/                              # ignored; never silently modified
├── interim/                          # ignored generated data
└── processed/                        # ignored generated data

data/external/                         # ignored external validation data

results/                              # versioned canonical result summaries
```

Do not commit third-party data unless redistribution terms clearly permit it. Commit acquisition scripts, manifests, checksums, and schemas.

## 7. Open release checklist

The repository contains implementations and tests for the principal distance, posterior,
tie-aware graph, calibration, and pipeline components. Before presenting the package as a frozen,
confirmatory release:

- [ ] add a locked model-provenance manifest with artifact hashes and exact checkpoint IDs;
- [ ] identify the immutable preregistration or explicitly label all analyses as retrospective;
- [ ] freeze model versions, inference settings, exclusions, and analysis partitions;
- [ ] document missing-data and multiple-comparison handling for each reported hypothesis;
- [ ] run the complete analysis from an empty cache in a clean environment;
- [ ] generate a release artifact manifest with hashes;
- [ ] complete and report the blinded coding/adjudication workflow before taxonomy claims;
- [ ] complete the Shadowspace user study before human-interface effectiveness claims.
