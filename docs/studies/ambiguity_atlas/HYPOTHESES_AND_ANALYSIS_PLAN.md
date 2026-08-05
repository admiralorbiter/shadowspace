# Hypotheses and Analysis Plan

## Study Hypotheses

### H1: Prevalence of Strict Doppelgängers in Human Annotations
Human NLI annotations in ChaosNLI contain non-trivial numbers of exact doppelgänger pairs (same majority count, same unordered minority counts, opposite minority assignment).

### H2: Summary Degeneracy Frontier in Continuous Space
In continuous probability space, a substantial portion of ChaosNLI item pairs exhibit negligible summary differences ($\Delta \text{Confidence} \le 0.01$, $\Delta \text{Entropy} \le 0.02$ bits) while maintaining significant geometric separation in full probability space ($d_H \ge 0.15$).

### H3: Posterior Stability under Dirichlet Uncertainty
Exact and tight approximate doppelgänger collisions discovered from point empirical vote counts remain statistically robust under Dirichlet posterior sampling ($\theta \sim \text{Dirichlet}(c + 0.5)$), demonstrating that the collision is not an artifact of small-sample quantization.

### H4: Model Contrast Collapse / Inversion
Standard neural language models (DeBERTa-v3, RoBERTa, Electra) frequently collapse ($R \approx 0$) or invert ($R < 0$) the minority disagreement orientation expressed by human annotators, and pointwise probability calibration does not reliably restore directional ambiguity retention.

---

## Detailed Milestone Execution Pipeline

### Milestone 0: Environment & Data Preflight
- Execute `research/ambiguity_atlas/preflight.py` to validate schema invariants on `canonical_items.parquet` and `oof_predictions.parquet`.
- Write baseline hash manifest `results/ambiguity_atlas/preflight_report.json`.

### Milestone 1: Theory Surface Validation
- Run `research/ambiguity_atlas/run_theory.py`.
- Test mathematical identities on a dense grid of $(m, \delta)$ values.
- Verify numerical agreement between closed-form expressions ($d_H, d_{FR}, d_{JS}, d_A$) and raw vector calculations.

### Milestone 2: Strict Human Doppelgänger Census
- Execute `research/ambiguity_atlas/run_strict_census.py`.
- Perform Polars count-permutation grouping.
- Analyze group sizes, majority class distributions, source dataset splits (SNLI vs MNLI vs cross-dataset), and Hellinger distance distributions.

### Milestone 3: Approximate Doppelgänger Census & Pareto Frontier
- Execute `research/ambiguity_atlas/run_approximate_census.py`.
- Compute pairwise distances across candidate item pairs with opposite minority orientation.
- Apply tolerance filters and extract Pareto-optimal pairs.

### Milestone 4: Posterior Stability Audit
- Execute `research/ambiguity_atlas/run_posterior_audit.py`.
- Draw 2,000 Dirichlet samples per shortlisted pair.
- Categorize pairs into `ROBUST_COLLISION`, `PROBABLE_COLLISION`, `UNCERTAIN_COLLISION`, and `POINT_ESTIMATE_ONLY`.

### Milestone 5: Model Retention Audit
- Execute `research/ambiguity_atlas/run_model_retention.py`.
- Compute orientation retention ratio $R = \Delta_M / \Delta_H$ across 3 models $\times$ 5 calibration tiers (T0-T4).
- Measure collapse rates, inversion rates, amplification rates, and rank correlations with human geometric separation.

### Milestone 6: Interactive Atlas Payload & Explorer
- Execute `research/ambiguity_atlas/build_atlas.py` to generate `atlas_payload.json`.
- Validate `docs/viz/ambiguity_atlas/index.html` interface.

### Milestone 7: Analysis Freeze & Manifest
- Execute `research/ambiguity_atlas/freeze_manifest.py`.
- Generate final results artifact and cryptographic checksum manifest.
