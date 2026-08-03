# ChaosNLI Research Program: Lab Roadmap

> **Branch**: `chaosnli-lab`  
> **Last Updated**: 2026-08-02  
> **Status**: E001 pilot_partial complete; E002 redesigned, pending run

---

## Overview

This document is the canonical reference for the ChaosNLI laboratory research program.
It records all completed work, all active experiments, all pending runs, and the full
forward agenda including Program B and beyond.

The core scientific question is:

> **Do model probability distributions over NLI labels recover the same relational geometry
> as human annotators?** If not, what is the nature of the gap, and can it be repaired?

---

## Repository Structure

```
research/chaosnli/
  lab/
    registry/          <- Experiment specs (TOML)
      E001.toml
      E002.toml
    summaries/         <- Run outputs (JSON + MD)
      E001_summary.json          <- pilot run (INVALID for citation)
      E001_v2_summary.md         <- corrected status + pending items
      E002_summary.json          <- pilot run (INVALIDATED by redesign)
    plans/             <- This file and future design docs
    artifacts/
      E001/<run_id>/   <- f32 support matrices + run_manifest.json
  rust_manifest/
    src/bin/
      e001_edge_support.rs       <- v2, ready to run
      e002_temperature_scaling.rs <- v2, ready to run after E001
  manifests/
    export_rust_inputs.py        <- exports model_probs.json + model_logits.json
```

---

## Program A: Measurement Validity (Active)

Program A establishes whether model neighborhoods contain real relational signal,
how large that signal is relative to human reproducibility, and whether scalar
temperature calibration can repair any of the gap.

---

### E001: Posterior Edge-Support Graph ⏳ pilot_partial

**Scientific question**: Do latent posterior neighborhoods of items (defined by
θ_i ~ Dirichlet(x_i + α)) align between model predictions and human vote
distributions more than expected under a stratified null?

**Status**: `pilot_partial` — directional results available; 3 headline claims
corrected; re-run required before citing.

#### Defensible pilot finding
- At Hellinger k=10: Q_model ≈ 0.00791–0.01083 vs. null mean ≈ 0.00325–0.00331
- Descriptively "2.39×–3.31× the mean null" — accurate
- Rankings are concordant (Kendall W ≈ 0.961) not invariant

#### Corrections embedded (see E001_v2_summary.md)

| Claim | Old (wrong) | Corrected |
|---|---|---|
| Significance | p < 0.001 | "Q > null_mean (pilot, B=100)" |
| Ranking | "rank-invariant" | "concordant, W ≈ 0.961; use tier groups" |
| Human gap | δ = 1 − Q (normalization artifact) | R_m = (Q_m − Q_null) / (Q_HH − Q_null) |
| Draw type | "posterior-predictive" | "latent posterior" (θ ~ Dirichlet, no x_new) |

#### What E001 v2 run will produce
- 10,000-perm null: exceedance counts, 2.5/97.5% intervals, add-one p-values, Z-scores
- Cross-fitted Q_HH: draws split A=250 / B=250; Q_HH = (1/Nk) Σ S^A·S^B
- Human-normalized recovery R_m per model
- SNLI / MNLI stratum-specific Q values
- f32 support matrices locked with run_manifest.json
- Item bootstrap intervals (B=2,000) for model tier groupings

#### Run command (tomorrow)
```bash
cd research/chaosnli/rust_manifest
cargo build --release --bin e001_edge_support
cargo run --release --bin e001_edge_support
```

#### Estimated runtime
~40–60 min (10,000 permutations × 9 models × 3 metrics × 4 k-values)

#### Go criterion
Directional result consistent across SNLI and MNLI, survives at least 3 neighborhood
scales and 2 distance geometries, add-one p < 0.01, R_m reported against Q_HH.

---

### E002: Temperature Calibration vs. Posterior Topology ⏳ redesigned, pending

**Scientific question**: Does scalar temperature calibration repair posterior-supported
relational alignment, or is the mismatch structural and uncorrectable by softmax sharpness?

**Status**: Prior run invalidated (same-sample T* selection, coarse grid, single selector).
Redesigned binary ready. Must run after E001 v2 completes.

#### Design (v2)
- **Cross-fitting**: 5-fold, stratified by source_dataset × majority_label × entropy_quintile
- **Temperature grid**: 21 points, log-spaced [0.10 ... 10.00]
- **4 selectors**: `raw_t1`, `T_NLL`, `T_JSD`, `T_topology` (each selected on 4 folds, evaluated on 5th)
- **Improvement criterion**: Q(T) − Q_null(T) > Q(1) − Q_null(1) — raw Q increase without null-adjusted excess does NOT count
- **Max-statistic permutation p-value**: Controls for optimization over 21-temperature grid
- **Degeneracy diagnostics**: entropy, pairwise distance collapse, micro-jitter stability
- **Consumes**: E001 f32 support matrices (does not regenerate)

#### Run command (tomorrow, after E001)
```bash
cargo run --release --bin e002_temperature_scaling
```

#### Estimated runtime
~2–4 hours (5-fold CV × 21 temps × 9 models × 3 metrics × 4 k-values + null distributions)

#### Outcome taxonomy

| Outcome | Description | Scientific Implication |
|---|---|---|
| **A** | T_NLL improves JSD but not topology | Pointwise calibration ≠ relational alignment; most interesting |
| **B** | T_NLL improves both | Overconfidence is a genuine contributor; quantify fraction repaired |
| **C** | Only T_Q* improves topology | Topology partially repairable but ordinary calibration misses it |
| **D** | High T raises raw Q, not null-adjusted Q | Entropy-inflation degeneracy; report as failure mode not improvement |
| **E** | No T materially improves topology | Mismatch is not softmax sharpness → strong motivation for Program B |

The pilot E002 strongly suggested **Outcome E**, but the uncorrected design means this
cannot be cited. If the rigorous v2 also produces Outcome E, it becomes the key
quantitative motivation for Program B.

#### Go criterion
Out-of-fold null-adjusted topology improvement positive, 95% CI excluding zero,
same direction in SNLI and MNLI, survives Hellinger and Jensen-Shannon,
does not coincide with pairwise distance collapse or micro-jitter instability.

---

### E001/E002 Pending Checklist

| Item | Status |
|---|---|
| E001 v2: 10K-perm null | ⬜ Run tomorrow |
| E001 v2: Cross-fitted Q_HH (250+250) | ⬜ Run tomorrow |
| E001 v2: R_m per model | ⬜ Run tomorrow |
| E001 v2: SNLI/MNLI split | ⬜ Run tomorrow |
| E001 v2: f32 artifact persistence + SHA256 manifest | ⬜ Run tomorrow |
| E001 v2: Item bootstrap rank intervals (B=2000) | ⬜ Run tomorrow |
| E001 v2: Fuzzy-min vs. product weight sensitivity | ⬜ Run tomorrow |
| E002 v2: 5-fold cross-fitted temperature eval | ⬜ After E001 |
| E002 v2: Degeneracy diagnostics per model | ⬜ After E001 |
| E002 v2: Max-statistic permutation p-value | ⬜ After E001 |
| E002 v2: SNLI/MNLI separate reporting | ⬜ After E001 |
| E002 v2: Lock E001 artifact hash in output | ⬜ After E001 |

---

### E003: Sensitivity & Triangulation (Planned)

**Scientific question**: Are E001 findings robust to the specific choice of human support
estimator, distance geometry, and neighborhood definition?

**Motivation**: Before Program B, we need to know which features of the signal are robust
vs. which are artifacts of specific design choices.

#### Planned sensitivity analyses

| Axis | Baseline | Alternative |
|---|---|---|
| Edge weight | product: W_ij · S_ij | fuzzy-min: min(W_ij, S_ij) |
| Draw type | latent posterior (θ ~ Dirichlet) | posterior-predictive (x_new ~ Mult(100, θ)) |
| Prior | Jeffreys α=0.5 | Bayes-Laplace α=1.0, weak α=0.1 |
| Distance | Hellinger | JSD, Total Variation (cross-metric) |
| Null | stratified item-identity permutation | exact profile-preserving permutation |
| k | 10 (primary) | 5, 20, 50 (already in E001 v2) |
| Human reference | Q_HH cross-fitted | Q_H avg (non-cross-fitted, as sanity check) |

#### Deliverable
A sensitivity table: each axis varied, all others held at baseline. Report R_m changes.
Flag any axis where R_m shifts by more than one model tier.

---

## Program B: Manifold-Preserving Alignment (Future)

**Motivation**: If E002 confirms Outcome E (no temperature repairs topology), the gap is
structural — caused by the model's learned representation, not its softmax temperature.
Program B investigates whether training-time interventions can force models to recover
the posterior edge-support geometry.

**Core hypothesis**: Models trained with a relational soft-label loss that directly targets
the human posterior edge-support structure will produce neighborhoods that align with human
judgment geometry in ways that temperature scaling cannot achieve.

### B.E001: Posterior Soft-Label Supervision Baseline

**Question**: Does training on the raw human vote distribution (standard label smoothing
with human distribution) improve Q_support compared to one-hot trained models?

**Design**:
- Fine-tune on ChaosNLI with soft targets = normalized human vote vector
- Compare Q_support at T=1 vs. E001 v2 results for same architecture
- Control: same architecture, one-hot label, matched hyperparameters

**Key distinction from E002**: This changes what the model learns, not how we
post-process its outputs.

---

### B.E002: Edge-Support Loss (Relational Objective)

**Question**: Can we improve Q_support further by adding a relational training loss
that directly penalizes neighborhood geometry mismatch?

**Proposed loss**:

```
L_total = L_ce + λ · L_edge

L_edge = (1/B²) Σ_{i,j in batch} (W^model_ij(k) − S_ij)²
```

where S_ij is the frozen E001 support matrix (loaded from f32 artifact).

**Key design decisions**:
- Batch sampling strategy (random vs. stratified by S_ij magnitude)
- Whether to symmetrize W^model (it is directed by construction)
- λ sweep and its interaction with convergence
- Whether to use product or fuzzy-min W in the loss

**Deliverable**: Matched comparison: Q_support after B.E001 (soft-label) vs. B.E002
(edge-support loss). Report R_m for both relative to Q_HH.

---

### B.E003: Manifold-Preserving Contrastive Objective

**Question**: Can contrastive learning (pull together items with high S_ij, push apart
items with low S_ij) produce better relational recovery than the quadratic edge loss?

**Proposed approach**: InfoNCE-style loss where positives are item pairs with S_ij ≥ τ_80
(core edges) and negatives are sampled from items with S_ij ≤ τ_20.

---

### B.E004: Joint Calibration + Manifold Alignment

**Question**: Can we achieve both pointwise calibration (good NLL) and topological
alignment (good Q_support) simultaneously, or is there a fundamental trade-off?

**Design**: Pareto frontier sweep of (L_ce, L_edge) trade-off parameter λ.
Measure ECE, NLL, Q_support, and R_m at each point.

---

## Program C: Validity & Generalization (Future)

After Program B establishes a training method that improves Q_support:

### C.E001: Cross-Dataset Generalization
Does relational alignment learned on ChaosNLI-SNLI generalize to ChaosNLI-MNLI
and vice versa? Does it transfer to other multi-annotator datasets?

### C.E002: Annotator Subgroup Analysis
Is the posterior edge-support structure homogeneous across annotator subgroups,
or do different annotator populations produce different geometries?

### C.E003: Task Generalization
Does manifold-preserving alignment trained on NLI improve relational geometry
on semantically adjacent tasks (STS, NLI variants, commonsense QA)?

---

## Artifact Persistence Protocol

All E001 and later runs must produce:

```
research/chaosnli/lab/artifacts/<experiment_id>/<run_id>/
  run_manifest.json          <- n_items, n_draws, seeds, timestamp, file list
  support_<metric>_<k>.f32   <- row-major N×N f32 matrix (diagonal=0)
  SHA256SUMS                 <- sha256 of each f32 file
```

E002 and later experiments must embed the E001 `run_id` in their output JSON.
This creates a verifiable provenance chain.

**SHA256 generation** (run after E001 completes):
```bash
Get-FileHash research/chaosnli/lab/artifacts/E001/<run_id>/*.f32 -Algorithm SHA256
```

---

## Open Design Decisions

| Decision | Options | Notes |
|---|---|---|
| Fuzzy-min vs. product weight | Both; compare in E003 | Product used in E001/E002 pilot |
| Prior α | 0.5 (Jeffreys) vs 1.0 (Laplace) | Jeffreys used so far; sensitivity in E003 |
| Cross-fitting folds | 5 (E002 v2) | Stratified by dataset × label × entropy |
| Null model | Stratified item-identity | Profile-preserving as robustness check in E003 |
| k (primary) | 10 | Varies 5, 20, 50 for sensitivity |
| Program B architecture | TBD | Likely RoBERTa-Large (best pilot model) |
| λ schedule (B.E002) | TBD | Grid search at first; Pareto in B.E004 |

---

## Key Methodological Notes

### On "latent posterior" vs "posterior-predictive"
We sample θ_i ~ Dirichlet(x_i + α) — these are **latent posterior draws**.
**Posterior-predictive draws** would additionally sample x_new ~ Multinomial(100, θ).
The latter has higher variance and represents "what a new annotator batch might produce."
E003 should include posterior-predictive as a sensitivity check.

### On the normalization identity
Q_HH computed on the same 500 draws used to build S will equal 1.0 by construction
(each row has total weight k in both W and S). Cross-fitting (A=250, B=250) breaks
this circularity. **Never report same-sample Q_HH as a "human reference."**

### On improvement criteria
Raw Q(T) increasing with temperature is not evidence of improvement.
Entropy inflation (high T → near-uniform) trivially raises Q because all
neighborhoods become similar. The correct criterion is:
> Q(T) − Q_null(T) > Q(1) − Q_null(1)

### On significance
With B=10,000 permutations, the minimum add-one p-value is 1/10,001 ≈ 0.0001.
Do not claim "p < 0.001" unless the exceedance count is 0 at B=10,000.
Report as: "0 exceedances in 10,000 permutations; add-one p = 0.0001."

---

## Commit History (chaosnli-lab)

| Hash | Description |
|---|---|
| `af26e68` | E001/E002 v2 code+docs — pre-run checkpoint |

---

## Next Actions (Ordered)

1. ⬜ **Run E001 v2** — produces support matrices + corrected stats (~40-60 min)
2. ⬜ **Run E002 v2** — consumes E001 artifacts, 5-fold CV (~2-4 hr)
3. ⬜ **Write SHA256SUMS** for E001 artifacts
4. ⬜ **Update this document** with E001/E002 v2 results and model tier table
5. ⬜ **Design E003** sensitivity spec (TOML + Rust bin)
6. ⬜ **Begin Program B** architecture selection and loss design
