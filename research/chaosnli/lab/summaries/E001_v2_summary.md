# E001 Summary (Pilot Partial — Validation Incomplete)

> **Status: `pilot_partial`** — Raw directional results are available but three headline claims
> from the initial pilot run have been corrected below. Do not cite the old summary.
> This document supersedes `E001_summary.md`.

---

## What Was Computed

- **N items**: chaosnli-canonical-2026-08-02 (SNLI + MNLI, counts pending re-run)
- **Latent posterior draws**: 500 (θ ~ Dirichlet(x + α), NOT posterior-predictive)
- **Metrics**: Hellinger, Jensen-Shannon, Total Variation
- **Scales**: k ∈ {5, 10, 20, 50}
- **Null model**: Stratified identity permutation (**100 perms only — insufficient**)

---

## Defensible Findings

> [!NOTE]
> Only these findings are defensible from the pilot run. Everything else is corrected below.

**Model neighborhoods contain real relational signal beyond a stratified identity-permutation baseline.**

At Hellinger k=10, observed scores Q_model ≈ 0.00791–0.01083 versus null means ≈ 0.00325–0.00331.
Descriptively, "2.39×–3.31× the mean null" is accurate. The directional ordering is concordant
(Kendall W ≈ 0.961, mean Kendall τ ≈ 0.886) across scales and seeds.

---

## Corrections to Initial Pilot Claims

### ❌ Correction 1: `p < 0.001` is NOT supported

The initial run used only 100 null permutations and stored only their mean (no exceedances).
With B=100, the minimum achievable add-one p-value is 1/101 ≈ 0.0099.

**Corrected language:**
> Q_model > Q_null_mean (pilot descriptive, B=100). Statistical significance requires re-run with B=10,000 permutations.

**Fix in E001 v2:** 10,000 permutations, exceedance counts, add-one p = (#{Q_null ≥ Q_obs} + 1) / (B+1), 95% null intervals.

---

### ❌ Correction 2: Rankings are concordant, not invariant

Initial report claimed "rank-invariant ordering." The observed W ≈ 0.961 and mean τ ≈ 0.886
indicate high but imperfect concordance. Top-model differences are likely within item-bootstrap intervals.

**Corrected language:**
> Rankings show high concordance across scales and distance geometries (Kendall W ≈ 0.961).
> Likely tier structure:
> - Tier 1: ALBERT-xxLarge, RoBERTa-Large, BART-Large (differences within uncertainty)
> - Tier 2: XLNet-Large
> - Tier 3: RoBERTa-Base, XLNet-Base, BERT-Large
> - Tier 4: BERT-Base
> - Tier 5: DistilBERT

**Fix in E001 v2:** Item bootstrap intervals (B=2,000) per model, tier groupings where overlapping.

---

### ❌ Correction 3: `mean_human_edge_support = 1.0` is a normalization identity

The initial pilot reported a "human reference" of 1.0 and computed δ_edge = 1 − Q_model as the
"human performance gap." This is mathematically incorrect: when θ_i ~ Dirichlet, each posterior
draw contributes a row with total weight k, so the average S also has total row weight k, and
Q_HH (naïvely computed on the same draws used to build S) equals 1.0 by construction.

**Corrected language:**
> The human-normalized recovery R_m = (Q_m − Q_null) / (Q_HH − Q_null) will be computed
> using cross-fitted Q_HH: draws split into group A (250) and group B (250),
> Q_HH = (1/Nk) Σ_ij S^A_ij · S^B_ij. Values to be reported after E001 v2 run.

---

### ⚠️ Correction 4: Terminology — "latent posterior" not "posterior-predictive"

The draws θ_i ~ Dirichlet(x_i + α) are **latent posterior draws**.
Posterior-predictive draws would additionally sample x_new ~ Multinomial(100, θ).
These are different objects with different variance properties.

---

### ⚠️ Correction 5: SNLI / MNLI stratified reporting not yet done

Initial results pool both datasets. E001 v2 will report Q separately for SNLI and MNLI items.
Directional consistency across strata is required for the go-criterion.

---

## Pending Before Status → `pilot`

| Item | Status |
|---|---|
| 10,000-perm null with exceedances + intervals + add-one p-values | ⬜ TODO |
| Cross-fitted Q_HH (A/B split, 250+250 draws) | ⬜ TODO |
| Human-normalized relational recovery R_m per model | ⬜ TODO |
| SNLI / MNLI stratified Q reporting | ⬜ TODO |
| Persist S_ij matrices as f32 binary + SHA256SUMS | ⬜ TODO |
| Persist core edge tables (τ=50, τ=80) | ⬜ TODO |
| Item bootstrap intervals (B=2,000) for rank tiers | ⬜ TODO |
| Fuzzy-min vs. product weight sensitivity comparison | ⬜ TODO |
| Posterior-predictive (100-vote) sensitivity comparison | ⬜ TODO |

---

## Run Notes

- E001 v2 Rust binary is ready at `research/chaosnli/rust_manifest/src/bin/e001_edge_support.rs`
- **DO NOT cite Q_model / null ratios as significant without re-running.**
- Next run required before E002 can consume E001 support matrices.
