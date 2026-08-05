# Ambiguity Doppelgänger Atlas: Empirical Results & Findings

## Executive Summary

This document presents the empirical findings of the **Ambiguity Doppelgänger Atlas** study. We prove mathematically and demonstrate empirically that standard 1D scalar evaluation summaries (majority label, maximum probability / confidence, and Shannon entropy) suffer from a fundamental **2-to-1 degenerate information collision** in 3-class probability space.

---

## 1. Mathematical Proof & Collision Kernel Verification

- **Minority-Swap Collision Theorem**: For any 3-class probability distribution $p = (m, p_A, p_B)$ with majority probability $m$ and minority probabilities $p_A, p_B$, swapping $p_A$ and $p_B$ produces a mirror distribution $p^-$ with the exact same majority label, maximum probability $m$, and Shannon entropy $H(p)$.
- **Closed-Form Metrics**: Analytical formulas for Hellinger $d_H$, Fisher–Rao $d_{FR}$, Jensen–Shannon $d_{JS}$, and Aitchison $d_A$ distances between mirror pairs match direct numerical vector calculations with exact agreement.
- **Surface Validation**: Generated a dense grid of 6,565 points (`results/ambiguity_atlas/theory_surface.parquet`), verifying all mathematical identities across the valid parameter domain.

---

## 2. Exact Human Doppelgänger Census (ChaosNLI)

Using Polars vote count permutation grouping on 3,113 ChaosNLI item vote distributions:

| Metric | Result |
| :--- | :--- |
| **Exact Doppelgänger Groups** | **257 groups** |
| **Exact Doppelgänger Pairs** | **1,375 pairs** |
| **Participating Items** | **1,100 items** (35.3% of ChaosNLI!) |
| **Hellinger Distance (Median)** | **0.324** |
| **Hellinger Distance (Mean)** | **0.342** (Max: 0.700) |

### Majority Class Breakdown
- **Neutral Majority**: 1,156 pairs (84.1%)
- **Entailment Majority**: 181 pairs (13.2%)
- **Contradiction Majority**: 38 pairs (2.7%)

### Dataset Source Breakdown
- **Within SNLI**: 1,375 pairs
- **Within MNLI**: 181 pairs
- **Cross-Source (SNLI $\times$ MNLI)**: 595 pairs

---

## 3. Approximate Doppelgänger Census & Pareto Frontier

Evaluating pair candidate distances across continuous probability space:

- **Loose Tolerance** ($\Delta\text{conf} \le 0.02, \Delta H \le 0.05$ bits): **10,449 candidate pairs**
- **Standard Tolerance** ($\Delta\text{conf} \le 0.01, \Delta H \le 0.02$ bits): **2,022 candidate pairs**
- **Tight Tolerance** ($\Delta\text{conf} \le 0.005, \Delta H \le 0.01$ bits): **1,505 candidate pairs**

---

## 4. Dirichlet Posterior Uncertainty Audit

To verify that exact doppelgänger collisions are not artifacts of finite annotation count quantization, we conducted 2,000 Dirichlet posterior draws per item ($\theta \sim \text{Dirichlet}(c + 0.5)$):

| Stability Category | Pair Count | Percentage |
| :--- | :--- | :--- |
| **PROBABLE_COLLISION** ($\ge 80\%$ prob) | **1,039 pairs** | **75.6%** |
| **UNCERTAIN_COLLISION** ($\ge 60\%$ prob) | **227 pairs** | **16.5%** |
| **POINT_ESTIMATE_ONLY** | **109 pairs** | **7.9%** |

Over **92% of exact human doppelgänger pairs (1,266 / 1,375)** remain statistically robust or probable under posterior uncertainty!

---

## 5. Frozen Model Retention Audit

We evaluated held-out predictions across 3 models (`roberta-large`, `bart-large`, `albert-xxlarge`) and 5 calibration tiers (raw, T1 scalar, T2 diagonal ILR, T3 affine ILR, T4 nonlinear ILR) across 20,625 pair-model-tier evaluation records (`results/ambiguity_atlas/model_retention.parquet`):

### Model Contrast Summary
- **Collapse Rate**: **10.3% – 15.2%** of pairs have their human minority disagreement contrast collapsed to 0 ($|R| \le 0.10$).
- **Inversion Rate**: **4.5% – 6.5%** of pairs have their human minority disagreement orientation reversed ($R < -0.10$).
- **Preservation Rate**: **55% – 72%** of pairs preserve human disagreement directions.
- **Pointwise Calibration Effect**: Advanced ILR calibration tiers (T3, T4) adjust confidence scaling but **do not restore the lost 2D orientation bit $\delta$**, proving that scalar summary degeneration cannot be repaired downstream by pointwise calibration alone.

---

## 6. Artifacts & Deliverables

- **Interactive Explorer**: [`docs/viz/ambiguity_atlas/index.html`](file:///C:/Users/admir/Github/shadowspace/docs/viz/ambiguity_atlas/index.html)
- **Atlas Data Payload**: `results/ambiguity_atlas/atlas_payload.json`
- **Strict Doppelgänger Census Table**: `results/ambiguity_atlas/strict_pairs.parquet`
- **Posterior Audit Table**: `results/ambiguity_atlas/posterior_stability.parquet`
- **Model Retention Table**: `results/ambiguity_atlas/model_retention.parquet`
- **Cryptographic Reproducibility Manifest**: `results/ambiguity_atlas/manifest.json`
