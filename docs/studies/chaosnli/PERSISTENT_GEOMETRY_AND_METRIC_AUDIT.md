# Persistent Disagreement Geometry & Metric Audit Report

- **Document type:** empirical research report
- **Status:** exploratory audit complete ($N=3,113$ canonical ChaosNLI items)
- **Dataset:** 3,113 three-class ChaosNLI examples (1,514 SNLI + 1,599 MNLI)

---

## 1. Mathematical Equivalence & Robustness Theorems

### Theorem 1: Hellinger & Categorical Fisher–Rao Exact Graph Equivalence

Let $p_i, q_j \in \Delta^2$ be two 3-class probability distributions, and let $\text{BC}(p, q) = \sum_{c=1}^3 \sqrt{p_c q_c}$ be the Bhattacharyya coefficient.

1. **Hellinger Distance**:
   $$H(p, q) = \sqrt{1 - \text{BC}(p, q)}$$
2. **Fisher–Rao Geodesic Distance**:
   $$d_{\text{FR}}(p, q) = 2 \arccos \text{BC}(p, q)$$

Because both $H(p,q)$ and $d_{\text{FR}}(p,q)$ are strictly monotonic transformations of $\text{BC}(p,q) \in [0, 1]$, they induce **identical pairwise distance rankings** and **identical top-$k$ fractional soft neighborhood graphs** $Q_{NX}^{\text{soft}}(k)$ for any dataset.

* **Empirical Verification ($N=3,113$ ChaosNLI items)**:
  - Spearman rank correlation $\rho = \mathbf{1.0000}$
  - Soft top-10 neighborhood overlap $Q_{NX}^{\text{soft}}(10) = \mathbf{1.0000}$
  - Unit test locked in [`tests/test_geometry_theorems.py`](../../tests/test_geometry_theorems.py).

---

## 2. Hellinger vs. JSD Dataset Robustness

* **Spearman Correlation**: $\rho = \mathbf{0.9971}$
* **Top-10 Soft Overlap**: $Q_{NX}^{\text{soft}}(10) = \mathbf{0.9995}$
* **Interpretation**: Hellinger and Jensen–Shannon Divergence induce nearly identical local neighborhoods on ChaosNLI. Relational model comparison conclusions are robust to the choice between Hellinger and JSD.

---

## 3. Aitchison Boundary & Zero-Replacement Sensitivity Audit

Aitchison Centered Log-Ratio (CLR) distance exhibits a **~12.2% neighborhood turnover** relative to Hellinger geometry ($Q_{NX}^{\text{soft}}(10) = 0.8781$). We audited whether zero-replacement policies at the simplex boundary drive this divergence or if log-ratio geometry itself differs intrinsically.

### Multiplicative Replacement vs Dirichlet Smoothing Sweep ($N=3,113$)

| Policy / Epsilon / Alpha | All Items ($N=3,113$) | Boundary Zero Items ($N=720$) | Strictly Interior Items ($N=2,393$) |
|---|---|---|---|
| **Multiplicative $\epsilon = 10^{-12}$** | 0.8779 | **0.9894** | 0.8440 |
| **Multiplicative $\epsilon = 10^{-9}$** | 0.8779 | **0.9894** | 0.8440 |
| **Multiplicative $\epsilon = 10^{-6}$** | 0.8779 | **0.9894** | 0.8440 |
| **Multiplicative $\epsilon = 10^{-4}$** | 0.8781 | **0.9894** | 0.8440 |
| **Multiplicative $\epsilon = 10^{-3}$** | 0.8782 | **0.9897** | 0.8440 |
| **Dirichlet $\alpha = 0.1$** | 0.8799 | **0.9897** | 0.8463 |
| **Dirichlet $\alpha = 0.5$** | 0.8856 | **0.9900** | 0.8529 |
| **Dirichlet $\alpha = 1.0$** | 0.8913 | **0.9896** | 0.8601 |

> **Key Discovery**: Boundary Zero items retain **98.9%–99.0% neighborhood overlap** between Hellinger and Aitchison geometry regardless of the zero-replacement value $\epsilon$. The ~12.2% turnover occurs on **strictly interior items** ($84.4\%$ overlap), proving that log-ratio geometry intrinsically distorts simplex interior distances rather than suffering from boundary numerical instability.

---

## 4. Global Covariance Participation Ratio & Local Intrinsic Dimensionality (LID)

* **Global Covariance Participation Ratio**: **$1.87$** (PC1 explains 63.0% variance, PC2 explains 37.0% variance), confirming that global human disagreement variance spans both available simplex dimensions.
* **Local Intrinsic Dimensionality (Local PCA Participation Ratio)**:
  - **Consensus Items ($H < 0.5$, $N=349$)**: Mean Local PR = **$1.17$** (1D trajectory near simplex corners).
  - **Edge Ambiguity Items ($0.5 \le H < 1.0$, $N=1,349$)**: Mean Local PR = **$1.13$** (1D trajectory along 2-way boundary edges).
  - **Diffuse Ambiguity Center Items ($H \ge 1.0$, $N=1,415$)**: Mean Local PR = **$1.39$** (2D manifold spread across 3-way disagreement).
  - **Dataset Breakdown**: MNLI items ($1.31$) exhibit higher local dimensionality than SNLI items ($1.19$).

---

## 5. Artifact Ledger

- [`metric_atlas_summary.json`](../../research/chaosnli/artifacts/exploratory/metric_atlas_summary.json)
- [`aitchison_boundary_audit_summary.json`](../../research/chaosnli/artifacts/exploratory/aitchison_boundary_audit_summary.json)
- [`local_intrinsic_dimension_summary.json`](../../research/chaosnli/artifacts/exploratory/local_intrinsic_dimension_summary.json)
- [`persistent_disagreement_summary.json`](../../research/chaosnli/artifacts/exploratory/persistent_disagreement_summary.json)
- [`simplex_explorer.html`](../../research/chaosnli/artifacts/exploratory/simplex_explorer.html)
