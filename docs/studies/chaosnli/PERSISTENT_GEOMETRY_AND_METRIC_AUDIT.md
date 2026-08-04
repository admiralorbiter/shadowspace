# Persistent Disagreement Geometry & Metric Audit Report

- **Document type:** empirical research report
- **Status:** scientific refinement audit complete ($N=3,113$ canonical ChaosNLI items)
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

### Theorem 2: The Calibration Ray Theorem & Ambiguity Angle Invariance (E016)

Let $q(T) = \text{softmax}(z/T)$ be the temperature-scaled prediction of a model with logits $z \in \mathbb{R}^C$.
In Centered Log-Ratio (CLR) space, $\text{clr}(q)_c = \log q_c - \frac{1}{C}\sum_d \log q_d = \frac{z_c - \bar{z}}{T}$.

1. **Calibration Ray Identity**:
   $$\text{clr}(q(T)) = \frac{1}{T} \text{clr}(q(1))$$
   A model's complete temperature scaling orbit $T \mapsto q(T)$ is an **exact positive ray from the origin in CLR space**. Temperature scaling alters vector magnitude $\|\text{clr}(q)\|$, but cannot change its direction.

2. **Ambiguity Angle Invariance**:
   For human target $h = \text{clr}(p)$ (with Dirichlet smoothing $\boldsymbol{\alpha}=0.5$) and model vector $m = \text{clr}(q(1))$, the **ambiguity angle**:
   $$\theta_i = \arccos \left( \frac{\langle h_i, m_i \rangle}{\|h_i\| \|m_i\|} \right)$$
   is **100% invariant** under scalar temperature scaling $T > 0$.

3. **Exact Orthogonal Decomposition**:
   $$h_i = \underbrace{\alpha^* m_i}_{\text{calibration-reachable}} + \underbrace{(h_i - \alpha^* m_i)}_{\text{orthogonal directional ambiguity error}}$$
   where $\alpha^* = \max\left(0, \frac{\langle h_i, m_i \rangle}{\|m_i\|^2}\right)$ and optimal CLR temperature $T^* = 1/\alpha^*$.

---

## 2. 3-Level Calibration Reachability Disaggregation

Across all 9 baseline models, we disaggregate scalar calibration error into 3 distinct reachability metrics:

1. **Itemwise Oracle Reachability**: Each item selects its own optimal $T_i^* \in [0.05, 100]$.
2. **Global Scalar Reachability**: Single dataset-wide $T^*$ minimizing mean Hellinger distance.
3. **Relational Graph Reachability**: Maximum soft neighborhood overlap $Q_{NX}^{\text{soft}}(10)$ along the global temperature curve.

> **Key Discovery**: Scalar temperature scaling can only adjust radial sharpness. The majority of residual human-distribution error lies **orthogonal to the calibration ray** (directional ambiguity error), geometrically explaining why temperature scaling improves NLL while failing to rotate relational neighborhood structure.

---

## 3. Weighted Geometric Tears & Posterior Support Loss

Rather than relying on unweighted hard edge thresholding ($98.8\%$), we evaluate fuzzy mass overlap and posterior support loss:

$$\text{Overlap}_{\min}(W_H, W_M) = \frac{1}{Nk} \sum_{ij} \min(W_{ij}^H, W_{ij}^M)$$
$$\text{HumanCoreLoss} = 1 - \frac{\sum_{ij} W_{ij}^H S_{ij}}{\sum_{ij} W_{ij}^M S_{ij}}$$

Where $S_{ij} = \text{Pr}(j \in \text{top-10}(i) \mid \boldsymbol{\theta} \sim \text{Dirichlet})$ is the posterior edge confidence matrix.

---

## 4. Semantic Torn Twins vs. Distribution Twins

We distinguish aggregate vote-profile matches from true semantic analogies:

- **Distribution Twins**: Items with $H(p_i, p_j) < 0.05$ (vote distribution similarity only).
- **Semantic Torn Twins**: Items with $H(p_i, p_j) < 0.05$ AND TF-IDF/embedding text similarity $d_{\text{text}}(i,j) \ge 0.40$ that the model tears apart ($H(q_i, q_j) > 0.35$).

---

## 5. Differential Belief Maps (E017)

We estimate the conditional model mapping $\mu_m(p) = \mathbb{E}[q^{(m)} \mid p]$ and its Jacobian $J_m(p) = \frac{\partial \mu_m(p)}{\partial p}$:

- **Area Compression**: $A_m(p) = |\det J_m(p)|$ ($A < 1$: local area compression; $A \approx 0$: local dimensional collapse).
- **Anisotropic Flattening**: $\kappa_m(p) = \sigma_2 / \sigma_1$ from singular values $\sigma_1 \ge \sigma_2$.
- **Conditional Dispersion**: $\Sigma_m(p) = \text{Var}(q^{(m)} \mid p)$ (quantifies content-dependent variation among items sharing identical human vote profiles).

---

## 6. Artifact Ledger

- [`calibration_ray_summary.json`](../../results/exploratory/calibration_ray_summary.json)
- [`weighted_geometric_tears_summary.json`](../../results/exploratory/weighted_geometric_tears_summary.json)
- [`semantic_torn_twins_summary.json`](../../results/exploratory/semantic_torn_twins_summary.json)
- [`differential_belief_maps_summary.json`](../../results/exploratory/differential_belief_maps_summary.json)
- [`geometry_lens.html`](../../docs/viz/chaosnli/geometry_lens.html)
