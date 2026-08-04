# A Gauge and Sheaf Theory of Human Disagreement Under Semantic Transformation

## Abstract & Central Claim

Standard evaluation of natural language processing and reasoning models treats dataset items as an unordered cloud of static points $(x_i, p_i^H, p_i^M)$, where $p_i^H \in \Delta^{C-1}$ is the human disagreement distribution and $p_i^M \in \Delta^{C-1}$ is the model predicted distribution. This static view evaluates performance purely pointwise (e.g., via cross-entropy, total variation distance, or KL divergence).

This paper introduces a geometric framework based on **differential geometry, gauge connections, parallel transport, holonomy, and cellular sheaf theory**. We propose that human and model judgments define fields over a **semantic transformation complex**. When judgments are transported through closed compositions of semantic operations (e.g., negation duality, role swapping, entity renaming), they exhibit **discrete path dependence (curvature)**: the transport of a disagreement vector around a closed loop $\gamma$ yields a holonomy matrix $H_\gamma \neq I$.

Our central theoretical result—the **Calibration-Holonomy Invariance Theorem**—proves that global, smooth, invertible recalibration of predictions acts on loop holonomy strictly by matrix conjugation ($H_\gamma^f = Df H_\gamma Df^{-1}$). Consequently, **no global invertible recalibrator can eliminate path-dependent semantic curvature**. Furthermore, we construct a **cellular sheaf of local calibrators** and show that the **sheaf Laplacian** $L_\mathcal{F} = \delta_0^* \delta_0$ and first cohomology group $H^1$ measure fundamental obstructions to globally gluing local contextual corrections. Finally, we formulate **GlueOOD**, a sheaf-theoretic metric that quantifies out-of-distribution failure as the inability of a novel semantic state to coherently glue into the existing transport sheaf.

---

## 1. The Semantic Transformation Complex

Let $\mathcal{S}$ be a space of semantic situations or sentences (e.g., NLI premise-hypothesis pairs). Define a finite set of elementary semantic generators $\mathcal{G} = \{g_1, g_2, \dots, g_m\}$, where each $g: \mathcal{S} \to \mathcal{S}$ represents a semantics-preserving or semantics-transforming operation:
- Entity renaming ($e_1 \leftrightarrow e_2$)
- Argument / role permutation ($P \leftrightarrow H$)
- Negation duality ($\neg \forall x P(x) \leftrightarrow \exists x \neg P(x)$)
- Quantifier swapping
- Controlled coreference shifting

The generators generate a transformation algebra $\mathcal{A}_{\mathcal{G}}$ with declared relations:
1. **Involutions**: $s_i^2 = e$
2. **Commutative Squares**: $a b = b a$
3. **Braid / Groupoid Relations**: $s_i s_{i+1} s_i = s_{i+1} s_i s_{i+1}$

We construct the **Semantic Transformation Complex** (or Cayley Complex) $X$:
- **0-Cells (Vertices)**: Semantic situations $x \in \mathcal{S}$.
- **1-Cells (Edges)**: Elementary transformations $x \xrightarrow{g} gx$.
- **2-Cells (Faces)**: Closed 2-dimensional polygons representing declared algebraic equations between transformation paths. Boundary operators $\partial_2: C_2 \to C_1$ distinguish contractible local face curvature from noncontractible global monodromy.

---

## 2. The Simplex Fiber Bundle & Homogeneous Affine Connections

Attached to every semantic state $x \in X$ is the local ambiguity space of probability distributions over $C$ target classes (for NLI, $C=3$: Entailment, Neutral, Contradiction):
$$M_x = \Delta^{C-1} = \left\{ p \in \mathbb{R}^C \;\Big|\; \sum_{c=1}^C p_c = 1, \; p_c > 0 \right\}$$

We map $\Delta^{C-1}$ into Euclidean space $\mathbb{R}^{C-1}$ using the **Isometric Log-Ratio (ILR)** transform:
$$z(x) = \text{ilr}(p(x)) = V^\top \log p(x) \in \mathbb{R}^{C-1}$$

For an elementary transformation edge $g: x \to gx$, we define the **homogeneous affine parallel transport operator**:
$$\mathbf{T}_{g,x} = \begin{pmatrix} A_{g,x} & b_{g,x} \\ \mathbf{0}^\top & 1 \end{pmatrix} \in \text{Aff}(C-1) \subset \mathbb{R}^{C \times C}$$
where $A_{g,x} \in \text{GL}(C-1)$ is the linear transport matrix and $b_{g,x} \in \mathbb{R}^{C-1}$ is the translation defect vector.

---

## 3. Loop Holonomy & Gauge Invariants Taxonomy

Let $\gamma = (x_0 \xrightarrow{g_1} x_1 \dots \xrightarrow{g_k} x_k = x_0)$ be a closed loop in $X$. The homogeneous holonomy matrix is:
$$\mathbf{H}_\gamma = \mathbf{T}_{g_k, x_{k-1}} \cdots \mathbf{T}_{g_1, x_0} = \begin{pmatrix} A_\gamma & b_\gamma \\ \mathbf{0}^\top & 1 \end{pmatrix}$$

### Similarity Invariants vs. Frame Diagnostics
1. **Global Similarity Invariants ($\text{GL}(C-1)$ Conjugation Invariant)**:
   - Trace: $\text{tr}(A_\gamma)$
   - Determinant: $\det(A_\gamma)$
   - Spectrum: $\text{spec}(A_\gamma)$
   - Rank of identity defect: $\text{rank}(A_\gamma - I)$
   - Linear Flatness: $A_\gamma = I_{C-1}$
   - Affine Flatness: $\mathbf{H}_\gamma = I_C \iff A_\gamma = I_{C-1} \text{ and } b_\gamma = \mathbf{0}$.
2. **Frame-Dependent Diagnostics** (Require metric-preserving frame transformations):
   - Polar Rotation Angle ($\theta_{\text{polar}}$): $\arccos\left(\frac{\text{tr}(R)}{C-1}\right)$ from $A_\gamma = R U$.
   - Frobenius Norm Curvature: $\|\log A_\gamma\|_F$.
   - Anisotropy: $\log(\sigma_{\min} / \sigma_{\max})$.

---

## 4. The Calibration-Holonomy Invariance Theorem

### Theorem 1A (Exact Connection Conjugacy)
*Let $\pi: E \to X$ be an ILR simplex bundle over a semantic transformation complex $X$, and let $A_{g,x} \in \text{GL}(C-1)$ be the transport connection. Let $f: \mathbb{R}^{C-1} \to \mathbb{R}^{C-1}$ be any global, smooth, invertible recalibration map applied to predictions. Then:*

1. *The recalibrated transport operator $A_{g,x}^f$ transforms under the local Jacobian pushforward:*
   $$A_{g,x}^f = Df(gx) \, A_{g,x} \, Df(x)^{-1}$$
2. *For any closed loop $\gamma$ starting and ending at base point $x_0$, the recalibrated loop holonomy $A_\gamma^f$ is conjugated by the Jacobian $Df(x_0)$:*
   $$A_\gamma^f = Df(x_0) \, A_\gamma \, Df(x_0)^{-1}$$
3. *The trace, determinant, spectrum, and $\text{rank}(A_\gamma - I)$ are invariant under global invertible recalibration:*
   $$\text{tr}(A_\gamma^f) = \text{tr}(A_\gamma), \quad \det(A_\gamma^f) = \det(A_\gamma), \quad \text{spec}(A_\gamma^f) = \text{spec}(A_\gamma), \quad \text{rank}(A_\gamma^f - I) = \text{rank}(A_\gamma - I)$$

---

## 5. Cellular Sheaf Theory & Data-Dependent Sheaf Laplacian

We evaluate local contextual calibrators $\{f_U\}_{U \in \mathcal{U}}$ defined over an open cover $\mathcal{U}$ of $X$.

### Data-Dependent Evaluation Restriction Maps
For overlap items $x_1, \dots, x_m \in U \cap V$, an affine calibrator parameter vector $\theta_U = (A_U, b_U) \in \text{Aff}(C-1) \cong \mathbb{R}^{C(C-1)}$ is restricted via the evaluation design matrix:
$$R_{U \to UV}: \mathbb{R}^{C(C-1)} \longrightarrow \mathbb{R}^{(C-1) m}$$
mapping parameters directly to predicted item values.

### Cohomology and Obstructions
- **Sheaf Laplacian**: $L_\mathcal{F} = \delta_0^* \delta_0$
- **0-Cohomology ($H^0 = \ker \delta_0$)**: Dimension of globally coherent calibration parameter sections.
- **1-Cohomology ($H^1 = \ker \delta_1 / \text{im} \delta_0$)**: First cohomology group measuring topological obstructions to gluing local calibrations into unified global sections across closed cycles or 2-cells.

---

## 6. Sheaf-Based Out-of-Distribution Metric (GlueOOD Solver)

Given $m$ neighboring contextual predictions $s_1, \dots, s_m \in \mathbb{R}^{C-1}$ and local restriction maps $\mathbf{T}_j = (A_j, b_j)$, **GlueOOD** solves the exact least-squares consensus optimization:
$$v^* = \arg\min_{v \in \mathbb{R}^{C-1}} \sum_{j=1}^m \| A_j v + b_j - s_j \|^2 = \left( \sum_{j=1}^m A_j^\top A_j + \lambda I \right)^{-1} \sum_{j=1}^m A_j^\top (s_j - b_j)$$

The normalized residual energy measures contextual incompatibility:
$$\text{GlueOOD}(x^*) = \frac{1}{m} \sum_{j=1}^m \| A_j v^* + b_j - s_j \|^2$$
