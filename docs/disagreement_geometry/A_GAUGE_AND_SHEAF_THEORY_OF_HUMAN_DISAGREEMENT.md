# A Gauge and Sheaf Theory of Human Disagreement Under Semantic Transformation

## Abstract & Central Claim

Standard evaluation of natural language processing and reasoning models treats dataset items as an unordered cloud of static points $(x_i, p_i^H, p_i^M)$, where $p_i^H \in \Delta^{C-1}$ is the human disagreement distribution and $p_i^M \in \Delta^{C-1}$ is the model predicted distribution. This static view evaluates performance purely pointwise (e.g., via cross-entropy, total variation distance, or KL divergence).

This paper introduces a geometric framework based on **differential geometry, gauge connections, parallel transport, holonomy, and cellular sheaf theory**. We propose that human and model judgments define fields over a **semantic transformation complex**. When judgments are transported through closed compositions of semantic operations (e.g., negation duality, role swapping, entity renaming), they exhibit **discrete path dependence (curvature)**: the transport of a disagreement vector around a closed loop $\gamma$ yields a holonomy matrix $\mathbf{H}_\gamma \neq I$.

Our central theoretical result—the **Calibration-Holonomy Invariance Theorem**—proves that global, smooth, invertible recalibration of predictions acts on loop holonomy strictly by matrix conjugation ($H_\gamma^f = Df H_\gamma Df^{-1}$). Consequently, **no global invertible recalibrator can eliminate path-dependent semantic curvature**. Furthermore, we construct a **cellular sheaf of local calibrators** and show that the **sheaf Laplacian** $L_\mathcal{F} = \delta_0^* \delta_0$ and first cohomology group $H^1$ measure fundamental obstructions to globally gluing local contextual corrections. Finally, we formulate **GlueOOD**, a sheaf-theoretic metric that quantifies out-of-distribution failure as the inability of a novel semantic state to coherently glue into the existing transport sheaf.

---

## 1. The Semantic Transformation Complex

Let $\mathcal{S}$ be a space of semantic situations or sentences (e.g., NLI premise-hypothesis pairs). Define a finite set of elementary semantic generators $\mathcal{G} = \{g_1, g_2, \dots, g_m\}$, where each $g: \mathcal{S} \to \mathcal{S}$ represents a semantics-preserving or semantics-transforming operation:
- Entity renaming ($e_1 \leftrightarrow e_2$)
- Argument / role permutation ($P \leftrightarrow Q$)
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
- **2-Cells (Faces)**: Planned 2-dimensional polygons representing declared algebraic equations between transformation paths, with boundary operators $\partial_2: C_2 \to C_1$ distinguishing contractible local face curvature from noncontractible global monodromy.

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

## 4. Connection Estimation & Errors-in-Variables Total Least Squares (TLS)

When source and target ambiguity coordinates $X, Y \in \mathbb{R}^{N \times d}$ contain symmetric measurement noise $X_c = Z_x + \epsilon_x$ and $Y_c = Z_y + \epsilon_y$ with $\epsilon_x, \epsilon_y \sim \mathcal{N}(0, \sigma^2 I)$, standard OLS introduces attenuation bias:
$$\hat{T}_{\text{OLS}} \longrightarrow \frac{r^2}{r^2 + \sigma^2} T$$

We implement **Multivariate Total Least Squares (TLS)** by performing singular value decomposition (SVD) on the stacked centered matrix $[X_c \quad Y_c] = U \Sigma V^\top$:
$$V = \begin{pmatrix} V_{11} & V_{12} \\ V_{21} & V_{22} \end{pmatrix} \in \mathbb{R}^{2d \times 2d}$$
$$\hat{T}_{\text{TLS}} = (-V_{12} V_{22}^{-1})^\top$$
which corrects errors-in-variables attenuation bias under symmetric measurement error.

### 3-Group Monte Carlo Loop Holonomy Inference
Statistical testing evaluates the 4-edge loop holonomy statistic $S_\gamma = \|\mathbf{H}_\gamma - I_3\|_F$:
1. **Group 1 (Calibration)**: 500 flat trials establish null threshold $\tau_{0.95} = Q_{0.95}(S_{\gamma, \text{flat}})$.
2. **Group 2 (Validation)**: 500 independent flat trials evaluate empirical $\text{FPR} = \Pr(S_{\gamma, \text{flat}} > \tau_{0.95})$.
3. **Group 3 (Power)**: 500 independent curved trials evaluate empirical $\text{Power} = \Pr(S_{\gamma, \text{curved}} > \tau_{0.95})$.

---

## 5. Bounded Model-Theoretic First-Order Logic Semantics ($k=3$)

We evaluate First-Order Logic formulas over a domain size $k=3$ ($D=\{e_1, e_2, e_3\}$) and unary predicates $P, Q$.
- Total First-Order Structures: $2^3 \times 2^3 = 64$ models.
- Non-empty $P$ Structures: $(2^3 - 1) \times 2^3 = 56$ models.
- **StrictFO**: Evaluates premise and hypothesis over all 64 models ($\models_{k=3, \{P,Q\}}$), returning Neutral for $\forall x(P(x) \to Q(x)) \models \exists x(P(x) \land Q(x))$ due to empty $P$ models.
- **ExistentialImport**: Restricts to the 56 non-empty $P$ models, returning Entailment.

---

## 6. Cellular Sheaf Theory & Data-Dependent Sheaf Laplacian

We evaluate local contextual calibrators $\{f_U\}_{U \in \mathcal{U}}$ defined over an open cover $\mathcal{U}$ of $X$.

### Data-Dependent Evaluation Restriction Maps
For overlap items $x_1, \dots, x_m \in U \cap V$, an affine calibrator parameter vector $\theta_U = (A_U, b_U) \in \text{Aff}(C-1) \cong \mathbb{R}^{C(C-1)}$ is restricted via the evaluation design matrix:
$$R_{U \to UV}: \mathbb{R}^{C(C-1)} \longrightarrow \mathbb{R}^{(C-1) m}$$
mapping parameters directly to predicted item values.

### Cohomology and Obstructions
- **Sheaf Laplacian**: $L_\mathcal{F} = \delta_0^* \delta_0$
- **0-Cohomology ($H^0 = \ker \delta_0$)**: Dimension of globally coherent calibration parameter sections.
- **1-Cohomology ($H^1 = \ker \delta_1 / \text{im} \delta_0$)**: First cohomology group measuring topological obstructions to gluing local calibrations into unified global sections across closed cycles.

---

## 7. Sheaf-Based Out-of-Distribution Metric (GlueOOD Solver)

Given $m$ neighboring contextual predictions $s_1, \dots, s_m \in \mathbb{R}^{C-1}$ and local restriction maps $\mathbf{T}_j = (A_j, b_j)$, **GlueOOD** solves the **ridge-regularized least-squares consensus optimization**:
$$v^* = \arg\min_{v \in \mathbb{R}^{C-1}} \sum_{j=1}^m \| A_j v + b_j - s_j \|^2 + \lambda \|v\|^2 = \left( \sum_{j=1}^m A_j^\top A_j + \lambda I \right)^{-1} \sum_{j=1}^m A_j^\top (s_j - b_j)$$

The normalized residual energy measures contextual incompatibility:
$$\text{GlueOOD}(x^*) = \frac{1}{m} \sum_{j=1}^m \| A_j v^* + b_j - s_j \|^2$$
