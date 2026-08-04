# A Gauge and Sheaf Theory of Human Disagreement Under Semantic Transformation

## Abstract & Central Claim

Standard evaluation of natural language processing and reasoning models treats dataset items as an unordered cloud of static points $(x_i, p_i^H, p_i^M)$, where $p_i^H \in \Delta^{C-1}$ is the human disagreement distribution and $p_i^M \in \Delta^{C-1}$ is the model predicted distribution. This static view evaluates performance purely pointwise (e.g., via cross-entropy, total variation distance, or KL divergence).

This paper introduces a geometric framework based on **differential geometry, gauge connections, parallel transport, holonomy, and cellular sheaf theory**. We propose that human and model judgments define fields over a **semantic transformation complex**. When judgments are transported through closed compositions of semantic operations (e.g., negation duality, role swapping, entity renaming), they exhibit **discrete path dependence (curvature)**: the transport of a disagreement vector around a closed loop $\gamma$ yields a holonomy matrix $H_\gamma \neq I$.

Our central theoretical result—the **Calibration-Holonomy Invariance Theorem**—proves that global, smooth, invertible recalibration of predictions acts on loop holonomy strictly by matrix conjugation ($H_\gamma^f = Df H_\gamma Df^{-1}$). Consequently, **no global invertible recalibrator can eliminate path-dependent semantic curvature**. Furthermore, we construct a **cellular sheaf of local calibrators** and show that the **sheaf Laplacian** $L_\mathcal{F} = \delta^* \delta$ and first cohomology group $H^1$ measure fundamental obstructions to globally gluing local contextual corrections. Finally, we formulate **GlueOOD**, a sheaf-theoretic metric that quantifies out-of-distribution failure as the inability of a novel semantic state to coherently glue into the existing transport sheaf.

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
- **2-Cells (Faces)**: Closed 2-dimensional polygons representing declared algebraic equations between transformation paths (e.g., the commuting square $x \xrightarrow{a} ax \xrightarrow{b} abx \approx bax \impliedby bx \impliedby x$).

---

## 2. The Simplex Fiber Bundle

Attached to every semantic state $x \in X$ is the local ambiguity space of probability distributions over $C$ target classes (for NLI, $C=3$: Entailment, Neutral, Contradiction):
$$M_x = \Delta^{C-1} = \left\{ p \in \mathbb{R}^C \;\Big|\; \sum_{c=1}^C p_c = 1, \; p_c > 0 \right\}$$

The tangent space to $\Delta^{C-1}$ at any point is $(C-1)$-dimensional. We map $\Delta^{C-1}$ into Euclidean space $\mathbb{R}^{C-1}$ using the **Isometric Log-Ratio (ILR)** transform:
$$z(x) = \text{ilr}(p(x)) = V^\top \log p(x) \in \mathbb{R}^{C-1}$$
where $V \in \mathbb{R}^{C \times (C-1)}$ is an orthonormal basis matrix for the zero-sum subspace $\mathbf{1}^\perp \subset \mathbb{R}^C$.

This defines a smooth fiber bundle $\pi: E \to X$ with base space $X$ and fiber $F_x \cong \mathbb{R}^{C-1}$. Human label distributions $p^H(x)$ and model predicted distributions $p^M(x)$ are sections $s^H, s^M \in \Gamma(E)$ of this bundle.

---

## 3. Connections and Parallel Transport

For an elementary transformation edge $g: x \to gx$, we define the **local parallel transport operator**:
$$T_{g,x}: F_x \longrightarrow F_{gx}$$

In ILR coordinates $z(x) \in \mathbb{R}^{C-1}$, $T_{g,x}$ is a linear map $T_{g,x} \in \text{GL}(C-1)$ or affine map $T_{g,x} \in \text{Aff}(C-1)$. It answers the fundamental question:
> *If an ambiguity distribution is perturbed along direction $v \in F_x$ at state $x$, how does that ambiguity perturbation transport through semantic operation $g$?*

Given a local neighborhood of base items $\{x_j\}$, the transport map $T_{g,x}$ is estimated via local multivariate linear regression:
$$z(g x_j) - \bar{z}_g \approx T_{g,x} \left( z(x_j) - \bar{z} \right)$$
or via Procrustes alignment.

---

## 4. Loop Holonomy & Gauge Invariants

Let $\gamma = (x_0 \xrightarrow{g_1} x_1 \xrightarrow{g_2} x_2 \dots \xrightarrow{g_k} x_k = x_0)$ be a closed path (loop) in the Cayley complex $X$. The **holonomy matrix** of the connection along $\gamma$ is:
$$H_\gamma = T_{g_k, x_{k-1}} \cdots T_{g_2, x_1} T_{g_1, x_0} \in \text{GL}(C-1)$$

If $H_\gamma = I$, the ambiguity field is **flat** along $\gamma$. If $H_\gamma \neq I$, the field exhibits non-zero **curvature**.

### Polar Decomposition and Invariants
Applying the polar decomposition $H_\gamma = R U$ isolates:
- $R \in \text{O}(C-1)$: Orthogonal rotation and reflection map.
- $U \in \text{SPD}(C-1)$: Positive-definite symmetric shear and scaling map.

We extract four coordinate-free **gauge invariants**:
1. **Rotation Angle ($\theta_\gamma$)**: $\theta_\gamma = \arccos\left(\frac{\text{tr}(R)}{C-1}\right)$
2. **Curvature Magnitude ($c_\gamma$)**: $c_\gamma = \|\log H_\gamma\|_F$
3. **Volume Distortion ($v_\gamma$)**: $v_\gamma = \log |\det H_\gamma|$
4. **Anisotropy ($a_\gamma$)**: $a_\gamma = \log \left( \frac{\sigma_{\min}(H_\gamma)}{\sigma_{\max}(H_\gamma)} \right)$

---

## 5. The Calibration-Holonomy Invariance Theorem

### Theorem 1 (Calibration-Holonomy Invariance)
*Let $\pi: E \to X$ be an ILR simplex bundle over a semantic transformation complex $X$, and let $T_{g,x} \in \text{GL}(C-1)$ be the transport connection. Let $f: \mathbb{R}^{C-1} \to \mathbb{R}^{C-1}$ be any global, smooth, invertible recalibration map applied to predictions. Then:*

1. *The recalibrated transport operator $T_{g,x}^f$ transforms under the local Jacobian pushforward:*
   $$T_{g,x}^f = Df(gx) \, T_{g,x} \, Df(x)^{-1}$$
2. *For any closed loop $\gamma$ starting and ending at base point $x_0$, the recalibrated loop holonomy $H_\gamma^f$ is conjugated by the Jacobian $Df(x_0)$:*
   $$H_\gamma^f = Df(x_0) \, H_\gamma \, Df(x_0)^{-1}$$
3. *The trace, determinant, and eigenvalue spectrum of loop holonomy are invariant under global invertible recalibration:*
   $$\text{tr}(H_\gamma^f) = \text{tr}(H_\gamma), \quad \det(H_\gamma^f) = \det(H_\gamma), \quad \text{spec}(H_\gamma^f) = \text{spec}(H_\gamma)$$

### Proof
Let $\gamma = (x_0 \xrightarrow{g_1} x_1 \dots \xrightarrow{g_k} x_k = x_0)$ be a closed loop. The recalibrated transport along edge $g_i: x_{i-1} \to x_i$ is given by differentiating the composite map $f \circ g_i \circ f^{-1}$ at state $x_{i-1}$:
$$T_{g_i, x_{i-1}}^f = Df(x_i) \, T_{g_i, x_{i-1}} \, Df(x_{i-1})^{-1}$$

Composing these transport operators sequentially around the closed loop $\gamma$:
$$H_\gamma^f = T_{g_k, x_{k-1}}^f \cdots T_{g_1, x_0}^f$$
$$= \left( Df(x_0) \, T_{g_k, x_{k-1}} \, Df(x_{k-1})^{-1} \right) \cdots \left( Df(x_1) \, T_{g_1, x_0} \, Df(x_0)^{-1} \right)$$

Notice that adjacent interior terms cancel:
$$Df(x_i)^{-1} \, Df(x_i) = I$$

Thus, all intermediate terms collapse, leaving:
$$H_\gamma^f = Df(x_0) \, H_\gamma \, Df(x_0)^{-1}$$

Because matrix trace, determinant, and characteristic polynomials (and hence spectrum) are invariant under matrix similarity conjugation $M \mapsto A M A^{-1}$, we have:
$$\text{tr}(H_\gamma^f) = \text{tr}(H_\gamma), \qquad \det(H_\gamma^f) = \det(H_\gamma), \qquad \text{spec}(H_\gamma^f) = \text{spec}(H_\gamma) \quad \blacksquare$$

### Significance
Temperature scaling ($p \mapsto \text{softmax}(z/T)$), diagonal ILR scaling, and global affine transformations are all global invertible maps. **Theorem 1 guarantees that no global invertible calibrator can flatten a curved ambiguity field.** To alter curvature, recalibration must be non-global (context-dependent), non-invertible, or rank-reducing.

---

## 6. Cellular Sheaf Theory & The Sheaf Laplacian

To evaluate local contextual calibrators $\{f_U\}_{U \in \mathcal{U}}$ defined over an open cover $\mathcal{U}$ of $X$, we construct a **Cellular Sheaf** $\mathcal{F}$ over the Cayley complex $X$.

### Stalks and Restrictions
- **Stalk $\mathcal{F}(U)$**: Space of local affine calibrator parameters $\theta_U = (A_U, b_U) \in \text{Aff}(C-1) \cong \mathbb{R}^{(C-1) \times C}$.
- **Restriction Map $\rho_{U \to U \cap V}$**: Restricts local calibrator $\theta_U$ to the overlap domain $U \cap V$.

### Coboundary Operator and Energy
The 0-cochain space $C^0(\mathcal{U}, \mathcal{F})$ consists of parameter assignments $\theta = \{\theta_U\}_U$. The coboundary operator $\delta: C^0 \to C^1$ measures incompatibility across overlaps:
$$(\delta \theta)_{U,V} = \theta_U \big|_{U \cap V} - \theta_V \big|_{U \cap V}$$

The total **gluing energy** is:
$$E_{\text{glue}}(\theta) = \|\delta \theta\|^2$$

### The Sheaf Laplacian
The **Sheaf Laplacian** operator $L_\mathcal{F}: C^0 \to C^0$ is defined as:
$$L_\mathcal{F} = \delta^* \delta$$

- $\text{ker}(L_\mathcal{F}) \cong H^0(X, \mathcal{F})$ represents the space of **globally coherent calibration sections**.
- The first cohomology group $H^1(X, \mathcal{F}) = C^1 / \text{im}(\delta)$ measures **cohomological obstructions**: local calibration families that fit individual patches but cannot be glued globally.
- Small non-zero eigenvalues $\lambda_i(L_\mathcal{F})$ identify fragile near-global consistency modes.

---

## 7. Sheaf-Based Out-of-Distribution Metric (GlueOOD)

Traditional OOD detection evaluates whether a feature point $x^*$ lies far from training data in density or embedding space. We introduce **GlueOOD**, which measures whether $x^*$ can be coherently attached to the existing transport sheaf.

Let $x^*$ overlap with $m$ known training contexts $U_1, U_2, \dots, U_m$. Each context $U_j$ predicts a local ambiguity section $\hat{s}_{U_j}(x^*)$.

$$\text{GlueOOD}(x^*) = \min_{v \in F_{x^*}} \sum_{j=1}^m \left\| R_{x^* \to U_j} (v) - \hat{s}_{U_j}(x^*) \right\|^2$$

- **Low GlueOOD**: The new item has a coherent extension into the existing transport sheaf.
- **High GlueOOD**: Incoming contextual paths demand mutually incompatible ambiguity states at $x^*$, predicting compositional failure regardless of embedding proximity.

---

## 8. Phase E0: Finite Ambiguity Laboratory

To validate the mathematics without natural language noise, Phase E0 establishes a synthetic universe of discourse with $N$ finite entities and unary/binary predicates.

### Latent Interpreter Mixtures
A pool of $R$ distinct logical interpreters $r_1, \dots, r_R$ (e.g., strict first-order, existential import, scalar implicature, prior-driven) assign deterministic NLI labels $y_r(x) \in \{0, 1, 2\}$. A weight distribution $w_r(x)$ combines them into a synthetic human probability distribution:
$$p^H(x) = \sum_{r=1}^R w_r(x) \, \mathbf{e}_{y_r(x)}$$

### Planted Worlds
1. **Flat World**: $w_r(x) = w_r$ (constant weights across $X$). Guaranteed $H_\gamma = I$ for all closed loops.
2. **Curved World**: $w(gx) = K_g(x) w(x)$ with non-commuting transition matrices $K_g(x)$. Produces known, controllable planted rotation $\theta_\gamma > 0$.
3. **Singular World**: Transition maps with $\text{rank}(T_g) < C-1$, testing ambiguity collapse vs. curvature.

---

## 9. Repository Structure

```
research/holonomy/
├── algebra/          # Cayley complex, generators, & groupoid relations
├── worlds/           # Finite model simulator & latent interpreter mixtures
├── geometry/         # ILR bundle, transport estimators, holonomy, gauge invariants
├── sheaf/            # Stalks, coboundary operator, Sheaf Laplacian, GlueOOD
├── experiments/      # Synthetic validation suites (E000 - E004)
└── viz/              # Interactive visualization tools (Ambiguity Compass)
```
