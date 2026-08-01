# Shadowspace Mathematical and Research Knowledge Base

**Research snapshot:** 2026-08-01  
**Purpose:** concise reference for implementation, interpretation, and future research

## 1. Core conceptual distinctions

### Projection

Given \(X\in\mathbb{R}^{N\times p}\) and an orthonormal basis \(F\in\mathbb{R}^{p\times2}\),

\[
Y=XF.
\]

A linear projection preserves all object identities and maps every point through the same linear operation. It loses any variation orthogonal to the selected plane.

### Embedding

An embedding algorithm such as t-SNE or UMAP computes 2D coordinates intended to preserve selected relationships. It is generally nonlinear, objective-dependent, parameter-dependent, and often non-unique. Its axes usually do not have the direct meaning of linear feature combinations.

### Morph

A morph moves corresponding 2D points between layouts. Unless the interpolation is itself produced by a declared analytical method, the intermediate frames help track identity but are not new analytical results.

### Representation

A representation is a coordinate description before projection: raw probabilities, square-root probabilities, logits, learned activations, or CLR coordinates. Representation choice can alter the geometry and neighborhood graph before any visualization occurs.

### Metric

A metric declares what “near” means. There is no universal nearest-neighbor graph independent of representation and metric.

## 2. Tours and projection space

A tour displays a smooth sequence of low-dimensional projections of high-dimensional data. The grand tour aims to explore the space of possible projections; guided and manual tours focus exploration.

Important references:

- Daniel Asimov, “The Grand Tour: A Tool for Viewing Multidimensional Data,” 1985.
- Lee et al., “The State-of-the-Art on Tours for Dynamic Visualization of High-dimensional Data,” 2022.
- Spyrison, Cook, and Marriott, [user-controlled radial tour study](https://arxiv.org/abs/2301.00077), which found substantially better variable-attribution accuracy than PCA or a grand tour in its tested task.
- Lekschas and Abdennur, [`dtour`](https://arxiv.org/abs/2605.04306), 2026 preprint.

### The Grassmannian

The set of two-dimensional linear subspaces of \(\mathbb{R}^{p}\) is

\[
\operatorname{Gr}(2,p).
\]

A matrix \(F\in\mathbb{R}^{p\times2}\) with orthonormal columns is a basis for one such plane. It is not a unique coordinate for the plane because \(F\) and \(FR\) span the same plane for any \(R\in O(2)\).

The dimension is

\[
\dim \operatorname{Gr}(2,p)=2(p-2).
\]

The projection-control space therefore becomes large quickly even though the displayed image remains 2D.

### Principal angles and distance

For bases \(F_a,F_b\), take the singular values of \(F_a^\top F_b\):

\[
\sigma_i=\cos\theta_i.
\]

The principal angles \(\theta_i\) measure separation between the planes. A common geodesic distance is

\[
d(F_a,F_b)=\sqrt{\theta_1^2+\theta_2^2}.
\]

Clamp singular values to \([0,1]\) before applying \(\arccos\).

### `dtour` & GLERP Geodesic Implementation

The classic `dtour` method uses Catmull–Rom interpolation followed by QR re-orthonormalization. In Shadowspace Sprint 10, we implemented **True Geodesic Grand Tour (GLERP)** via SVD principal angle decomposition (`grassmann_geodesic` in `paths.py`), ensuring constant angular velocity along exact geodesics on the Grassmannian manifold $\mathrm{Gr}(k, p)$.

Shadowspace records path metadata for full reproducibility:
- path keyframes & basis matrices;
- interpolation method (`geodesic_algorithm: "GLERP"`);
- pacing metric (geodesic L2 norm of principal angles);
- global coordinate normalization ($\mathbf{X} \to [-1, 1]$) & fixed client viewport;
- dynamic Feature Loadings HUD showing real-time subspace contributions $\|V_i\| = \sqrt{V_{i,1}^2 + V_{i,2}^2}$.

`dtour` is still a strong rendering and interaction foundation, with Python and JavaScript interfaces and separated `@dtour/viewer` and `@dtour/scatter` packages.

## 3. Projection reliability

Every 2D view can produce two broad failure types:

1. **Missing structure:** a high-dimensional relation is not visible.
2. **False structure:** a visible relation is unsupported in the source space.

For neighborhoods:

- **false neighbor / intrusion:** appears close in 2D but is not close in source space;
- **torn neighbor / extrusion:** is close in source space but appears separated.

### Trustworthiness and continuity

Trustworthiness emphasizes false neighbors. Continuity emphasizes source neighbors lost from the display. These are useful but depend on neighborhood size and the declared source metric.

A global score is insufficient for interpretation. Shadowspace should show selected-point or regional evidence.

### Stress

Stress compares pairwise distances, but formulas and normalizations differ. Do not present “stress” without naming the exact definition. Recent work has warned that some so-called normalized forms are not invariant to uniform scaling.

### Gaps and empty regions

Pointwise metrics can miss visually persuasive empty space. The July 2026 preprint [“Measuring Distortion in the Empty Regions of Dimensionality Reduction Scatterplots with the Gap Index”](https://arxiv.org/abs/2607.28324) proposes comparing 2D Delaunay triangles with corresponding high-dimensional geometry. This is promising for a later regional overlay, but it is new and should remain experimental.

### Local distortion systems

The literature includes interactive local metric-distortion displays, neighborhood-fragmentation views, and local “isometrization.” These support Shadowspace's focus-plus-context approach: show evidence around a selected point rather than covering the entire display with links.

### Reliability workflow

The 2025 survey [“Unveiling High-dimensional Backstage”](https://arxiv.org/abs/2501.10168) reviews 133 papers and frames reliability across preprocessing, dimensionality reduction, evaluation, visualization, interaction, and sensemaking. Shadowspace should preserve provenance across this entire chain rather than treating distortion as a post-hoc score.

## 4. Non-uniqueness and stability

There is rarely one uniquely correct embedding. Different seeds, samples, hyperparameters, objectives, models, or representations may yield equally defensible but visually different layouts.

The 2026 preprint [“The Rashomon Effect for Visualizing High-Dimensional Data”](https://arxiv.org/abs/2604.00485) formalizes sets of similarly good embeddings and examines persistent neighbor relationships across them.

For Shadowspace, stability can be evaluated across:

- projection planes;
- stochastic embedding seeds;
- bootstrap samples;
- algorithm parameters;
- representations;
- source metrics;
- model checkpoints;
- model architectures.

A **stability atlas** should report the conditions under which a structure persists, not merely average all views into one consensus picture.

## 5. The probability simplex

A categorical distribution with \(K\) outcomes is

\[
p=(p_1,\ldots,p_K),\qquad p_i\ge0,\qquad\sum_i p_i=1.
\]

The simplex has intrinsic dimension \(K-1\).

- \(K=3\): triangle/ternary plot.
- \(K=4\): tetrahedron in 3D; specialized 2D methods exist.
- \(K=10\): nine-dimensional simplex.

Probability vectors are constrained compositions, so raw Euclidean geometry is only one possible lens.

## 6. Probability representations and distances

### Raw probability coordinates

Use \(p\) directly.

Euclidean distance:

\[
d_E(p,q)=\|p-q\|_2.
\]

Advantages:

- simple;
- finite with zeros;
- familiar linear mixtures.

Limitations:

- ignores the simplex's alternative information and ratio geometries;
- relative changes near zero may be compressed;
- should be labeled a baseline, not the canonical truth.

### Square-root coordinates and Fisher–Rao geometry

Map

\[
\phi(p)=(\sqrt{p_1},\ldots,\sqrt{p_K}).
\]

Then \(\|\phi(p)\|_2=1\), so distributions lie in the positive orthant of a unit sphere.

Bhattacharyya coefficient:

\[
BC(p,q)=\sum_i\sqrt{p_iq_i}.
\]

Using the convention chosen for Shadowspace,

\[
d_{\mathrm{FR}}(p,q)=2\arccos(BC(p,q)).
\]

Some texts omit the factor of 2 because of a different scaling convention. Store the convention in metadata.

Hellinger distance:

\[
H(p,q)=\sqrt{1-BC(p,q)}
      =\frac{1}{\sqrt2}\|\sqrt p-\sqrt q\|_2.
\]

Implementation cautions:

- clamp \(BC\) to \([0,1]\);
- zeros are allowed;
- do not confuse Euclidean projection in square-root coordinates with a complete Fisher–Rao analysis;
- spherical geodesic interpolation remains in the positive orthant only under appropriate endpoints/path handling.

### Centered log-ratio and Aitchison geometry

For strictly positive \(p\),

\[
\operatorname{clr}(p)_i=
\log p_i-\frac1K\sum_j\log p_j.
\]

The transformed coordinates sum to zero. Aitchison distance is

\[
d_A(p,q)=\|\operatorname{clr}(p)-\operatorname{clr}(q)\|_2.
\]

Advantages:

- focuses on relative ratios;
- standard geometry for compositional data.

Limitations:

- undefined at zero;
- zero replacement changes the data and must be visible;
- CLR coordinates are singular in ambient \(K\)-space because they lie in a \(K-1\) subspace;
- for numerical algorithms, an isometric log-ratio basis may later be preferable.

### Jensen–Shannon distance

Let \(m=(p+q)/2\). The Jensen–Shannon divergence is

\[
JS(p,q)=\tfrac12 KL(p\|m)+\tfrac12 KL(q\|m).
\]

Its square root is a metric under the standard construction:

\[
d_{JS}(p,q)=\sqrt{JS(p,q)}.
\]

Record the logarithm base. It is symmetric, bounded under a fixed base, and tolerates zeros with the convention \(0\log0=0\).

### Logits

Classifier logits \(z\in\mathbb{R}^{K}\) produce probabilities

\[
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

Softmax is invariant to adding the same constant to every logit. Raw logits therefore contain a redundant common-shift direction unless centered or otherwise normalized for analysis.

Comparing logit and probability spaces can reveal compression introduced by softmax, but distance in raw logit space needs an explicit treatment of this invariance.

### Wasserstein distance

Wasserstein distance requires a ground cost between outcomes. It is meaningful when outcomes live on an ordered or spatial domain. Fashion-MNIST class names do not supply a natural metric by themselves. Do not offer generic “Wasserstein between class probabilities” without defining and defending the class-to-class cost matrix.

## 7. Interpolation semantics

Different paths answer different questions.

### Mixture path

\[
p(t)=(1-t)p+tq.
\]

Meaning: probabilistic mixing in raw simplex coordinates.

### Spherical/Fisher–Rao-related path

Interpolate square-root coordinates along a spherical arc, then square the components. Meaning: a geodesic under the selected Fisher–Rao scaling.

### Log-ratio path

Interpolate in CLR or ILR coordinates and map back. Meaning: multiplicative/compositional change.

The same endpoints can yield different intermediate distributions. Shadowspace should display probability bars during the path so this difference is tangible.

## 8. Exact and controlled reference visualizations

### Three parts

A ternary plot shows the complete three-part composition exactly. This is the primary calibration world.

### Four parts

[The Simplex Projection](https://arxiv.org/abs/2403.11141) proposes a lossless 2D construction for four-part compositional data and provides a mathematical extension to finite dimensions. It is a valuable later reference, but it is not a conventional single-position scatterplot and may add cognitive load. Treat it as a validation module rather than an MVP dependency.

## 9. Source-object grounding

A high-dimensional point becomes interpretable only when connected to what it represents.

For Fashion-MNIST, each object should expose:

- image;
- true class;
- predicted class;
- logits;
- probabilities;
- entropy;
- correctness;
- optional calibration or perturbation metadata.

Do not interpret color-coded class clusters without inspecting source objects. A visible bridge may reflect real ambiguity, preprocessing artifacts, model errors, or projection distortion.

Fashion-MNIST contains 70,000 28×28 grayscale images across ten classes, with 60,000 training and 10,000 test examples: [Xiao, Rasul, and Vollgraf, 2017](https://arxiv.org/abs/1708.07747).

## 10. Research-tool landscape and novelty boundary

### `dtour`

Strong at:

- guided, manual, and grand tours;
- static projection previews;
- smooth reversible traversal;
- Python/JavaScript integration;
- GPU-scale rendering;
- linear tours and sequential embedding tours.

Shadowspace adds:

- first-class representation and metric semantics;
- moving local integrity evidence;
- source-object interpretation;
- stability and provenance;
- explicit distinction among path types.

### Embedding Atlas

Strong at scalable loading, clustering, labeling, filtering, and inspection of large embeddings. Shadowspace should not become a generic atlas.

### DimBridge and ClusterSense

Strong at connecting visible structures to original variables or feature weighting. Shadowspace can later integrate question-driven explanations but should not duplicate these systems as its initial contribution.

### Distortion-focused systems

Strong at evaluating a fixed embedding. Shadowspace's opportunity is to make local evidence move with projection navigation and remain tied to representation semantics.

## 11. Interpretation rules

1. A cluster in 2D is a hypothesis.
2. Class-color separation is not unsupervised evidence.
3. A high global quality score does not certify every region.
4. Neighbor truth is conditional on representation, metric, and \(k\).
5. A smooth animation does not imply meaningful intermediate states.
6. Stability across several assumptions is stronger evidence than visual appeal in one view.
7. Persistent identity and source inspection are required for interpretation.
8. Probability zeros, smoothing, and log bases are analytical choices, not implementation details.
9. A saved screenshot without basis, representation, metric, and provenance cannot reproduce a finding.
10. Exploratory visualization suggests questions; confirmatory claims require independent analysis.

## 12. Annotated reading list

### Foundation

- Asimov (1985), *The Grand Tour* — foundational projection-tour concept.
- Lee et al. (2022), *State-of-the-Art on Tours* — tour taxonomy and modern context.
- [`tourr`](https://ggobi.github.io/tourr/) — practical tour ecosystem.

### Direct prototype foundation

- Lekschas & Abdennur (2026), [`dtour`](https://arxiv.org/abs/2605.04306) — closest interaction and rendering base.
- [`dtour` repository](https://github.com/flekschas/dtour) — Python widget, React viewer, and lower-level renderer.

### Human interaction

- Spyrison, Cook & Marriott (2022/2023), [radial-tour user study](https://arxiv.org/abs/2301.00077) — evidence that user-controlled tour interaction can improve variable-attribution tasks.

### Reliability

- Jeon et al. (2025), [reliable DR visual analytics survey](https://arxiv.org/abs/2501.10168) — taxonomy and open problems.
- Ros, Arleo & Paulovich (2026 preprint), [Gap Index](https://arxiv.org/abs/2607.28324) — empty-region distortion.
- Sun et al. (2026 preprint), [Rashomon Effect for DR](https://arxiv.org/abs/2604.00485) — multiplicity and persistent neighbors.

### Probability and composition

- Schmitt et al. (2024), [Simplex Projection](https://arxiv.org/abs/2403.11141) — lossless four-part compositional visualization.
- Standard information-geometry and compositional-data references should be added when implementation conventions are frozen.

### Demonstration data

- Xiao, Rasul & Vollgraf (2017), [Fashion-MNIST](https://arxiv.org/abs/1708.07747).

## 13. Terms to use consistently

| Term | Shadowspace meaning |
|---|---|
| source space | declared representation plus metric used as the comparison reference |
| display space | current 2D coordinates |
| view | one static projection or embedding |
| path | a declared family or sequence of views |
| preserved neighbor | neighbor in source and display spaces |
| false neighbor | display neighbor absent from source neighborhood |
| torn neighbor | source neighbor absent from display neighborhood |
| integrity | evidence about preservation and distortion; not an absolute truth score |
| stability | persistence across a declared family of assumptions or runs |
| finding | saved view plus evidence, source identities, provenance, and note |
