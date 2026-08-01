# Shadowspace Research Roadmap

## 1. Research identity

Shadowspace is not a new dimensionality-reduction algorithm and not merely another embedding viewer.

Its proposed contribution is the integration of:

- navigable projection families;
- multiple representations and geometries of the same source objects;
- live local evidence about projection integrity;
- explicit semantics for every animation;
- source-object grounding;
- stability across analytical choices;
- reproducible saved investigations.

The research theme is **epistemic reliability during interactive high-dimensional visualization**.

## 2. Novelty boundary

Related systems already cover important individual capabilities:

- `dtour`: steerable guided/manual/grand projection tours and scalable rendering;
- `tourr`/spinifex: statistical tour methods and manual variable manipulation;
- Embedding Atlas: low-friction exploration of large embeddings;
- DimBridge: explanations of visible patterns through original variables;
- ClusterSense: feature-weight sensitivity;
- local distortion systems: fixed-embedding reliability and isometrization;
- Rashomon-set research: multiple similarly valid embeddings and persistent relationships.

Shadowspace should claim novelty only where it demonstrates added value from combining and semantically organizing these ideas. “We used a different dataset in `dtour`” is not a contribution.

## 3. Prototype research questions

### RQ1 — Moving integrity feedback

Does local distortion feedback during a projection tour help users distinguish genuine relationships from projection artifacts?

**Conditions**

A. static PCA/UMAP view  
B. steerable tour without integrity overlays  
C. Shadowspace tour with selected-point preserved/false/torn-neighbor evidence

**Tasks**

- identify genuine neighbors;
- reject false clusters or bridges;
- find a source neighbor hidden by a projection;
- state confidence and limitations.

**Measures**

- accuracy;
- confidence calibration;
- time;
- source-object recall;
- interaction traces;
- reported strategy.

### RQ2 — Representation literacy

Can users understand that changing representation and metric changes the meaning of proximity?

Compare:

- raw probability plus Euclidean distance;
- square-root probability plus Fisher–Rao or Hellinger;
- CLR plus Aitchison distance.

Tasks should use known synthetic answers and ask users to:

- predict nearest neighbors;
- explain why neighbor order changes;
- identify where zero smoothing matters;
- compare interpolation paths.

### RQ3 — Animation semantics

Does explicitly labeling path semantics prevent users from treating interpolation frames as analytical evidence?

Compare:

- unlabeled smooth morph;
- subtle label;
- persistent semantic badge plus warning at invalid intermediate frames.

Measure whether participants incorrectly cite a morph midpoint as an embedding result.

### RQ4 — Projection versus representation dependence

Can users correctly classify a visible structure as:

- persistent;
- projection-dependent;
- representation-dependent;
- metric-dependent;
- unsupported;
- unresolved?

The controlled four-class generator provides ground truth.

## 4. Near-term experiment sequence

### Experiment 0 — Developer validation

No participants. Verify exact fixtures, deliberate distortions, semantic guards, and replay.

### Experiment 1 — Formative think-aloud sessions

Approximately 5–8 participants comfortable with data visualization but unfamiliar with tours.

Goals:

- vocabulary;
- control discoverability;
- cognitive load;
- overlay clutter;
- mistaken mental models.

Do not perform inferential statistics. Revise the interface.

### Experiment 2 — Controlled synthetic comparison

Use the three- and four-class belief worlds. Randomize task order and projection cases. Pre-register primary outcomes before collection.

Potential primary outcome: correct classification of false versus supported visible structure.

### Experiment 3 — Fashion-MNIST ecological task

Ask participants to investigate model uncertainty and errors through source images. Evaluate whether Shadowspace yields more reproducible and qualified findings than a static embedding.

### Experiment 4 — Domain-expert transfer

Use a scientific or operational dataset with genuine stakeholders. This should happen only after the controlled mechanism works.

## 5. Candidate hypotheses

- **H1:** Moving local integrity overlays improve artifact-detection accuracy over a tour alone.
- **H2:** Semantic path labels reduce misuse of embedding-morph intermediate frames.
- **H3:** Representation switching with source-grounded explanations improves metric/representation literacy.
- **H4:** A saved-view atlas improves recall of which findings persist across projections.
- **H5:** Stability summaries reduce confidence in visually strong but assumption-sensitive clusters.
- **H6:** Question-driven tours reduce search time for targeted structures compared with grand tours.
- **H7:** Integrity overlays initially increase task time but improve confidence calibration and claim quality.

Null or negative findings are valuable: an overlay that is mathematically correct but unusable should not become a feature.

## 6. Question-driven tours

A later system can search projection space according to analytical intent.

Example objective:

\[
J(F)=
\alpha S(F)+
\beta N(F)+
\gamma T(F)-
\delta C(F),
\]

where:

- \(S\): target separation or contrast;
- \(N\): neighborhood preservation;
- \(T\): stability across resamples or views;
- \(C\): visual clutter or complexity.

User-facing questions:

- Show a view where selected errors are distinct without destroying their local neighborhoods.
- Find a view that exposes model disagreement.
- Find a view where a suspected bridge persists.
- Find the most stable projection near the current plane.
- Rotate one variable or concept direction into the view.

Research challenges:

- multi-objective tradeoffs;
- local optima in projection space;
- communicating what was optimized;
- preventing labels from becoming self-fulfilling “discoveries”;
- comparing intrinsic Grassmannian paths with practical interpolations.

## 7. Stability and the Rashomon atlas

A later atlas can track structures across:

- projection bases;
- embedding seeds;
- bootstrap samples;
- hyperparameters;
- representations;
- metrics;
- model checkpoints;
- model architectures.

For each proposed structure, store:

```text
definition of the object set
conditions tested
frequency of persistence
neighbor-edge stability
shape/layout variability
quality metrics
source-object examples
failure cases
```

Avoid reducing stability to a single percentage. A cluster may preserve membership while moving, stretching, or splitting at its boundary.

A useful interface would distinguish:

- stable membership;
- stable neighborhood relationships;
- stable separation;
- stable interpretation;
- stable screen location, which is usually not substantively meaningful.

## 8. Regional and topological reliability

Later experimental diagnostics:

- Gap Index or related empty-region deformation;
- local anisotropy;
- steadiness/cohesiveness;
- persistent-homology comparisons;
- bridge and component events during tours;
- warnings when a visible hole or gap is not supported.

These should remain secondary to selected-point neighborhoods until their interpretation and computational behavior are validated.

## 9. Counterfactual trajectories

For model beliefs, show how one source object moves when altered:

- blur or occlude an image;
- adjust temperature;
- interpolate between source objects;
- apply a learned counterfactual;
- perturb one concept or activation direction.

The trajectory can be compared in logit, probability, and other geometries.

Required caution: an input perturbation is not automatically a causal intervention. The interface must label the transformation and avoid causal language unless the study design supports it.

## 10. Future application modules

### Model comparison

Question: what does each model consider similar, and which relationships persist?

Representations:

- outputs;
- logits;
- hidden layers;
- concept scores;
- disagreement vectors.

### Scientific ensembles

Each object is a simulation run. Source inspector shows maps, curves, or trajectories. Stability across model assumptions is more important than class labels.

### Function space

Each object is an entire signal or function. Clicking a point displays the curve. Projection controls expose modes of variation, boundary conditions, or instability.

### Optimization space

Each object is a model checkpoint or parameter state. A view can be centered on gradient, Hessian, or trajectory directions, with explicit warning that a 2D loss section is not the whole landscape.

### LLM representation comparison

Only pursue a focused question such as:

- layer-to-layer stability;
- model disagreement;
- concept-direction dependence;
- semantic-neighbor persistence.

Do not build another generic text-embedding atlas.

## 11. Publication lanes

Potential contribution types:

### HCI / visualization system

A working system plus controlled evidence that semantic labels and moving integrity feedback improve interpretation.

### Visualization methodology

A formal model of path semantics, representation provenance, and live diagnostics during projection navigation.

### Applied machine-learning interpretability

A belief-space analysis method demonstrating findings about classifier uncertainty or model comparison.

### Mathematical education

An interactive curriculum comparing exact simplexes, projections, representations, and distortion.

### Software contribution

Reusable schemas and adapters for reliable projection-tour workflows.

## 12. Evidence required before expansion

Do not expand into a new domain until the current stage produces:

- a known fixture where the feature is correct;
- a known fixture where it fails;
- a manual explanation a nondeveloper can repeat;
- a reproducible exported finding;
- measured interaction value;
- a clear distinction from existing systems.

## 13. Research watch topics

Review new work periodically in:

- projection tours and projection pursuit;
- reliable dimensionality reduction;
- embedding multiplicity and stability;
- local and regional distortion metrics;
- information geometry and compositional visualization;
- visual analytics for model comparison;
- user studies of animated high-dimensional views;
- scalable browser rendering.

For every candidate paper, record:

```text
problem solved
data and task
mathematical assumptions
interaction contribution
evaluation design
available code/data
how it overlaps Shadowspace
what it leaves open
```

## 14. Long-term success criterion

Shadowspace succeeds when a user leaves with a **better-calibrated claim**, not merely a more attractive picture:

> “This bridge persists across several linear projections and two probability geometries, but it weakens under CLR/Aitchison geometry; its central points are genuinely ambiguous garment images, while the apparent empty gap beside it is projection-dependent.”

That form of statement is the target product and research outcome.
