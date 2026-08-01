# Shadowspace Testing and Validation

## 1. Purpose

Shadowspace makes claims about geometry, projection, and reliability. A rendering that “looks right” is insufficient. Testing must cover:

- mathematical correctness;
- data identity and provenance;
- semantic honesty;
- integration with the renderer;
- known misleading cases;
- performance and accessibility;
- human interpretation.

The main strategy is to combine exact fixtures, property-based testing, deliberate failure injection, and short manual investigations.

## 2. Test layers

| Layer | Purpose | Typical tools |
|---|---|---|
| Unit | Formula and schema behavior | pytest |
| Property | Invariants over many generated cases | Hypothesis (Sprint 0: infrastructure + one lightweight property; Sprint 2: full mathematical suite) |
| Contract | Bundle and adapter compatibility | pytest, Pydantic |
| Integration | End-to-end data-to-view workflows | pytest, notebook smoke |
| Regression | Preserve known misleading and corrected cases | saved fixtures |
| Visual | Detect UI or overlay changes | browser snapshots later |
| Performance | Prevent unusable latency and memory | pytest-benchmark or custom harness |
| Usability | Test whether people interpret evidence correctly | scripted studies |

## 3. Coverage targets

Coverage is a signal, not proof.

- **95% branch coverage:** `geometry`, `metrics`, and probability transforms.
- **90% branch coverage:** bundle schemas, readers, writers, and cache keys.
- **80% overall:** during the research prototype.
- UI line coverage is secondary to interaction and semantic-state tests.
- Mutation testing is recommended after M4 for formulas and semantic guards.

No release may lower critical-module coverage without a documented reason.

## 4. Canonical fixtures

### Fixture A — three-class calibration simplex

This is the Sprint 0 smoke fixture. It is deterministic with no seed required.

Include exactly 15 points:

- 3 simplex corners: \((1,0,0)\), \((0,1,0)\), \((0,0,1)\);
- 3 edge midpoints: \((0.5,0.5,0)\), \((0.5,0,0.5)\), \((0,0.5,0.5)\);
- 1 uniform center: \((1/3,1/3,1/3)\);
- 8 deterministic interior points at known coordinates.

All 15 points have fixed IDs, known hand-calculated pairwise distances under every declared metric, and can be plotted exactly in a ternary chart. This fixture supports exact numerical tests from Sprint 0 onward without requiring the full geometry kernel.

### Fixture B — orthogonal projection fixture

Use a small matrix with named axes and known bases:

- identity plane;
- plane with one collapsed dimension;
- same plane with basis rotation;
- orthogonal plane;
- rank-deficient source matrix.

### Fixture C — neighborhood deception

Construct points such that a selected 2D projection:

- creates one false neighbor;
- tears one genuine neighbor;
- preserves one neighbor;
- creates an apparent gap unsupported by source distances.

Expected IDs are stored explicitly.

### Fixture D — deterministic four-class generator

Store generator component labels and one pinned seed. This fixture tests higher-dimensional behavior without relying on class-color intuition.

### Fixture E — semantic path fixture

Include:

- valid projection keyframes;
- independent 2D embeddings;
- a declared domain path;
- an invalid path whose metadata falsely claims semantic validity.

The validator must reject the last case.

## 5. Mathematical invariant tests

### Probability vectors

For each row \(p\):

```text
all values finite
p_i >= 0 within tolerance
sum(p) == 1 within tolerance
```

Test both strict rejection and declared normalization behavior. Do not silently repair materially invalid input.

### Square-root representation

For \(s_i=\sqrt{p_i}\):

```text
s_i >= 0
||s||_2 == 1
s_i^2 reconstructs p_i
```

### Centered log-ratio

After an explicit zero policy:

\[
\operatorname{clr}(p)_i=\log p_i-\frac{1}{D}\sum_j\log p_j.
\]

Test:

```text
all values finite
sum(clr(p)) == 0
adding a common positive scale before closure does not change clr
zero input fails when no zero policy is declared
```

### Softmax and model outputs

For stored logits \(z\) and probabilities \(p\):

```text
softmax(z) approximately equals p
argmax(z) equals argmax(p)
stored predicted label matches argmax
entropy uses the declared log base
```

### Metric properties

For metrics declared as true metrics:

```text
d(x, y) >= 0
d(x, y) == d(y, x)
d(x, x) == 0
d(x, z) <= d(x, y) + d(y, z)
```

Use tolerances appropriate to floating-point behavior. Do not apply the triangle test to a divergence or similarity that is not a metric.

Known probability cases should cover:

- identical distributions;
- distinct one-hot corners;
- uniform distribution;
- symmetric pairs;
- distributions containing zeros;
- nearly identical points.

### Projection bases

For \(F\in\mathbb{R}^{p\times2}\):

\[
F^\top F\approx I_2.
\]

Test:

- invalid dimension;
- nonfinite entries;
- nearly collinear columns;
- re-orthonormalization behavior;
- deterministic sign and orientation conventions where imposed.

### Grassmannian distance

Let singular values of \(F_a^\top F_b\) define principal angles. Test:

- zero for the same subspace;
- symmetry;
- invariance under \(F_aR\) and \(F_bS\) for orthogonal \(R,S\);
- expected maximum bounds;
- stability near singular values 0 and 1 through clamping.

### Projection identity

\[
Y=XF.
\]

Test exact small examples and shape contracts. A saved basis plus source representation must reproduce the saved projection to tolerance.

## 6. Neighborhood and diagnostic tests

For selected object \(i\), define source and displayed neighbor sets \(N_H(i)\) and \(N_L(i)\).

```text
preserved = N_H ∩ N_L
torn      = N_H - N_L
false     = N_L - N_H
```

Tests:

- sets match known fixture IDs;
- they do not include \(i\);
- deterministic tie-breaking;
- \(k=1\), \(k=N-1\), and invalid \(k\);
- duplicate-point behavior;
- source neighbor set changes when representation or metric changes;
- source neighbor set does not change merely because the projection moves;
- displayed neighbor set changes when coordinates move;
- stale cache is rejected after any semantic input changes.

For approximate neighbors:

- compare against exact \(k\)-NN on sampled subsets;
- store recall@\(k\);
- set a minimum acceptable recall before using the graph as evidence;
- label approximate results in provenance.

## 7. Semantic honesty tests

These tests are release blockers.

### Linear projection path

Required:

- representation is fixed;
- every frame can be expressed by a valid \(p\times2\) basis;
- `intermediate_frames_semantically_valid=true`.

### Sequential embedding path

Required:

- independently meaningful keyframe coordinates are recorded;
- default `intermediate_frames_semantically_valid=false`;
- interface displays an endpoints/keyframes-only warning during interpolation;
- exported findings from an intermediate frame carry the warning.

### Domain geodesic

Required:

- domain and metric are named;
- implementation and parameters are versioned;
- frame validity is tested against domain constraints.

### Representation transition

Required:

- endpoint representations named;
- selected object IDs preserved;
- interpolation meaning explicitly declared;
- defaults to endpoints-only evidence.

A generic animation object without a path kind must fail validation.

## 8. Bundle and provenance tests

### Schema

- required files exist;
- schema version is supported;
- dimensions match table columns;
- feature columns are numeric and finite where required;
- every representation has the same declared ID set or an explicit subset;
- no duplicate IDs;
- payload references resolve or are explicitly external.

### Hashes

- every immutable artifact has SHA-256 metadata;
- hash mismatch blocks evidence replay;
- cache keys include representation, metric parameters, algorithm, seed, and source hash.

### Round trip

Write, read, and rewrite a bundle. Canonical data and semantics must be equal. Nonsemantic file ordering may differ, but canonical hashes should be stable where promised.

### Finding replay

An exported investigation record must restore:

- dataset;
- representation;
- metric;
- basis or embedding;
- filters;
- selection;
- neighborhood \(k\);
- diagnostic state;
- warning state.

## 9. Failure-injection tests

Intentionally introduce:

- a swapped row order;
- one changed probability;
- an invalid basis;
- a stale neighbor graph;
- missing payload;
- wrong metric label;
- false semantic-validity flag;
- a view from another representation;
- a corrupted manifest;
- an unrecognized schema major version.

The system should fail early with actionable messages. It must not render plausible but semantically misaligned data.

## 10. Manual test scripts

### A. Ten-minute synthetic investigation

1. Load the pinned four-class bundle.
2. Select the Fisher–Rao-related square-root representation.
3. Find the visible bridge between two populations.
4. Save and name the current view.
5. Select three bridge points and inspect source images or probability bars.
6. Turn on preserved, torn, and false neighbors.
7. Change to raw probability coordinates.
8. Decide whether the bridge is persistent, projection-dependent, or representation-dependent.
9. Restore the original view.
10. Export and reload the finding.

**Pass condition:** the user can state what evidence supports the bridge and what the visualization cannot establish.

### B. Semantic-path test

1. Play a linear projection tour.
2. Pause between keyframes.
3. Confirm the interface labels the frame as a valid projection.
4. Load independent UMAP or other embedding keyframes.
5. Pause midway.
6. Confirm the interface says the midpoint is a correspondence-preserving morph, not an embedding result.
7. Export the midpoint and inspect its warning metadata.

### C. Known-neighbor test

1. Load the neighborhood-deception fixture.
2. Select the designated object.
3. Verify the expected preserved, false, and torn IDs.
4. Rotate the view to remove the collapse.
5. Confirm the false/torn state changes.
6. Change source metric and confirm the source-neighbor label changes.

### D. Accessibility test

Complete core tasks:

- keyboard only;
- with animation paused;
- with reduced motion;
- without relying on color;
- at 200% browser zoom;
- with concise text summaries enabled.

## 11. Visual validation

Do not use screenshots as the only proof of mathematical behavior.

Later browser snapshots should target stable interface states:

- semantic badge and warning;
- legend;
- selected object panel;
- preserved/false/torn line categories;
- error states;
- saved-view metadata.

Avoid brittle snapshots of exact scatter coordinates unless using a pinned fixture and deterministic transform.

## 12. Performance targets

Targets become gates after M4; before then they are recorded measurements.

On documented reference hardware:

- 70,000 objects render at a usable median of at least 30 FPS during a tour.
- Selected-point diagnostics respond within 150 ms when source \(k\)-NN is precomputed.
- A local cached Fashion-MNIST bundle loads within 5 seconds.
- The workbench remains below approximately 1 GB resident memory.
- Saving or restoring a view takes less than 500 ms.
- CI mathematical tests complete in under 2 minutes; slower statistical tests use a separate marker.

Every benchmark record includes hardware, browser, backend, point count, dimension, and visual settings.

## 13. Numerical tolerances

Centralize tolerances rather than scattering constants:

```python
PROB_SUM_ATOL = 1e-10
ORTHONORMAL_ATOL = 1e-10
DISTANCE_ATOL = 1e-10
REPLAY_ATOL_FLOAT64 = 1e-9
REPLAY_ATOL_FLOAT32 = 1e-5
```

Values must be justified by scale and algorithm. Tests should include both well-conditioned and near-boundary inputs.

## 14. Human validation

The tool is successful only if it improves reasoning.

Early formative sessions should record:

- what participants think a moving point means;
- whether they understand the declared source metric;
- whether semantic warnings change interpretation;
- whether integrity links are overwhelming;
- whether they can distinguish persistent from view-specific structure;
- confidence before and after checking diagnostics;
- which controls they ignore.

Do not use experts alone. Include people comfortable with data but unfamiliar with tours.

## 15. Release checklist

- [ ] Critical coverage targets met.
- [ ] Property tests pass on the pinned seed set.
- [ ] Semantic-path tests pass.
- [ ] Bundle corruption tests pass.
- [ ] Three-class exact fixture passes.
- [ ] Deliberately misleading projection triggers expected diagnostics.
- [ ] Manual investigation completed from a clean checkout.
- [ ] Reduced-motion and keyboard paths work.
- [ ] Performance recorded.
- [ ] Research and dependency versions updated.
- [ ] Known limitations visible in the UI and release notes.
