# Shadowspace Implementation Plan

**Planning horizon:** prototype through research-ready platform  
**Sprint size:** approximately one focused week each; combine or split based on findings  
**Primary gate:** prove the mathematical interaction in Python before building a custom frontend

> **Status as of 2026-08-01**  
> Sprint 0: COMPLETE — 51 tests, dtour 0.4.4 installed, conventions locked, DtourAdapter protocol confirmed.  
> Sprint 1: COMPLETE — 60 tests, 90% coverage, ruff/mypy clean, bundle writer/reader/validator, 3-class calibration and 4-class synthetic generators, CLI (`shadowspace generate`, `shadowspace validate-bundle`), `src/shadowspace/math/clr.py` (shared CLR implementation, not in original plan).  
> Deferred: `BundleManifest.created_at` is `str`; upgrade to Pydantic `datetime` before Sprint 2 manifest-comparison logic is added.

## 1. Delivery strategy

Development proceeds through evidence-producing milestones. Each sprint must answer a question about feasibility or usefulness, not merely add code.

Every sprint ends with:

1. deterministic artifacts;
2. automated tests;
3. a short manual test script;
4. a recorded answer to the sprint's research questions;
5. a go, revise, or stop decision.

The project should remain runnable at the end of every sprint.

## 2. Milestone map

| Milestone | Sprints | Outcome |
|---|---:|---|
| M0 — Reproducible foundation | 0 | Package, CI, pip environment, `dtour` smoke test, 15-point calibration fixture |
| M1 — Controlled mathematical world | 1–2 | Artifact contract, synthetic beliefs, probability geometries |
| M2 — Projection navigation | 3 | Valid bases, tours, semantic path model, `dtour` adapter |
| M2b — Flask workbench shell | 3b | `flask run` renders calibration fixture; placeholder panels in place |
| M3 — Projection integrity | 4 | Selected-point true/false/torn-neighbor diagnostics in Flask panels |
| M4 — Research MVP | 5–6 | Integrated workbench, saved views, calibration and four-class study |
| M5 — Real model beliefs | 7 | Fashion-MNIST source objects, classifier outputs, belief-space exploration |
| M6 — Flask expansion decision | 8 | Evidence-based decision on expanding Flask vs. contributing upstream |
| M7 — Research extensions | 9+ | Stability atlas, question-driven tours, formal user studies |

---

## Sprint 0 — Foundation and dependency reconnaissance

### Objective

Create a reproducible repository and prove that `dtour` can display a small local dataset without committing Shadowspace to its internal API.

### Deliverables

- Python 3.12 package managed with `pip`.
- Pinned dependency set with `dtour==0.4.4` in `requirements.txt`.
- `src/`, `tests/`, `notebooks/`, `data/`, and `docs/` structure.
- Ruff, mypy, pytest, Hypothesis (infrastructure only; see Sprint 2 for substantive property tests), and coverage configuration.
- Makefile replacing CI (no GitHub Actions per user preference).
- One Jupyter notebook displaying a tiny Polars dataframe with `dtour.Widget`.
- `DtourAdapter` protocol with a minimal implementation.
- `src/shadowspace/conventions.py` with all pinned numerical constants.
- Dependency and license notes (dtour license: **MIT**).
- 15-point three-class calibration fixture (deterministic, no seed required):
  - 3 simplex corners;
  - 3 edge midpoints;
  - 1 uniform center;
  - 8 deterministic interior points.

### Automated tests

- Package imports in a clean environment.
- Adapter rejects nonnumeric or fewer-than-two-dimensional feature matrices.
- Smoke fixture produces the expected 15 rows and 3 feature columns.
- One lightweight Hypothesis generation property (e.g., generated probability rows sum to 1).
- CI installs from `requirements.txt`.
- Installed `dtour` version matches pinned value; wheel hash is recorded.

### Manual verification

1. Clone into an empty directory.
2. Run `pip install -r requirements.txt`.
3. Run all checks.
4. Open the smoke Jupyter notebook.
5. Select points and move through a `dtour.little_tour()` generated tour.
6. Confirm a Jupyter kernel restart gives the same data and initial view.

### Edge cases

- Missing WebGPU support.
- WebGL fallback or notebook rendering failure.
- A Polars dataframe containing IDs, strings, nulls, and numeric columns.
- Two numeric dimensions exactly.
- More features than rows.
- Empty or one-row data.
- `dtour` version drift.
- Private Python package mirror that does not carry `dtour` (record the failure mode; use direct PyPI or a mirror that carries the wheel).

### Questions answered

- Can `dtour` be used as a dependency without a fork?
- Does its Python widget support the object-identity and selection events needed by Shadowspace?
- Which capabilities require `@dtour/viewer` or `@dtour/scatter` later?
- Are there browser or notebook constraints that alter the plan?

### Exit gate

A new checkout installs cleanly from `requirements.txt`, renders the deterministic 15-point calibration fixture in a Jupyter notebook via `dtour.Widget`, and runs the complete test suite. The installed `dtour` wheel hash is recorded in `docs/DEPENDENCY_NOTES.md`. If selection or externally controlled views cannot be integrated through a stable boundary, document the gap before proceeding.

---

## Sprint 1 — Artifact bundle and deterministic synthetic data

### Objective

Make the source object, representation, metric, view, and provenance concepts concrete before implementing advanced mathematics.

### Deliverables

- Pydantic schemas for:
  - `SourceObject`;
  - `RepresentationSpec`;
  - `MetricSpec`;
  - `ViewSpec`;
  - `PathSpec`;
  - `DiagnosticSpec`;
  - `BundleManifest`.
- Writer and reader for the Shadowspace artifact bundle.
- Stable object IDs and explicit row ordering.
- Three-class calibration generator.
- Four-class synthetic generator with labeled latent families.
- CLI commands:

```bash
shadowspace generate synthetic --classes 4 --seed 20260801 --output data/bundles/synthetic-v1
shadowspace validate-bundle data/bundles/synthetic-v1
```

### Synthetic structures

The generator must create separable, known phenomena rather than random points alone:

- corner/confident Dirichlet populations;
- pairwise ambiguity bands;
- center/high-entropy population;
- one narrow bridge between populations;
- isolated and near-cluster outliers;
- an ordered evidence-update trajectory;
- optional duplicate and near-duplicate diagnostic records.

Every record includes `generator_component` as ground truth.

### Automated tests

- Same seed produces byte-identical or value-identical canonical tables.
- Different seeds change samples but preserve schema and counts.
- IDs are unique, stable, and identical across tables.
- Probability rows are finite, nonnegative, and sum to one.
- Manifest hashes match artifacts.
- Reader detects missing files, wrong dimensions, changed order, and corrupt hashes.
- Round-trip write/read preserves values and metadata.

### Manual verification

1. Generate the bundle twice with the same seed.
2. Diff manifests and canonical data.
3. Inspect ten records from each latent family.
4. Plot the three-class fixture in a ternary reference.
5. Confirm the four-class family labels correspond to intended probability bars.
6. Deliberately edit one file and confirm validation fails clearly.

### Edge cases

- Zero and near-zero probabilities.
- One-hot vectors.
- Duplicate objects.
- Null metadata.
- Missing payload references.
- Non-UTF-8 labels.
- Very long feature names.
- Manifest from a future schema version.

### Questions answered

- Can every screen point remain tied to one source object across all representations?
- Is the bundle understandable without the original notebook?
- Do the synthetic structures create useful, controllable test cases?
- Can a result be reproduced from bundle plus manifest alone?

### Exit gate

A generated bundle validates independently, survives a round trip, and contains ground-truth structures that can be inspected without projection.

---

## Sprint 2 — Probability representations and geometry

### Objective

Implement a small, principled set of probability-space representations and metrics with explicit assumptions.

### MVP representations

1. `probability`: raw \(p\).
2. `sqrt_probability`: \(\sqrt p\), unit-sphere coordinates.
3. `clr_probability`: centered log-ratio coordinates after declared smoothing.
4. `logit`: introduced only for model-generated data; schema may exist now.

### MVP metrics

- Euclidean distance for declared Euclidean coordinate spaces.
- Hellinger distance.
- Fisher–Rao distance using one documented scaling convention.
- Aitchison distance for positive compositions.
- Jensen–Shannon distance as an optional comparison.

Do not add Wasserstein distance until a defensible ground metric over outcomes exists.

### Deliverables

- Pure transformation functions in `src/shadowspace/math/` (CLR is already implemented; extend for sqrt_probability and logit).
- `MetricRegistry` with domain checks and metadata.
- Pairwise and selected-point distance APIs.
- Exact three-class reference notebook.
- Explanatory cards for each representation:
  - coordinates;
  - assumptions;
  - meaning of a straight path;
  - zero behavior;
  - default metric;
  - known limitations.
- Upgrade `BundleManifest.created_at` from `str` to Pydantic `datetime` type before any manifest-comparison logic is added.

### Automated tests

- Probability validation.
- `sqrt_probability` has nonnegative coordinates and unit norm.
- CLR rows are finite and sum to approximately zero after smoothing.
- Transform output dimensions are correct.
- Metric nonnegativity, symmetry, and identity of indiscernibles.
- Triangle inequality tests only for functions declared as metrics.
- Known distances:
  - identical distributions;
  - simplex corners;
  - uniform distribution;
  - edge midpoint.
- Numerical clamping keeps inverse trigonometric formulas in range.
- Invalid use, such as CLR with untreated zeros, fails rather than silently changing data.

### Manual verification

1. Select corner, center, and midpoint distributions.
2. Compare nearest neighbors under each metric.
3. Inspect how smoothing changes a nearly one-hot distribution.
4. Verify that the same object remains selected during representation switching.
5. Export a table explaining why neighbor order changed.

### Edge cases

- Exact zeros and ones.
- Tiny values below floating-point precision.
- Rows that sum to \(1\pm\epsilon\).
- Negative values from upstream numerical error.
- Distributions with one or two outcomes.
- Log-base choice for Jensen–Shannon calculations.
- Extreme smoothing values.

### Questions answered

- Are representation-dependent neighborhood changes visible and explainable?
- Which two or three metrics are sufficiently distinct to justify MVP controls?
- Can zero handling remain honest and comprehensible?
- Does the exact three-class fixture expose implementation errors?

### Exit gate

All formulas pass unit and property tests; each metric declares its compatible representation and assumptions; manual comparison produces an understandable example where neighbor identity changes for a defensible mathematical reason.

---

## Sprint 3 — Projection and tour core

### Objective

Represent projection planes correctly, generate useful tours, and enforce the semantic difference between projections and embedding morphs.

### Deliverables

- Orthonormal basis validator and canonicalization utilities.
- Projection operation \(Y=XF\).
- PCA fitted **separately per representation** using `dtour.little_tour()` or true Grassmannian geodesic Grand Tour as the underlying generator.
  - Default initial tour: `probability` representation.
  - Later: `sqrt_probability`, `clr_probability`, `logits` each get their own independently fitted tour.
  - Every `ViewSpec` for a PCA tour records:
    - `representation_id`;
    - object-ID fit hash;
    - feature-schema hash;
    - centering and scaling policy;
    - component indices;
    - eigenvalues;
    - implementation version.
  - A basis fitted in `probability` space is **rejected** if applied to `sqrt_probability` coordinates, even when dimensions match.
- Grassmannian distance from principal angles and exact GLERP (Grassmannian Linear Interpolation) geodesic trajectory generation.
- Feature preprocessing & viewport stability contract:
  - Input matrices are Z-score normalized per-feature prior to Grassmann projection.
  - Generated 2D trajectories are globally scaled to \([-1, 1]\) across all animation frames.
  - Web client uses a fixed \([-1.1, 1.1]\) viewport to guarantee zero scale jitter and prevent point clipping.
- Real-time Feature Loadings HUD:
  - Computes top feature subspace contributions \(\|V_i\| = \sqrt{V_{i,1}^2 + V_{i,2}^2}\) dynamically per frame.
- Guided Subspace Optimization (`/api/optimize-view`):
  - Fisher Linear Discriminant Analysis (`find_discriminative_basis`) for maximum class separability.
  - Local covariance optimization (`find_integrity_optimal_basis`) for local neighborhood preservation.
  - Coincident basis guard that automatically sweeps to minor discriminant components ($v_3, v_4$) when top components coincide with PCA.
- Canvas Marquee Multi-Selection:
  - Shift+Drag box marquee selection for selecting subsets of objects on the 2D canvas.
  - Subset Summary inspector panel computing object counts, average confidence, and distinct class breakdown.
- Path metadata (`geodesic_algorithm: "GLERP"`) and dynamic semantic badge (`representation_morph`).
- `DtourAdapter` accepting a representation matrix, keyframe bases, IDs, and selections.
- `representation_morph` path kind for animated transitions between separately fitted representation layouts:
  - Procrustes alignment may be applied for visual smoothness;
  - intermediate frames are **not** valid projections in either representation;
  - `intermediate_frames_semantically_valid=false` enforced in UI with amber warning badge.
- Initial saved-view object containing basis, representation, metric, labels, and provenance.
- Intrinsic geodesic interpolation (GLERP) active as default Grand Tour and Subspace Optimization trajectory generator.

### Automated tests

- \(F^\top F\approx I\).
- Projected shape is \(N\times2\).
- Grassmannian distance is zero for bases spanning the same plane.
- Distance is invariant to in-plane orthogonal basis changes.
- Symmetry and expected principal-angle bounds.
- Repeated or zero-distance keyframes do not produce invalid frames.
- A `linear_projection` path cannot contain arbitrary 2D coordinates without a basis.
- A `sequential_embedding` path defaults to `intermediate_frames_semantically_valid=false`.
- A `representation_morph` path defaults to `intermediate_frames_semantically_valid=false`.
- A basis from one representation is rejected when passed to a different representation's projector.
- Saved view round trip reproduces the coordinates to tolerance.
- ViewSpec records match the required provenance fields (representation hash, eigenvalues, etc.).

### Manual verification

1. Render the `probability` representation through its own PCA little tour.
2. Render the `sqrt_probability` representation through its own separately fitted PCA little tour.
3. Confirm that the two tours' keyframes are different objects in provenance.
4. Rotate a basis within its own plane and verify only screen orientation changes.
5. Switch between guided, manual, and grand modes.
6. Pause between keyframes and inspect the semantic badge.
7. Load a deliberately independent embedding sequence and confirm the warning changes.
8. Trigger a `representation_morph` between two representations; confirm the intermediate-frame warning is present.
9. Save, reload, and compare a view.

### Edge cases

- \(p<2\), \(p=2\), and rank-deficient matrices.
- Duplicate PCA eigenvalues.
- Sign flips and basis swaps.
- Antipodal or nearly orthogonal subspaces.
- Repeated keyframes.
- Very small or very large coordinate scales.
- Closed loops with an orientation discontinuity.
- Applying a probability-space basis to a sqrt-probability matrix of the same shape.

### Questions answered

- Is projection-space navigation legible enough to sustain interaction?
- Which controls create useful intentional movement rather than visual noise?
- Can the interface reliably communicate path semantics, including representation-morph warnings?
- Does `dtour` provide enough hooks for the MVP?

### Exit gate

The user can traverse a deterministic tour, select objects without identity loss, save a view, and explain whether every intermediate frame is analytically meaningful. Per-representation PCA tours produce distinct, correctly labelled provenance records. The minimal Flask shell is running and serves the workbench page.

---

## Sprint 3b — Minimal Flask workbench shell

### Objective

Stand up the Flask application that will host all custom UI panels going forward, before those panels become complex. The Python math core is unchanged; Flask is a thin serving layer around it.

### Rationale

Building integrity overlays, semantic badges, source inspectors, and saved-view panels inside Jupyter notebook output cells requires fighting `ipywidgets` layout constraints. A Flask shell with plain HTML and CSS is cleaner, more maintainable, and plays to the team's existing skills. The scatter plot rendering stays with `@dtour/viewer` (the JavaScript package); Flask owns everything surrounding it.

### Deliverables

- `src/shadowspace/server/` — Flask application package (stub exists from Sprint 0):
  - `src/shadowspace/server/__init__.py` — Flask factory `create_app()` (expand from Sprint 0 stub).
  - `src/shadowspace/server/routes.py` — routes for the workbench page and data API.
  - `src/shadowspace/server/templates/workbench.html` — main workbench shell.
  - `src/shadowspace/server/static/style.css` — base styles.
  - `src/shadowspace/server/static/main.js` — minimal JS glue to mount `@dtour/viewer`.
- `app.py` at the repo root already exists and delegates to `create_app()`; no changes needed.
- Flask serves the bundle's Parquet representation table as an Arrow IPC or JSON endpoint.
- `@dtour/viewer` script tag loads from CDN or local copy; receives data from the Flask endpoint.
- Placeholder panels (empty `<section>` elements with IDs) for:
  - integrity overlay;
  - semantic badge;
  - source object inspector;
  - saved-view atlas.
- `flask run` launches the workbench with the Sprint 0 calibration fixture.
- `requirements.txt` updated with `flask`.

### Automated tests

- Flask test client returns 200 for the workbench route.
- Data endpoint returns valid Arrow IPC or JSON for the calibration fixture.
- Panel placeholder IDs are present in the rendered HTML.

### Manual verification

1. Run `flask run`.
2. Open the workbench in a browser.
3. Confirm `@dtour/viewer` renders the calibration fixture scatter plot.
4. Confirm placeholder panel sections are visible in the layout.
5. Confirm a page refresh does not change the fixture data.

### Edge cases

- Browser without WebGPU (dtour falls back to WebGL).
- Large bundle that exceeds reasonable JSON size (use Arrow IPC or streaming).
- Port conflict on `flask run`.

### Questions answered

- Does `@dtour/viewer` embed cleanly in a Flask template without a React build step?
- Is the Flask data API fast enough for the calibration fixture?
- Which panel layout fits the workbench tasks without clutter?

### Exit gate

`flask run` renders the calibration fixture in `@dtour/viewer` with four clearly identified placeholder panels. The Jupyter notebook from Sprint 0 is kept for headless math validation but is no longer the primary interface.

---

## Sprint 4 — Local integrity diagnostics

### Objective

Make the selected object's high-dimensional relationships visible in the current projection.

### Definitions

For selected point \(i\) and neighborhood size \(k\):

- **preserved neighbor:** in both source-space and projected-space \(k\)-NN;
- **torn neighbor:** source-space neighbor absent from projected \(k\)-NN;
- **false neighbor:** projected neighbor absent from source-space \(k\)-NN.

### Deliverables

- Exact \(k\)-NN implementation for small fixtures.
- Precomputed or approximate neighbor graphs for larger data.
- Selected-point diagnostic panel.
- Focus-plus-context links:
  - preserved;
  - torn;
  - false.
- Local precision, recall, and overlap summaries.
- Metric and representation shown beside every diagnostic.
- Controls for \(k\), with a conservative default.
- Optional current-view trustworthiness/continuity summaries.

### Automated tests

- Exact known-neighbor fixtures.
- Tie handling is deterministic.
- Diagnostics use matching IDs, representation, and metric.
- Preserved/torn/false sets partition the relevant unions correctly.
- No self-neighbor unless explicitly requested.
- Approximate graph recall is measured against exact graphs on sampled subsets.
- Projection changes update projected neighbors while source neighbors remain fixed.
- Representation or metric changes invalidate incompatible caches.

### Manual verification

1. Use the exact three-class reference and select a known corner point.
2. Force a projection that collapses two distinct directions.
3. Confirm false and torn links appear as expected.
4. Switch \(k\) and inspect whether the explanation remains stable.
5. Switch representation and verify the source-neighbor set is recomputed and relabeled.
6. Select duplicate points and inspect tie behavior.
7. Turn overlays off and on without moving points.

### Edge cases

- Duplicates and equal-distance ties.
- \(k\ge N\).
- Tiny datasets.
- All-identical points.
- A selected point filtered out of view.
- Neighbor graph built with an obsolete representation hash.
- Visual overload from many links.
- Missing source payload.

### Questions answered

- Can users distinguish “looks close” from “is close under the declared geometry”?
- Which overlay vocabulary is understandable without mathematical training?
- What neighborhood sizes are useful?
- Is local feedback responsive enough during a tour?

### Exit gate

On planted fixtures, the interface identifies false and torn neighbors correctly and updates them without stale-cache errors. A user can inspect one point and state why at least one visible relationship is trustworthy or questionable.

---

## Sprint 5 — Integrated research MVP

### Objective

Join the data, geometry, tours, diagnostics, and source inspector into one coherent workbench.

### Deliverables

- One-command launch.
- Main projection view.
- Representation and metric selectors.
- Path-semantic status.
- Source-object inspector.
- Probability bars.
- selected-point integrity overlays.
- saved-view atlas with names and notes.
- exportable investigation record.
- guided tutorial using the synthetic dataset.
- reduced-motion mode and keyboard controls.

### Investigation record

Every saved finding must include:

- bundle and schema version;
- artifact hashes;
- source representation and metric;
- projection basis or embedding identifier;
- path semantics;
- selected and filtered IDs;
- neighborhood \(k\);
- random seed;
- code and dependency versions;
- user note and timestamp.

### Automated tests

- End-to-end bundle load through view save and reload.
- Cross-representation selection persistence.
- State serialization.
- Semantic warning tests.
- Accessibility state tests where automatable.
- Snapshot tests for explanatory text and schema, not unstable pixel layout.

### Manual verification

Run the documented “ten-minute investigation”:

1. Locate a planted bridge.
2. Save the view.
3. inspect three bridge points.
4. reveal false and torn neighbors.
5. switch representation.
6. determine whether the bridge persists.
7. restore the first view.
8. export the finding.
9. reload it in a fresh session.
10. explain what is established and what remains uncertain.

### Edge cases

- Changing representation while a tour is playing.
- Loading a saved view after filters change.
- Selection with missing payload.
- Attempting an incompatible metric.
- Export from an intermediate embedding-morph frame.
- Reduced-motion mode during automated tours.
- Browser/notebook restart.

### Questions answered

- Does the combined workflow support a real investigation?
- Are users learning about the data, or merely watching animation?
- Which explanations are essential and which are clutter?
- Is the notebook workbench sufficient for an initial public prototype?

### Exit gate

A user unfamiliar with the code can complete the guided investigation and produce a reproducible finding that includes an explicit limitation statement.

---

## Sprint 6 — Four-class validation and optional exact reference

### Objective

Test Shadowspace against a higher-dimensional simplex whose generating truth is known.

### Deliverables

- Four-class controlled experiment.
- Comparative report across representations and metrics.
- Projection catalog:
  - view revealing corner groups;
  - view hiding the bridge;
  - view creating false separation;
  - view emphasizing entropy.
- Optional implementation of the published lossless Simplex Projection as a reference, isolated behind its own module.
- Regression fixtures built from deliberately misleading views.

### Automated tests

- All planted structures are recoverable under at least one declared view.
- Deliberately misleading views trigger expected diagnostics.
- If implemented, exact-reference encode/decode round trip.
- Reference and Shadowspace use identical IDs and source distributions.

### Manual verification

Ask a tester to classify each visible feature as:

- persistent;
- representation-dependent;
- projection-dependent;
- unsupported;
- unresolved.

Compare the answer with generator truth.

### Questions answered

- Does moving projection plus integrity feedback outperform a static chart for controlled tasks?
- Which mathematical distinctions survive contact with an actual interface?
- Is a specialized four-part composition view additive or distracting?
- What should become the baseline condition in later studies?

### Exit gate

The team can name at least three repeatable cases where Shadowspace prevents or corrects a plausible misinterpretation. Otherwise, revise the core interaction before adding real data.

---

## Sprint 7 — Fashion-MNIST belief space

### Objective

Demonstrate the method on recognizable source objects and real model outputs.

### Deliverables

- Reproducible small classifier training or pinned checkpoint.
- Train/test split provenance.
- Exported images, logits, probabilities, entropy, labels, predictions, and correctness.
- Square-root and smoothed CLR representations.
- Source-image inspector.
- Curated investigations:
  - shirt/pullover/coat ambiguity;
  - sneaker/ankle-boot ambiguity;
  - confident errors;
  - high-entropy regions;
  - model-belief bridges.
- Calibration metrics recorded separately from geometric displays.

### Automated tests

- Dataset counts and label mapping.
- Checkpoint hash and deterministic inference tolerance.
- `softmax(logits)` equals stored probabilities.
- Representation invariants.
- No train/test leakage in exported metadata.
- Payload references resolve.
- Nearest-neighbor caches match the current model and representation hash.

### Manual verification

1. Select a confident correct prediction.
2. Select a confident error.
3. Compare their neighbors in logits and probability geometry.
4. Inspect ambiguous images along a visible bridge.
5. verify that classes are not interpreted as an ordered transport space.
6. Save and replay a finding from a clean environment.

### Edge cases

- Saturated one-hot-like outputs.
- Miscalibrated confidence.
- Numerically tiny probabilities.
- Incorrect or unavailable image payload.
- Model retraining changing IDs or outputs.
- Class imbalance in filtered subsets.
- Color labels creating false confidence.

### Questions answered

- Do real source objects make the geometry easier to understand?
- Are representation changes substantively useful rather than decorative?
- Can the prototype generate questions about a model that static plots miss?
- What latency and bundle-size constraints appear at 70,000 objects?

### Exit gate

At least two model-behavior findings are reproducible, source-grounded, and not based solely on class-color separation.

---

## Sprint 8 — Flask application expansion decision

### Objective

The minimal Flask shell has been running since Sprint 3b. This sprint decides whether to expand it into a fuller standalone application, contribute hooks upstream to `dtour`, or hold at the current workbench level.

### Decision evidence

- user-test observations from Experiments 1 and 2;
- interaction hooks that cannot be satisfied through `@dtour/viewer`'s public API;
- performance traces at 70,000 objects;
- reproducibility and sharing needs;
- deployment audience (local research tool vs. hosted demo);
- maintenance cost of expanding Flask vs. contributing upstream;
- whether a React build step is justified by interaction requirements.

### Possible outcomes

1. **Flask workbench is sufficient as-is** — polish the existing HTML/CSS panels and ship.
2. **Expand Flask with richer HTML/CSS panels** — more complex source inspector, animated overlays, saved-view gallery; no additional JS framework needed.
3. **Add a targeted React component** — only if a specific interaction (e.g., the integrity overlay canvas layer) cannot be built cleanly in vanilla JS.
4. **Switch `@dtour/viewer` to `@dtour/scatter`** — lower-level rendering control if the viewer's API is too constraining.
5. **Contribute a hook upstream to `dtour`** — if the gap is small and general enough to benefit the broader `dtour` community.
6. **Stop or redirect** — if Experiments 1–2 show the core interaction did not improve reasoning.

### Exit gate

A written architecture decision records the chosen outcome and the evidence. The Flask workbench already exists; this sprint either extends it or consciously freezes it. Do not add a JS build pipeline or new framework merely because it is technically interesting.

---

## Sprint 9+ — Research hardening

Potential work is ordered by evidence, not novelty alone:

1. stability/Rashomon atlas;
2. question-driven multi-objective projection search;
3. moving regional distortion and gap diagnostics;
4. model-comparison module;
5. formal user study;
6. scientific ensemble or function-space module;
7. intrinsic Grassmannian path experiments;
8. counterfactual trajectories.

See [Research Roadmap](RESEARCH_ROADMAP.md).

---

## 3. Cross-sprint engineering rules

### Definition of done

A feature is done only when it has:

- typed public API;
- automated tests;
- domain and failure documentation;
- deterministic fixture;
- manual verification step;
- provenance in exported artifacts;
- no silent fallback that changes mathematical meaning.

### Feature flags

Experimental mathematics should be opt-in:

- `experimental_intrinsic_tour`;
- `experimental_gap_index`;
- `experimental_simplex_projection`;
- `experimental_rashomon_atlas`.

An experiment must not silently become the default.

### Version discipline

- Pin `dtour` during the prototype.
- Keep it behind `DtourAdapter`.
- Record Python, package, browser, and bundle schema versions.
- Treat the GitHub repository's Apache-2.0 license as the governing upstream license unless clarified; verify metadata before redistribution.
- Upgrade dependencies in dedicated changes with smoke and visual tests.

### Stop conditions

Pause feature development when:

- users cannot distinguish projection from embedding semantics;
- diagnostics cannot be explained with known fixtures;
- representation switching produces no useful questions;
- an upstream tool already solves the intended contribution;
- performance work is proposed without a measured bottleneck;
- a real-data finding cannot be traced to source objects and provenance.
