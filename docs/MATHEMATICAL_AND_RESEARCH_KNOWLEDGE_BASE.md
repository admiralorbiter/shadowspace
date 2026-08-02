# Shadowspace Project Decisions

This file records durable decisions and the evidence that should trigger reconsideration.

## ADR-001 — Build a research instrument, not a game

**Status:** accepted

Shadowspace uses game-like navigation and discovery, but its purpose is reliable intuition and analysis of high-dimensional spaces.

**Consequences**

- Every interaction must expose mathematical meaning.
- Success is measured by better reasoning, not time-on-task alone.
- Animation, rewards, or challenges may be added only when they reinforce interpretation.

**Revisit when:** a separate educational game mode has a defined audience and does not compromise the analytical interface.

---

## ADR-002 — Projection is the primary interaction

**Status:** accepted

The user navigates the space of views. A fixed UMAP or PCA chart is a reference, not the whole product.

**Consequences**

- Projection basis and path are first-class data.
- View replay and provenance are required.
- The interface must preserve point identity during motion.

**Revisit when:** user testing shows that continuous movement prevents rather than improves understanding.

---

## ADR-003 — Use `dtour` before building a renderer

**Status:** accepted

`dtour` already provides guided, manual, and grand tours, browser-scale rendering, Python integration, and lower-level JavaScript packages.

**Consequences**

- Pin the dependency during the prototype.
- Wrap it behind `DtourAdapter`.
- Do not fork or copy private internals.
- Treat the upstream GitHub **MIT** license as authoritative; include the dtour copyright notice in `THIRD_PARTY_NOTICES.txt` when distributing.
- Build a custom UI only after an evidence-based gate.

**Revisit when:** required selection, overlay, state, or semantic hooks cannot be implemented through public APIs.

---

## ADR-004 — Keep renderer-independent mathematical core

**Status:** accepted

Data, geometry, metrics, views, diagnostics, and provenance must work without a notebook widget.

**Consequences**

- Core accepts arrays, tables, IDs, and schema objects.
- Renderer types stay in `adapters/`.
- Tests can verify mathematics headlessly.
- A later frontend does not force a data rewrite.

**Revisit when:** never; only the exact boundary may change.

---

## ADR-005 — Make representation and metric first-class

**Status:** accepted

The same source objects can have multiple coordinate systems and geometries. There is no universal neighbor graph.

**Consequences**

- Every diagnostic declares representation and metric.
- Switching representation invalidates incompatible caches.
- Controls explain assumptions, zero handling, and path meaning.
- Cross-representation comparisons preserve source IDs.

**Revisit when:** a domain has one mathematically mandated geometry, in which case alternatives may be hidden but still recorded.

---

## ADR-006 — Probability space is the first domain

**Status:** accepted

Probability vectors are common, constrained, mathematically rich, easy to synthesize, and directly produced by classifiers.

**Sequence**

1. three-class exact calibration;
2. four-class controlled synthetic world;
3. Fashion-MNIST classifier beliefs.

**Revisit when:** synthetic tests show no useful distinction among the selected representations or the interface cannot explain those distinctions.

---

## ADR-007 — Semantic path types are release-blocking

**Status:** accepted

A smooth animation may connect valid projections, independent embeddings, domain states, or representations. These are not interchangeable.

**Consequences**

- Every path has a declared type.
- Intermediate-frame validity is explicit.
- Sequential-embedding and representation morphs default to endpoints-only evidence.
- Findings exported from nonanalytical frames retain warnings.

**Revisit when:** never; new path types may be added, but unlabeled paths remain invalid.

---

## ADR-008 — Reproducible artifact bundles are the durable interface

**Status:** accepted

The bundle, not a notebook's hidden state, is the unit of exchange.

**Consequences**

- Stable IDs, manifests, hashes, schemas, and provenance.
- Arrow/Parquet for tables; NPZ initially for bases.
- Findings can be replayed independently.
- Cache keys include all semantic inputs.

**Revisit when:** a more suitable open standard can represent the same semantics without loss.

---

## ADR-009 — Start with local selected-point integrity

**Status:** accepted

Global distortion scores are useful but too abstract. Full-plot overlays can overwhelm.

**Consequences**

- MVP uses preserved, false, and torn neighbors around a selection.
- Global metrics are context, not verdicts.
- Gap, topology, and regional overlays remain experimental.

**Revisit when:** formative tests identify a clearer or less cluttered local explanation.

---

## ADR-010 — No Rust, custom GPU work, or 3D view in the MVP

**Status:** accepted

These add engineering scope without addressing the main uncertainty: whether projection navigation plus integrity feedback improves reasoning.

**Revisit when:**

- a measured Python or browser bottleneck cannot be solved within the existing stack;
- the 2D interaction is proven and a 3D explanatory inset has a specific tested role;
- performance requirements exceed upstream renderer capabilities.

---

## ADR-011 — Controlled truth before real data

**Status:** accepted

Real datasets rarely reveal whether a visible structure is genuine. Synthetic worlds provide known answers.

**Consequences**

- Every new diagnostic begins with a planted success and failure case.
- Fashion-MNIST follows, rather than replaces, exact fixtures.
- A visually interesting real-data example is not accepted as validation by itself.

**Revisit when:** never; real domains may add other validation sources, but known fixtures remain required.

---

## ADR-012 — Exploration is not confirmation

**Status:** accepted

Shadowspace generates and qualifies hypotheses. Formal domain claims require independent analysis.

**Consequences**

- UI language uses “suggests,” “persists under,” and “unsupported under,” not “proves.”
- Saved findings include limitations.
- Class labels and guided objectives are identified as analytical inputs.
- Study protocols separate exploratory and confirmatory phases.

**Revisit when:** a specific, pre-registered confirmatory workflow is designed and statistically validated.

---

## ADR-013 — Fisher–Rao distance convention

**Status:** accepted

Shadowspace uses the factor-of-two Fisher–Rao convention:

\[
d_{\mathrm{FR}}(p,q)=2\arccos\!\left(\sum_i\sqrt{p_i q_i}\right).
\]

This corresponds to the square-root embedding \(p\mapsto 2\sqrt{p}\) on a sphere of radius 2 and the ordinary unscaled Fisher information metric. Distinct simplex vertices are at distance \(\pi\).

The no-factor-two value is exposed separately as:

```python
bhattacharyya_angle(p, q)  # = arccos(BC(p, q))
```

It is not another silently scaled implementation of `fisher_rao`. The source of truth is:

```python
# src/shadowspace/conventions.py
FISHER_RAO_CONVENTION = "canonical_fisher_information"
FISHER_RAO_SCALE = 2.0
```

Manifests, saved findings, tests, and displayed values must copy or reference these constants.

**Consequences**

- `fisher_rao` always uses the factor-of-two form.
- `bhattacharyya_angle` is a named alias for the no-factor-two form.
- Using a different scaling convention requires a new named metric id, not a parameter.

**Revisit when:** an upstream reference with wide adoption uses a different convention and the distinction creates significant confusion in published comparisons.

---

## ADR-014 — CLR zero policy

**Status:** accepted

The default zero treatment for centered log-ratio coordinates is multiplicative replacement:

\[
p_i^* = \begin{cases}
\delta, & p_i = 0, \\
(1 - m\delta)\,p_i, & p_i > 0,
\end{cases}
\]

where \(m\) is the count of exact zeros and \(\delta\) is the replacement value. This preserves ratios among originally positive components.

The source of truth is:

```python
# src/shadowspace/conventions.py
CLR_ZERO_POLICY   = "multiplicative_replacement"
CLR_ZERO_DELTA    = 1e-6
CLR_ZERO_MATCH    = "exact_zero_only"
```

The implementation must:

- Preserve the original zero mask.
- Record replacement counts and \(\delta\) in provenance.
- Leave small positive values untouched.
- Reject negative and nonfinite inputs.
- Reject inputs where \(m\delta \geq 1\).
- Support later sensitivity analysis over alternative policies via named policy identifiers.

This is explicitly a **prototype visualization convention**, not a universally correct treatment of structural, censored, or sampling zeros.

**Consequences**

- CLR with untreated exact zeros raises an error; it does not silently modify data.
- Every CLR result in a bundle records `zero_policy`, `zero_delta`, `zero_count`, and `zero_match` in its `RepresentationSpec`.
- Alternative zero policies (e.g., Bayesian-Laplace, detection-limit replacement) may be added as named options without changing the default.

**Revisit when:** domain data has structural zeros where multiplicative replacement is scientifically indefensible, or a different policy becomes the published standard for compositional probability data.

---

## ADR-015 — Active Representation Mean-Centering & Dynamic Basis Refitting

**Status:** accepted

When reprojecting catalog bases or computing 2D coordinates across representations (`probability`, `sqrt_probability`, `clr_probability`, `logits`), the projection center $\mu_{\text{rep}} = \operatorname{mean}(X_{\text{rep}})$ is computed directly in the active representation feature space ($Y = (X_{\text{rep}} - \mu_{\text{rep}}) F$).

For supervised/discriminative view catalogs (`fisher_lda`), the discriminative basis is dynamically refitted on the active representation matrix so class separation is optimized in that feature space.

**Consequences**

- 2D catalog projections maintain origin stability and basis orthonormality across all representation switches.
- Fisher LDA bases continuously adapt to representation transformations.
- `SavedView` snapshots preserve full basis matrices, reprojected coordinates, and matrix SHA-256 hashes in `metadata`.

---

## ADR-016 — SQLite-First Single-File Bundle Storage (`sqlite-vec`)

**Status:** accepted

For high-volume dataset scaling, Shadowspace uses a single-file `.db` SQLite bundle architecture powered by `sqlite-vec` (v0.1.9) for zero-infrastructure vector similarity search.

**Consequences**

- Dense representation matrices are stored as binary float64 BLOBs for fast full-matrix reads.
- Virtual tables (`vec_{rep_id}`) enable C-level exact brute-force similarity search directly inside SQLite.
- Eliminates multi-file Parquet directory complexity for Tier 3 scale datasets.

