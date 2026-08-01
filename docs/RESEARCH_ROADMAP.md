# Shadowspace Architecture and Data Contract

## 1. Architectural goal

Shadowspace must separate four concerns that visualization systems often collapse:

1. **Source objects:** the things being studied.
2. **Representations and geometry:** how those objects become coordinates and how distance is defined.
3. **Views and paths:** how coordinates are mapped to the screen and how that map changes.
4. **Diagnostics and evidence:** what the current view preserves, hides, or invents.

The renderer is replaceable. The artifact bundle and semantic model are the durable core.

## 2. System boundaries

```text
Source data / model
        │
        ▼
┌───────────────────────┐
│ Python preparation    │
│ objects, features,    │
│ transforms, metrics   │
└──────────┬────────────┘
           │ Shadowspace Artifact Bundle
           ▼
┌───────────────────────┐
│ Analysis core         │
│ tours, neighbors,     │
│ diagnostics, views    │
└──────────┬────────────┘
           │ adapter contract
           ▼
┌───────────────────────┐
│ Interactive renderer  │
│ dtour notebook first  │
│ browser application   │
│ only after MVP gate   │
└───────────────────────┘
```

No renderer-specific object should leak into the mathematical core.

## 3. Domain model

### `SourceObject`

One persistent identity across every representation and view.

```python
class SourceObject:
    id: str
    metadata: dict[str, JsonValue]
    payload_ref: str | None
```

Rules:

- IDs are unique and immutable within a bundle.
- Selection is keyed by ID, never by transient row number.
- Metadata may change between bundle versions; identity may not be silently reused.
- Payloads can be images, documents, functions, or external references.

### `RepresentationSpec`

Describes one coordinate representation of all or a declared subset of source objects.

```python
class RepresentationSpec:
    id: str
    dimension: int
    path: str
    object_id_column: str
    feature_columns: list[str]
    constraints: list[str]
    transform: TransformProvenance
    compatible_metrics: list[str]
    default_metric: str
    zero_policy: ZeroPolicy | None
```

Examples:

- raw class probabilities;
- square-root probabilities;
- centered log-ratio probabilities;
- logits;
- learned activations.

A representation is not “just another view.” Changing representation may change geometry and neighbor identity.

### `MetricSpec`

```python
class MetricSpec:
    id: str
    display_name: str
    representation_ids: list[str]
    is_metric: bool
    parameters: dict[str, JsonValue]
    units_or_scale: str | None
    implementation_version: str
```

Rules:

- incompatible representation/metric pairs fail explicitly;
- “distance” functions that do not satisfy metric axioms declare `is_metric=false`;
- every result records parameters such as smoothing, logarithm base, and neighborhood size;
- Wasserstein-type metrics require a documented ground cost.

### `ViewSpec`

A static displayed state.

```python
class ViewSpec:
    id: str
    representation_id: str
    kind: Literal["linear_projection", "embedding"]
    basis_ref: str | None
    coordinates_ref: str | None
    display_transform: DisplayTransform
    created_by: str
    seed: int | None
    provenance: dict[str, JsonValue]
```

For a linear projection, `basis_ref` is required and points to a \(p\times2\) orthonormal basis. For an embedding, `coordinates_ref` is required.

A rotation or reflection applied only for screen layout belongs in `display_transform`; it must not masquerade as a new mathematical view.

### `PathSpec`

A sequence or continuous family of views.

```python
class PathSpec:
    id: str
    kind: Literal[
        "linear_projection",
        "sequential_embedding",
        "domain_geodesic",
        "representation_transition",
        "representation_morph",
    ]
    keyframes: list[str]
    interpolation_method: str
    intermediate_frames_semantically_valid: bool
    semantics_note: str
    pacing_metric: str | None
```

Enforced defaults:

| Kind | Default semantic validity |
|---|---|
| `linear_projection` | `true`, provided every frame has a valid basis |
| `sequential_embedding` | `false` |
| `domain_geodesic` | `true` only with a declared domain implementation |
| `representation_transition` | `false` unless analytically justified |
| `representation_morph` | `false`; Procrustes alignment may be applied for smoothness but does not validate intermediate frames |

The interface must display the kind and validity during playback.

### `DiagnosticSpec`

```python
class DiagnosticSpec:
    id: str
    source_representation_id: str
    source_metric_id: str
    view_id: str
    method: str
    parameters: dict[str, JsonValue]
    artifact_ref: str
```

Diagnostics are always relative to a declared source representation and metric. There is no representation-free “true neighborhood.”

## 4. Artifact bundle

A bundle is portable, immutable once published, and self-describing.

```text
bundle/
├── manifest.json
├── objects.parquet
├── representations/
│   ├── probability.parquet
│   ├── sqrt_probability.parquet
│   └── clr_probability.parquet
├── neighbors/
│   ├── probability/
│   │   ├── euclidean.parquet
│   │   └── fisher_rao.parquet
│   └── clr_probability/
│       └── aitchison.parquet
├── views/
│   ├── pca_little_tour.npz
│   └── curated_views.npz
├── diagnostics/
├── payloads/
└── README.md
```

### `objects.parquet`

Recommended columns:

```text
object_id: string
payload_ref: string?
split: string?
generator_component: string?
true_label: string?
predicted_label: string?
correct: bool?
entropy: float?
...domain metadata
```

### Representation tables

Use wide tables for the first implementation because they map directly to numerical matrices and `dtour`:

```text
object_id
feature_000
feature_001
...
feature_p_minus_1
```

The object ID order must exactly match the canonical order or the reader must reorder explicitly and log it. Silent positional joins are prohibited.

### Neighbor tables

Long form:

```text
object_id
neighbor_id
rank
distance
```

Metadata in the manifest records:

- source representation;
- metric and parameters;
- exact or approximate algorithm;
- \(k\);
- random seed;
- implementation version;
- source artifact hash;
- approximate recall estimate, if applicable.

### View basis storage

A compressed NumPy archive is sufficient initially:

```text
bases: float64[K, p, 2]
view_ids: string[K]
```

Float32 may be exported to the browser, but the preparation and validation path should retain float64.

## 5. Example manifest

```json
{
  "schema_version": "0.1.0",
  "bundle_id": "synthetic-beliefs-4c-v1",
  "created_at": "2026-08-01T00:00:00Z",
  "object_table": {
    "path": "objects.parquet",
    "object_count": 5000,
    "sha256": "..."
  },
  "representations": [
    {
      "id": "probability",
      "path": "representations/probability.parquet",
      "dimension": 4,
      "constraints": ["finite", "nonnegative", "row_sum_1"],
      "compatible_metrics": [
        "euclidean",
        "hellinger",
        "fisher_rao",
        "jensen_shannon"
      ],
      "default_metric": "fisher_rao",
      "sha256": "..."
    },
    {
      "id": "clr_probability",
      "path": "representations/clr_probability.parquet",
      "dimension": 4,
      "constraints": ["finite", "row_sum_0"],
      "compatible_metrics": ["aitchison"],
      "default_metric": "aitchison",
      "zero_policy": {
        "method": "multiplicative_replacement",
        "parameter": 1e-6
      },
      "sha256": "..."
    }
  ],
  "paths": [
    {
      "id": "sqrt-pca-little-v1",
      "kind": "linear_projection",
      "representation_id": "sqrt_probability",
      "keyframe_ref": "views/pca_little_tour.npz",
      "interpolation_method": "dtour_default",
      "intermediate_frames_semantically_valid": true,
      "semantics_note": "Every rendered frame is a linear projection basis."
    }
  ],
  "provenance": {
    "generator": "shadowspace.synthetic:1",
    "seed": 20260801,
    "git_commit": "...",
    "python": "3.11.x",
    "dependency_lock_sha256": "..."
  }
}
```

## 6. Package modules

```text
src/shadowspace/
├── data/
│   ├── synthetic.py
│   └── fashion_mnist.py
├── representations/
│   ├── base.py
│   └── probability.py
├── metrics/
│   ├── base.py
│   └── probability.py
├── geometry/
│   ├── bases.py
│   └── grassmann.py
├── tours/
│   ├── paths.py
│   ├── pca.py
│   └── saved_views.py
├── diagnostics/
│   ├── neighbors.py
│   ├── local_integrity.py
│   └── stability.py
├── bundles/
│   ├── schema.py
│   ├── reader.py
│   ├── writer.py
│   └── validate.py
└── adapters/
    └── dtour.py
```

Dependencies point inward:

- adapters depend on core;
- core never depends on notebook or browser types;
- bundle schemas do not import `dtour`;
- diagnostics accept arrays and IDs, not widgets.

## 7. Core interfaces

```python
class Representation(Protocol):
    spec: RepresentationSpec
    def transform(self, objects: ObjectTable) -> MatrixWithIds: ...

class Metric(Protocol):
    spec: MetricSpec
    def pairwise(self, x: NDArray, y: NDArray | None = None) -> NDArray: ...
    def distances_from(self, x: NDArray, index: int) -> NDArray: ...

class Projector(Protocol):
    def project(self, matrix: NDArray, basis: NDArray) -> NDArray: ...

class TourProvider(Protocol):
    def keyframes(self, matrix: NDArray, seed: int | None) -> list[ViewSpec]: ...

class RendererAdapter(Protocol):
    def load(self, objects, representation, path) -> None: ...
    def set_selection(self, object_ids: set[str]) -> None: ...
    def current_state(self) -> ViewState: ...
```

Public APIs validate at boundaries. Internal numerical kernels can assume validated arrays for speed.

## 8. Cache rules

A cache key must include all inputs that affect meaning:

```text
bundle_hash
representation_hash
metric_id + parameters
algorithm_id + version
k
random_seed
filter_hash
```

Changing representation, zero policy, model checkpoint, or metric parameters invalidates neighbor and diagnostic caches.

A projected-neighbor cache additionally includes the view basis or coordinate hash.

## 9. Projection and display transformations

For a valid basis \(F\):

\[
Y=XF.
\]

The plane represented by \(F\) is unchanged by right multiplication with a \(2\times2\) orthogonal matrix \(R\):

\[
\operatorname{span}(F)=\operatorname{span}(FR).
\]

The plotted coordinates become \(YR\), a display rotation or reflection. This distinction is important:

- `basis` defines a subspace;
- `display_transform` defines chart orientation;
- saved views should preserve both when exact replay matters;
- Grassmannian comparisons ignore in-plane orientation.

## 10. `dtour` integration policy

`dtour` is an upstream rendering and interaction dependency, not the source of truth for Shadowspace semantics.

`dtour` is released under the **MIT license**. Include the dtour copyright notice in `THIRD_PARTY_NOTICES.txt` when distributing Shadowspace.

The adapter is responsible for:

- converting tables to supported Arrow/Parquet or dataframe formats;
- converting keyframes to \(p\times2\) bases;
- mapping selections to stable object IDs;
- recording current view state;
- surfacing whether an intermediate frame is valid;
- normalizing upstream API differences.

Do not:

- subclass private widget internals;
- rely on undocumented row-order behavior;
- store Shadowspace metadata only inside widget state;
- assume a `dtour` animation is an intrinsic Grassmannian geodesic;
- fork before an adapter-based integration has failed.

## 11. Provenance and reproducibility

Every exported finding should be replayable. At minimum record:

- bundle ID and SHA-256 hashes;
- schema version;
- source representation and metric;
- zero policy;
- model or generator version;
- basis or embedding coordinates;
- filter and selected IDs;
- neighborhood parameters;
- path type and semantic validity;
- seed;
- software versions;
- user annotation.

A screenshot without this record is an illustration, not a reproducible result.

## 12. Privacy and data governance

The initial datasets are public or synthetic, but the architecture should be safe for later sensitive data:

- payloads may remain outside the bundle;
- IDs should support pseudonymous values;
- exports should declare whether source payloads are included;
- no automatic remote upload;
- derived embeddings may still leak information and must not be treated as anonymized by default;
- bundle manifests should allow license, consent, and retention metadata.

## 13. Schema evolution

Use semantic versions:

- patch: metadata additions that older readers may ignore;
- minor: backward-compatible optional fields;
- major: incompatible meaning or file-layout changes.

Readers should:

- reject unsupported major versions;
- warn on newer minor versions;
- validate hashes before loading;
- never guess the semantics of an unknown path or metric type.
