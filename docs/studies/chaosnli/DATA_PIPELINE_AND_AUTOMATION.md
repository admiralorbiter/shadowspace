# ChaosNLI Data Pipeline and Automation Plan

## 1. Objective

Convert third-party data and model outputs into:

1. validated canonical tables;
2. uncertainty-aware human distributions;
3. versioned model distributions;
4. exact neighbor graphs and comparison metrics;
5. automatically selected review cases;
6. deterministic HTML/PNG review packets;
7. Shadowspace bundles;
8. a complete reproducibility manifest.

No scientific result should depend on manually clicking through the workbench.

---

## 2. Acquisition

### 2.1 ChaosNLI

```bash
git clone https://github.com/easonnie/ChaosNLI.git vendor/ChaosNLI
```

Expected data:

```text
chaosNLI_snli.jsonl
chaosNLI_mnli_m.jsonl
chaosNLI_alphanli.jsonl
```

Primary analysis uses SNLI and MNLI only. The official repository also documents supplied model prediction files.

### 2.2 External validation

Acquire separately and record license/version/ID mapping:

- NLI disagreement taxonomy;
- VariErr NLI;
- LiTEx annotations where applicable;
- LiveNLI;
- optional formal-semantic tags;
- original SNLI/MNLI metadata.

Do not assume overlap. Emit a join audit:

```text
source
rows
unique IDs
matched to ChaosNLI
unmatched
duplicate matches
text-exact matches
text-normalized matches
manual-review matches
```

### 2.3 Source manifest

For every file:

```yaml
source_id:
source_type:
upstream_revision:
downloaded_at_utc:
original_filename:
sha256:
byte_size:
license:
redistribution_allowed:
citation:
acquisition_command:
```

Never silently replace raw data. Changed hashes create new source versions.

---

## 3. Canonical object schema

One row per NLI item:

```text
object_id
source_dataset
source_pair_id
premise
hypothesis
genre
original_gold_label
original_labels_json
human_count_entailment
human_count_neutral
human_count_contradiction
human_p_entailment
human_p_neutral
human_p_contradiction
human_entropy_bits
human_majority_label
human_majority_count
human_agreement_rate
has_zero_count
taxonomy_labels_json
taxonomy_high_level_json
varierr_has_ambiguity
varierr_error_labels_json
varierr_valid_label_set_json
split
```

Rules:

- preserve source text exactly;
- normalize labels only in derived columns;
- enforce count total 100;
- enforce `[entailment, neutral, contradiction]`;
- store external fields as nullable;
- never fill unavailable truth with predictions;
- make every join traceable.

---

## 4. Model output schema

One row per item × model × prediction variant:

```text
object_id
model_id
model_revision
training_data
inference_method
prompt_id
seed
sample_count
calibration_id
logit_entailment
logit_neutral
logit_contradiction
p_entailment
p_neutral
p_contradiction
predicted_label
entropy_bits
inference_timestamp
software_environment_hash
```

Variants:

- official supplied logits;
- locally reproduced classifier logits;
- uncalibrated probabilities;
- temperature-scaled probabilities;
- MC-dropout means;
- deep-ensemble means;
- LLM Monte Carlo label distributions;
- LLM log-probability distributions.

Never overwrite raw output with calibrated output.

### 4.1 Label-map contract

```yaml
model_id: example-model
source_labels:
  0: contradiction
  1: neutral
  2: entailment
canonical_order:
  - entailment
  - neutral
  - contradiction
```

A unit test should send known synthetic logits through every adapter.

---

## 5. Study manifest

```yaml
study_id: chaosnli-human-opinion-topology-v1
source_manifest: manifests/sources.lock.yaml
datasets:
  - chaosnli_snli
  - chaosnli_mnli
exclude:
  - alpha_nli
human_prior:
  family: dirichlet
  alpha: [0.5, 0.5, 0.5]
posterior_draws: 2000
models:
  registry: configs/models.lock.yaml
metrics:
  pointwise_primary: jensen_shannon
  neighborhood_primary: hellinger
  sensitivity:
    - jensen_shannon
    - total_variation
    - euclidean
    - aitchison_delta_1e-6
k_values: [5, 10, 20, 50]
primary_k: 10
split_half:
  repetitions: 1000
  labels_per_half: 50
random_seed: 20260801
confirmatory_split_manifest: manifests/items.lock.parquet
software_lock: manifests/environment.lock
```

Copy the frozen manifest into every release.

---

## 6. Pipeline DAG

```text
fetch
  ↓
verify-sources
  ↓
normalize
  ↓
audit-joins
  ↓
human-posterior
  ↓
model-predict ──→ calibrate
  ↓                  ↓
model-tables ←───────┘
  ↓
build-spaces
  ↓
compute-neighbors
  ↓
compare-graphs
  ↓
run-statistics
  ↓
select-review-cases
  ↓
render-review-packets
  ↓
import-codings
  ↓
adjudicate
  ↓
build-shadowspace-bundle
  ↓
build-report
  ↓
verify-release
```

A Makefile, `doit`, Snakemake, or Python task graph is sufficient. Every task must be deterministic, idempotent, hash-cacheable, independently testable, and command-line runnable.

---

## 7. Proposed CLI

```bash
shadowspace chaosnli fetch --manifest configs/study.yaml
shadowspace chaosnli verify-sources --manifest configs/study.yaml
shadowspace chaosnli normalize --manifest configs/study.yaml
shadowspace chaosnli audit-joins --manifest configs/study.yaml
shadowspace chaosnli human-posterior --manifest configs/study.yaml
shadowspace chaosnli predict --model roberta-large --manifest configs/study.yaml
shadowspace chaosnli calibrate --model roberta-large --manifest configs/study.yaml
shadowspace chaosnli build-spaces --manifest configs/study.yaml
shadowspace chaosnli compute-neighbors --manifest configs/study.yaml
shadowspace chaosnli compare-graphs --manifest configs/study.yaml
shadowspace chaosnli analyze --plan configs/analysis.lock.yaml
shadowspace chaosnli select-cases --packet configs/review_packets.yaml
shadowspace chaosnli render-packets --packet configs/review_packets.yaml
shadowspace chaosnli import-codings coding/round1.csv
shadowspace chaosnli build-bundle --manifest configs/study.yaml
shadowspace chaosnli report --manifest configs/study.yaml
shadowspace chaosnli verify-release artifacts/releases/v1
```

Each command emits JSON status:

```text
task
started_at
finished_at
input_hashes
output_hashes
row_counts
warnings
errors
software_version
git_commit
```

---

## 8. Modules

```text
src/shadowspace_chaosnli/
├── acquisition.py
├── schema.py
├── normalize.py
├── joins.py
├── posterior.py
├── model_registry.py
├── inference.py
├── calibration.py
├── distances.py
├── neighbors.py
├── graph_metrics.py
├── mismatch.py
├── case_selection.py
├── packets.py
├── coding.py
├── statistics.py
├── bundle.py
├── report.py
└── cli.py
```

### Acquisition

- download/locate sources;
- compute SHA-256;
- verify counts;
- record license;
- fail on unexpected changes.

### Normalization

- parse JSONL;
- map labels;
- preserve text;
- validate counts;
- recompute distributions/entropy;
- compare supplied fields;
- create canonical Parquet.

### Posterior

- Dirichlet parameters;
- deterministic draws;
- entropy intervals;
- majority probabilities;
- edge support;
- split-half reliability.

Do not store all draws by default. Store parameters, seed, summaries, and optional compressed frozen draws.

### Inference

- tokenize exact premise/hypothesis;
- preserve exact model input;
- map labels;
- store logits;
- pin revision;
- batch/resume;
- no hidden fallbacks.

### Calibration

Temperature scaling must use a separate calibration set. Never fit on the locked ChaosNLI confirmatory set.

Store:

```text
calibration_dataset
calibration_object_hash
temperature
objective
optimizer
seed
before_score
after_score
```

### Distance and neighbors

Exact pairwise computation is feasible at \(N=3113\).

```text
distances/
  human__hellinger.f32.npy
  human__jsd.f32.npy
  model-id__hellinger.f32.npy

neighbors/
  human__hellinger__k005.parquet
  human__hellinger__k010.parquet
```

Neighbor table:

```text
source_id
neighbor_id
rank
distance
space_id
metric_id
k
```

Use stable object-ID tie breaking.

### Graph comparison

Outputs:

- local overlap/Jaccard;
- human-only/model-only edges;
- \(Q_{NX}\);
- LCMC;
- multi-scale curves;
- posterior edge support;
- cross-model consensus;
- split-half distributions.

### Statistics

Consume a locked plan and write:

- tidy result table;
- estimates;
- intervals;
- corrected p-values where applicable;
- exclusions;
- failed assumptions;
- robustness;
- machine-readable claim status.

No notebook should be the sole source of a published number.

---

## 9. Automated case selection

### 9.1 Case types

```text
human_only_neighbor
model_only_neighbor
model_consensus_human_unsupported
human_supported_most_models_miss
majority_correct_shape_wrong
uncertainty_collapse
spurious_uncertainty
unsupported_label_mass
calibration_improves_jsd_not_topology
geometry_sensitive
zero_policy_sensitive
posterior_uncertain
taxonomy_homogeneous_neighborhood
taxonomy_mixed_neighborhood
random_matched_control
```

### 9.2 Selection record

```text
case_id
packet_type
focal_object_id
comparison_object_ids
selection_score
selection_rule_version
stratum
random_seed
matched_control_id
hidden_selection_reason
```

Hide selection reasons from blind coders.

### 9.3 Matched controls

Match on combinations of:

- source;
- majority label;
- entropy bin;
- premise/hypothesis length;
- genre;
- model majority correctness.

This reduces superficial selection leakage.

---

## 10. Review packet generation

Generate HTML and PNG. HTML is primary for selectable/accessibly sized text.

### Packet layout

**A — Source**

- premise/hypothesis;
- dataset/genre;
- source ID.

**B — Human judgments**

- count bars;
- empirical distribution;
- posterior ternary region;
- entropy interval;
- majority probability.

**C — Models**

- distributions;
- optionally blinded identity;
- raw/calibrated;
- residual arrows.

**D — Neighborhoods**

- top human/model neighbors;
- shared, human-only, model-only;
- edge support;
- source texts.

**E — Sensitivity**

- metric agreement;
- \(k\)-curve;
- zero-policy sensitivity;
- projection integrity.

**F — Coding**

- disagreement source;
- valid variation/error/unresolved;
- confidence;
- rationale;
- request-more-context.

### Per-item images

```text
ternary_human_posterior.png
ternary_human_models.png
distribution_bars.png
neighbor_overlap.png
edge_support_heatmap.png
metric_sensitivity.png
model_residuals.png
```

### Dataset-level images

```text
human_simplex_density.png
entropy_distribution.png
model_human_jsd_by_entropy.png
qnx_curves.png
split_half_ceiling.png
graph_overlap_by_model.png
taxonomy_homophily.png
consensus_vs_human_support.png
mismatch_type_counts.png
```

Every figure includes data version, metric, \(k\), model revision, calibration, and release/state ID.

### Accessibility

- no color-only encoding;
- direct labels/line styles;
- alt text;
- tabular equivalent;
- 200% zoom;
- reduced motion;
- readable text.

---

## 11. Feedback loop

```text
automated candidate generation
        ↓
blind coding round 1
        ↓
disagreement report
        ↓
adjudication
        ↓
coding import
        ↓
updated analysis
        ↓
new candidate queue
```

Keep coding-development, reliability, and locked evaluation sets separate. Do not train an automatic coder on adjudicated cases and evaluate on those same cases.

---

## 12. Shadowspace bundle design

### Source objects

`objects.parquet` contains canonical schema and text payload references.

### Representations

```text
human_probability
model__<model-id>__uncalibrated
model__<model-id>__temperature
human_sqrt
human_clr__delta-1e-6
text_embedding__<encoder-id>
model_residual__<model-id>
```

Do not apply probability metrics to text embeddings.

### Metrics

```text
jensen_shannon
hellinger
fisher_rao
total_variation
euclidean
aitchison__delta-1e-6
cosine__text-embedding
```

### Views

- exact human ternary;
- exact model ternary;
- dual human/model ternary;
- residual arrows;
- taxonomy color;
- posterior edge support.

### Diagnostics

```text
human_neighbor_support
model_human_local_overlap
model_human_qnx
model_consensus
metric_sensitivity
posterior_uncertainty
taxonomy_homophily
projection_integrity
```

Every diagnostic identifies its source space.

---

## 13. Tests

### Data

- exact row counts;
- unique IDs;
- counts sum 100;
- probabilities sum one;
- entropy recomputation;
- label order;
- text round-trip;
- join audit.

### Math

- distance symmetry/bounds;
- Hellinger/Fisher–Rao rank equivalence;
- ternary Euclidean proportionality;
- posterior reproducibility;
- split-half reproducibility;
- edge support in \([0,1]\);
- \(Q_{NX}(N-1)=1\);
- LCMC chance baseline under permutation.

### Model

- label-map fixture;
- deterministic logits;
- calibration-data declaration;
- finite probabilities;
- probabilities sum one;
- pinned revisions.

### Packet

- source text resolves;
- all figures share state hash;
- selection reason absent from blind packet;
- tables match plots;
- alt text present.

### Release

```bash
make chaosnli-release STUDY=configs/study.lock.yaml
make chaosnli-verify RELEASE=artifacts/releases/<release-id>
```

The verifier checks hashes and every primary estimate.

---

## 14. Storage

One \(3113\times3113\) float32 matrix is about 39 MB. Multiple models/metrics remain manageable.

Policy:

- exact chunked distances;
- persist primary matrices;
- top-\(k\) edges for secondary metrics;
- memory mapping;
- Parquet for tidy outputs;
- no large JSON numeric arrays.
