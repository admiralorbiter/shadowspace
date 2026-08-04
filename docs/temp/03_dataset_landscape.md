# Dataset Landscape and Cross-Dataset Triangulation Plan

## Executive answer

The existing data are sufficient for a substantial research program without collecting a new dataset immediately.

They are sufficient to test:

- whether disagreement geometry is compressible across domains;
- whether models under-resolve human judgment structure;
- whether calibration changes relational resolution;
- whether ensembles recover complementary structure;
- whether demographic or cultural groups occupy different judgment geometries where group metadata exist;
- whether public datasets exhibit shared disagreement archetypes.

They are not sufficient to build or validate a full individual or collective digital twin with causal or longitudinal claims.

The correct strategy is **cross-dataset triangulation**, not naive dataset merging.

## 1. ChaosNLI

### What it provides

- 100 annotations per item;
- 3,113 SNLI/MNLI items in the present project;
- dense empirical label distributions;
- enough votes for posterior and annotation-budget analysis;
- a clean three-label simplex;
- strong classifier and local-LLM baselines.

### What it is ideal for

- item-distribution geometry;
- posterior-supported neighborhood graphs;
- relational recovery;
- prototype compression;
- annotation-budget saturation;
- calibration stress tests;
- classifier/LLM comparisons.

### What it does not provide

- repeated annotator identities suitable for individual geometry;
- demographic group labels;
- rationales for individual choices;
- longitudinal observations;
- reliable distinction between uncertainty and stable camps.

### Claim language

Use:

- statistical minority interpretation;
- low-frequency label mass;
- ambiguity type;
- collective vote distribution.

Do not use:

- demographic minority erasure;
- stable ideological camps;
- individual preference twin.

## 2. OpinionQA

### What it provides

OpinionQA is based on public-opinion survey questions and includes individualized human responses and demographic group summaries. The associated research evaluates alignment with 60 US demographic groups.

### Questions it enables

- group-specific probability geometry;
- cross-group overlap;
- which groups are poorly represented by a model;
- whether steering changes relational rather than only marginal alignment;
- demographic missing-region analysis;
- issue-dependent coalition geometry.

### Limitations

- US public opinion;
- survey option structure;
- demographic categories can be broad and correlated;
- questions may not form one coherent latent domain;
- normative care is required to avoid stereotyping.

## 3. PRISM

### What it provides

PRISM connects:

- 1,500 participants;
- 75 countries;
- participant characteristics and stated preferences;
- 8,011 live conversations;
- 21 LLMs;
- fine-grained ratings and contextual feedback.

### Questions it enables

- individual and demographic heterogeneity;
- contextual preference geometry;
- cross-cultural overlap;
- model-specific missing regions;
- whether user profiles predict stable relational neighborhoods;
- whether alignment methods average away contextual structure.

### Limitations

- sparse repeated observations relative to a purpose-built panel;
- conversational contexts differ;
- ratings may not share a simple common label simplex;
- missingness and selection effects require careful modeling.

## 4. Demographic rationale datasets

“Being Right for Whose Right Reasons?” augments rationale annotations with demographic information across sentiment and commonsense tasks.

### Questions it enables

- whether groups agree on labels but disagree on rationales;
- label-geometry versus rationale-geometry comparison;
- whose explanations a model reproduces;
- whether relational alignment is driven by output labels or reasoning patterns.

### Limitation

The size and task scope are smaller than a dedicated collective-twin panel.

## 5. CrowdTruth resources

CrowdTruth datasets and metrics model ambiguity jointly across:

- input units;
- workers;
- annotations.

### Questions enabled

- whether item ambiguity predicts relational instability;
- whether worker quality and item ambiguity can be separated;
- comparison of CrowdTruth ambiguity metrics with graph-based support.

## 6. Debate and argument datasets

Examples include:

- ArguAna;
- Kialo-derived structures;
- PERSPECTRA;
- pro/con argument graphs.

### Questions enabled

- viewpoint counting;
- position coverage;
- argument-neighborhood geometry;
- whether a model confuses distinct positions;
- Overton coverage versus distributional alignment.

### Limitations

- arguments are not population samples;
- frequency in a debate corpus is not population prevalence;
- synthetic expansion can distort natural viewpoint distributions.

## 7. Pluralistic safety and value datasets

Current resources include:

- PRISM;
- OpinionQA;
- DICES-derived safety data;
- Demo-SafetyBench;
- pluralistic-value alignment collections;
- Overton pluralism benchmarks.

### Questions enabled

- safety-boundary geometry;
- demographic variation in refusal preferences;
- representation of reasonable alternatives;
- whether one scalar reward collapses group-specific structure.

### High-stakes boundary

These datasets support evaluation research. They do not justify autonomous safety or policy decisions for real communities.

## 8. Cross-dataset questions answerable now

### 8.1 Universal compressibility

> Does each domain possess a smooth rate–distortion curve, and how many prototype-equivalent states are needed?

Compare:

- NLI;
- public opinion;
- safety preference;
- conversational feedback;
- rationales;
- debate perspectives.

### 8.2 Model under-resolution

> Do open models systematically occupy a lower effective-resolution regime than human data across domains?

### 8.3 Calibration versus pluralism

> Does optimizing a pointwise score improve or reduce relational resolution across tasks?

### 8.4 Ensemble complementarity

> Does combining model families add prototype-equivalent resolution consistently?

### 8.5 Group overlap

> Which parts of the judgment geometry are shared across groups, and which are group-specific?

### 8.6 Missing-region analysis

> Are some human-supported regions consistently absent from model outputs?

### 8.7 Geometry transfer

> Are there recurring disagreement archetypes across unrelated domains?

### 8.8 Data requirements

> How many annotations or participants are needed to recover pointwise, prototype, and relational structure?

## 9. What cannot be answered without new data

### 9.1 True individual digital twins

Requirements:

- repeated people;
- many common items;
- enough observations per person;
- stable identity linkage;
- consent and privacy protections.

### 9.2 Causal opinion dynamics

Requirements:

- repeated time points;
- controlled interventions or credible natural experiments;
- exposure information;
- attrition modeling.

### 9.3 Why people disagree

Requirements:

- rationales;
- interpretation tags;
- interviews;
- pairwise “same reason?” judgments;
- possibly qualitative coding.

### 9.4 Community representation claims

Requirements:

- defensible sampling;
- sufficient subgroup counts;
- measurement-equivalence checks;
- transparent category definitions.

## 10. Do not concatenate datasets

A raw union would confuse:

- labels;
- tasks;
- populations;
- sampling frames;
- annotation processes;
- model prompts;
- response scales.

Instead build a common analysis contract.

## 11. Common data contract

Each dataset adapter should produce:

```text
dataset_id
item_id
item_text_or_reference
response_space_id
response_options
human_counts_or_ratings
annotator_id_optional
group_attributes_optional
rationale_optional
timestamp_optional
sampling_weight_optional
source_split
license
provenance_hash
```

Derived artifacts:

```text
empirical_distribution
posterior_specification
fold_id
stratum_id
distance_metric
human_support_graph
prototype_curve
group_graphs_optional
model_probability_artifacts
```

## 12. Harmonization levels

### Level 1 — Within-dataset replication

Use each dataset’s native response space and report comparable normalized quantities.

### Level 2 — Meta-analysis

Combine effect sizes:

- calibration–relational gap;
- effective-bit deficit;
- group-overlap deficit;
- ensemble gain.

### Level 3 — Structural comparison

Use distance-matrix or graph-level comparison, including GW/FGW, without pretending labels are identical.

### Level 4 — Shared archetype model

Fit cross-domain meta-prototypes only after demonstrating stable within-domain structure.

## 13. Local-model strategy

Paid frontier APIs are not required to produce a high-impact result.

A compelling open-model panel could include:

- Gemma family;
- Llama family;
- Qwen family;
- Mistral family;
- smaller distilled or reasoning variants.

Freeze:

- exact model digest;
- quantization;
- KV-cache precision;
- prompt;
- candidate labels;
- sampling method;
- random seeds;
- runtime version.

The scientific contribution is the evaluation primitive, not access to the most expensive endpoint.

## 14. Recommended public-data program

### Study A — Cross-domain compression

Datasets:

- ChaosNLI;
- OpinionQA;
- PRISM subset with common rating structure;
- one rationale or safety dataset.

Outputs:

- rate–distortion curves;
- effective bits;
- sample-size sensitivity.

### Study B — Collective overlap

Datasets:

- OpinionQA demographic groups;
- PRISM country or preference groups.

Outputs:

- marginal overlap;
- graph cross-support;
- group-specific prototypes;
- missing-region recall;
- overlap uncertainty.

### Study C — Model pluralism stress test

Models:

- several local open families.

Outputs:

- distributional fit;
- Overton coverage where applicable;
- relational recovery;
- effective bits;
- calibration effects;
- ensemble complementarity.

## 15. New-data collection trigger

Collect a custom panel only after public-data triangulation establishes at least two of:

1. stable group-specific geometries;
2. systematic missing regions;
3. cross-domain prototype recurrence;
4. strong value from repeated identities;
5. evidence that rationales explain residual geometry.

Then design the new dataset to answer a precise missing question rather than collecting “more disagreement data” generically.
