# ChaosNLI Coding and Review Protocol

## 1. Purpose

Human review is used to:

- interpret automatically detected geometric/model mismatches;
- validate whether neighborhood relations share plausible disagreement causes;
- distinguish valid variation from likely annotation error;
- evaluate generated explanations;
- create a scoring rubric for a later Shadowspace user study.

Human review must not become an informal search process where the same person selects, interprets, and reports interesting examples without blinding.

---

## 2. External taxonomies

### 2.1 Disagreement sources

**Uncertainty in sentence meaning**

- Lexical
- Implicature
- Presupposition
- Probabilistic enrichment
- Imperfection

**Underspecification in guidelines**

- Coreference
- Temporal reference
- Interrogative hypothesis

**Annotator behavior**

- Accommodating minimally added content
- High overlap

Treat these as multi-label; more than one source may apply.

### 2.2 Variation versus error

VariErr distinguishes:

- valid human variation;
- annotation error;
- mixed;
- unresolved.

Do not force every dissenting label to be valid or erroneous.

### 2.3 Explanation reasoning

LiTEx and related work show:

- identical labels can have different reasoning;
- different labels can have similar reasoning;
- explanation strategy and label choice may vary by annotator.

Record both label interpretation and reasoning phenomenon.

---

## 3. Roles

Recommended:

- **Coder A:** primary linguistic review;
- **Coder B:** independent review;
- **Adjudicator:** resolves disagreements without changing original records;
- **Pipeline operator:** prepares blinded packets and cannot alter them after coding begins.

During pilots the owner may hold multiple roles. Confirmatory coding should separate preparation from adjudication where possible.

---

## 4. Coding sets

### Development

Refine instructions, find missing categories, test packet usability. No reliability claims.

### Reliability

Estimate agreement, freeze definitions, identify confusion. Independently double-coded.

### Locked evaluation

Test hypotheses and produce final interpretation. No instruction changes after start. A required change creates a new version and recoding.

---

## 5. Sampling

Suggested first reliability round:

```text
25 human-only neighborhood cases
25 model-only neighborhood cases
25 model-consensus/human-unsupported cases
25 uncertainty-collapse cases
25 geometry-sensitive cases
25 random matched controls
Total: 150
```

Stratify across dataset, majority label, entropy, model, and length/genre where possible. Hide strata from coders.

---

## 6. Blinding stages

### Round 1 — Source-only

Show:

- premise;
- hypothesis;
- label definitions;
- optional genre.

Hide:

- human distribution;
- models;
- selection reason;
- model identity;
- neighbors unless pair coding is required.

Code:

- plausible label set;
- disagreement source(s);
- valid variation/error/unresolved;
- confidence;
- rationale.

### Round 2 — Human evidence

Reveal:

- 100-vote counts;
- posterior uncertainty;
- human neighbors.

Code:

- whether judgment changes;
- whether disagreement appears systematic;
- whether neighbors share a reason.

### Round 3 — Model evidence

Reveal:

- blinded model distributions;
- human/model neighbor differences;
- metric sensitivity.

Code:

- mismatch type;
- whether model behavior is defensible;
- human-supported/model-specific/unresolved.

### Round 4 — Debrief

Reveal selection reason, model identity, and external annotations. This is qualitative and does not replace blinded records.

---

## 7. Item coding form

```text
case_id
coder_id
protocol_version
round

plausible_labels:
  entailment: yes/no/uncertain
  neutral: yes/no/uncertain
  contradiction: yes/no/uncertain

primary_label_if_forced:
disagreement_present: yes/no/uncertain

disagreement_sources:
  lexical
  implicature
  presupposition
  probabilistic_enrichment
  imperfection
  coreference
  temporal_reference
  interrogative_hypothesis
  accommodating_minimal_content
  high_overlap
  other
  insufficient_context

variation_validity:
  valid_variation
  likely_annotation_error
  mixed
  unresolved

confidence: 0-100
rationale:
additional_context_needed:
```

Do not force one taxonomy label.

---

## 8. Pair/neighborhood form

For focal item \(A\), neighbor \(B\):

```text
case_id
focal_id
neighbor_id

same_majority_label:
similar_label_distribution:
shared_disagreement_source:
shared_reasoning_pattern:
semantically_related_text:
neighbor_relation_meaningful:

relation_type:
  human_supported
  model_supported
  both
  neither
  unresolved

confidence:
rationale:
```

Do not ask whether items are simply “similar” without specifying the sense.

---

## 9. Adjudication

Preserve:

- Coder A;
- Coder B;
- adjudicated record;
- adjudicator rationale;
- guideline version.

Options:

- A accepted;
- B accepted;
- merged multi-label;
- new `other`;
- unresolved.

Do not erase expert disagreement; it is evidence about taxonomy interpretability.

---

## 10. Reliability reporting

### Categorical fields

Report:

- raw agreement;
- prevalence;
- category-wise precision/recall;
- Krippendorff’s alpha or appropriate chance-adjusted coefficient;
- bootstrap interval.

Rare categories require category-level reporting; an aggregate can hide failure.

### Label sets

- exact-set agreement;
- mean Jaccard;
- per-label agreement.

### Confidence

- Spearman correlation;
- mean absolute difference;
- calibration against adjudicated correctness where available.

### Rationales

Do not use automatic semantic similarity as the sole quality measure. Manually audit samples.

---

## 11. LLM-assisted review rules

LLMs may:

- format packets;
- propose categories after human coding;
- check missing fields;
- summarize rationales;
- retrieve prior coded cases;
- prepare an adjudication disagreement report.

LLMs are not gold annotators.

For LLM assistance:

- freeze model and prompt;
- store every input/output;
- omit selection reason;
- evaluate on reliability set;
- report abstentions;
- prevent leakage;
- retain human override and rationale.

Existing evidence shows strong models remain below humans in separating valid variation from error; automatic labels remain provisional.

---

## 12. Packet quality checklist

- [ ] Complete premise/hypothesis.
- [ ] No model identity leak in blind rounds.
- [ ] Selection reason hidden.
- [ ] Label order declared.
- [ ] Human counts match canonical table.
- [ ] Posterior region uses locked prior.
- [ ] Model probabilities match frozen logits.
- [ ] Neighbors use displayed metric and \(k\).
- [ ] Every plot uses same state hash.
- [ ] Text readable without separate file.
- [ ] Non-color encoding.
- [ ] `Unresolved` available.
- [ ] `More context needed` available.
- [ ] Save action auditable.
- [ ] Packet version recorded.

---

## 13. Later user-study rubric

Full credit requires both relation status and limitation.

Example:

> The model and humans agree on the majority label, but the model collapses a human neutral/entailment split. The focal human-neighbor edge has high posterior support and is stable under Hellinger and JSD. External coding is consistent with implicature, but the visualization alone cannot establish that cause.

Partial credit:

- mismatch identified but uncertainty omitted;
- projection sensitivity identified but linguistic cause overclaimed.

No credit:

- visible proximity as sole evidence;
- confidence equated with human agreement;
- taxonomy prediction treated as verified;
- unresolved warning ignored.

---

## 14. Automated feedback outputs

```text
coder_disagreement_matrix.csv
category_prevalence.csv
reliability_report.md
adjudication_queue.html
guideline_confusion_cases.html
unresolved_cases.html
selection_bias_audit.md
```

The selection-bias audit compares coded cases with the full dataset on:

- source;
- majority;
- entropy;
- length;
- genre;
- model accuracy;
- model–human JSD.

Final reporting must distinguish claims from:

- existing external annotations;
- new independent coding;
- adjudication;
- automated suggestions.
