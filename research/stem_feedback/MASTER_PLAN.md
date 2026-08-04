# Disagreement-Aware STEM Feedback
## Master Research, Product, and Technical Plan

**Status:** Internal master planning document  
**Audience:** Project owner & research team  
**Primary purpose:** Organize the research program and guide implementation  
**Core thesis:** **AI feedback should respect human disagreement.**

---

# Table of Contents

1. [Vision](#1-vision)
2. [Current Research Foundation](#2-current-research-foundation)
3. [Research-to-Product Program](#3-research-to-product-program)
4. [Development Roadmap](#4-development-roadmap)
5. [Critical Open Questions](#5-critical-open-questions)
6. [Strategic Principles](#6-strategic-principles)
7. [Product Requirements](#7-product-requirements)
8. [Teacher Workflow](#8-teacher-workflow)
9. [Attention Signals](#9-attention-signals)
10. [Product Screens](#10-product-screens)
11. [Success Metrics](#11-success-metrics)
12. [Technical Architecture](#12-technical-architecture)
13. [Core Data Model](#13-core-data-model)
14. [Analysis Pipeline](#14-analysis-pipeline)
15. [Application Structure](#15-application-structure)
16. [Privacy and Governance](#16-privacy-and-governance)
17. [Baselines](#17-baselines)
18. [Implementation Milestones](#18-implementation-milestones)
19. [Definition of V1 Complete](#19-definition-of-v1-complete)
20. [Immediate Next Steps](#20-immediate-next-steps)

---

# 1. Vision

The long-term goal is to build an educational feedback system that does not treat grading as a simple prediction problem.

Instead of asking only:

> What score should this response receive?

the system should also ask:

> Where is human judgment likely to matter, why might reasonable graders disagree, and which student responses deserve closer teacher attention?

The product should help teachers allocate their time—not replace their judgment.

The eventual system should provide:

1. A class-level map of reasoning patterns.
2. A prioritized queue of responses requiring teacher review.
3. Clear explanations of why each response was flagged.
4. Evidence about whether the issue is likely to involve:
   - a common misconception;
   - borderline rubric performance;
   - unusual but potentially valid reasoning;
   - inconsistent evidence;
   - rubric ambiguity;
   - model uncertainty or disagreement.

The first version will focus on **teacher attention routing**, not automatic grading or autonomous feedback.

---

# 2. Current Research Foundation

The existing ChaosNLI research provides the methodological foundation.

## 2.1 Working findings

1. **Human disagreement has relational structure.**  
   Examples can be organized according to similarities in collective human judgment distributions.

2. **Individual models recover only part of that structure.**  
   Models recognize broad properties such as the majority label, uncertainty magnitude, and the leading type of ambiguity.

3. **Calibration is not relational alignment.**  
   Probability calibration can improve likelihood-based scores while doing very little to make model relationships more human-like.

4. **Model ensembles recover more structure.**  
   Different model families appear to contribute partially nonredundant relational information.

5. **Human judgment geometry is compressible.**  
   A relatively small set of prototype judgment states can reproduce substantial portions of the human relational target.

6. **Model quality may be interpreted as effective resolution.**  
   Individual models and ensembles can be compared with compressed human representations containing different numbers of prototype states.

These findings suggest that educational assessment systems should be evaluated not only by whether their scores match teachers, but also by whether they preserve the structure of teacher judgments.

---

# 3. Research-to-Product Program

## 3.1 Paper 1: Measurement foundation

**Working title:**  
*Calibration Is Not Relational Alignment: Human Disagreement Geometry in Natural Language Inference*

**Purpose:** Establish the core measurement framework.

**Primary components:**

- E001: Baseline relational alignment.
- E002: Calibration versus relational alignment.
- E003: Post-hoc transformations and model ensembles.
- E004: Modern generative LLM validation, if results are sufficiently stable.

**Main contribution:**  
Models can improve pointwise probability fit without improving the relational organization of human disagreement.

---

## 3.2 Paper 2: Resolution and compression

**Possible title:**  
*At What Resolution Do Models Represent Human Disagreement?*

**Primary components:**

- E005: Conditional disagreement hierarchy.
- E006: Human annotation-budget requirements.
- E007: Ensemble census and relational Shapley attribution.
- E008: Human geometry compression and effective prototype complexity.
- E009: Temperature-topology phase diagrams.

**Main contribution:**  
Models and ensembles can be placed on a common scale describing the resolution at which they represent human disagreement.

---

## 3.3 Paper 3: Educational application

**Possible title:**  
*Disagreement-Aware Teacher Attention Routing for STEM Reasoning Assessment*

**Primary question:**  
Can disagreement-aware relational methods identify student responses where teacher attention is most valuable?

**Initial scope:**

- Generic STEM reasoning.
- One or more open datasets.
- Local-first inference.
- Teacher review queue plus class-level pattern summary.
- No autonomous grading or feedback delivery.

---

# 4. Development Roadmap

## Phase 0: Freeze the core research

Complete and document:

- E004 Stage 1B.
- E005 full-data run.
- E006 annotation-budget study.
- E007 full coalition census.
- E008 full-data compression curve.
- Uncertainty intervals and robustness tests.
- Clear separation between confirmed and exploratory claims.

**Exit gate:**  
The relational measurement framework is stable, reproducible, and documented.

---

## Phase 1: Synthetic educational wind tunnel

Create synthetic STEM student responses with known structures:

- clear correct reasoning;
- clear misconceptions;
- partially correct reasoning;
- multiple valid strategies;
- rubric-borderline responses;
- intentionally ambiguous rubric cases;
- outlier reasoning;
- inconsistent final answers and work.

Use synthetic data to test:

- schemas;
- ingestion;
- graph construction;
- review ranking;
- visualization;
- failure handling;
- reproducibility.

**Claim boundary:**  
Synthetic data validates system mechanics, not real teacher behavior.

**Exit gate:**  
The system reliably recovers the structures intentionally placed in the synthetic data.

---

## Phase 2: Public STEM pilot

Select one or more public datasets containing:

- STEM prompts or problems;
- student responses or solution traces;
- scores, partial credit, feedback, or multiple annotations;
- sufficient licensing for research use.

The first public pilot does not need to be exclusively Algebra 2. Generic STEM reasoning is acceptable.

**Exit gate:**  
The system produces interpretable attention flags on real educational responses and outperforms simple baselines such as model confidence or score margin.

---

## Phase 3: Small teacher-rated pilot

Construct a deliberately small but high-quality dataset.

For each student response, collect:

- rubric-based scores from multiple teachers;
- confidence or uncertainty;
- short rationale;
- whether another interpretation is reasonable;
- whether the rubric is clear;
- recommended review priority.

Three raters per response may be enough for an initial operational pilot, but more may be required for stable disagreement distributions.

**Exit gate:**  
The model identifies high-attention cases with acceptable recall and produces explanations teachers consider useful.

---

## Phase 4: Workflow evaluation

Test the tool as a teacher-facing workflow.

Compare:

- ordinary grading;
- grading with a random review queue;
- grading with confidence-based triage;
- grading with disagreement-aware relational triage.

Measure:

- review time;
- number of responses manually inspected;
- missed high-attention cases;
- changes in scoring consistency;
- teacher trust;
- rubric revisions prompted by the tool;
- quality of teacher feedback.

**Exit gate:**  
The system reduces workload or improves consistency without concealing consequential cases.

---

## Phase 5: Productization

Develop a deployable, privacy-conscious application supporting:

- assignment and rubric upload;
- response ingestion;
- local inference;
- class-level pattern summaries;
- teacher review queues;
- teacher decisions and audit logs;
- configurable thresholds;
- export of reviewed results.

---

# 5. Critical Open Questions

## 5.1 Scientific questions

1. Does relational structure add predictive value beyond entropy, confidence, and score margin?
2. How many human raters are required for stable educational disagreement geometry?
3. Which disagreements represent valid variation rather than grading error?
4. Can the system distinguish unusual valid reasoning from misconceptions?
5. Do model ensembles improve teacher-attention routing?
6. Does calibration improve scores while reducing disagreement resolution?
7. How well do findings generalize across STEM subjects and grade levels?
8. Do score-distribution relationships align with relationships among teacher explanations?
9. Can reasoning-based graphs predict where rubrics are underspecified?
10. How much student data is necessary before useful class-level patterns emerge?

## 5.2 Product questions

1. What should the initial attention categories be?
2. Should the tool rank individual responses, class patterns, or both?
3. How many responses can a teacher reasonably review?
4. What explanation is sufficient for a teacher to trust a flag?
5. How should teachers dismiss, confirm, or relabel flags?
6. How should rubric ambiguity be presented?
7. What is an acceptable false-negative rate?
8. Should teachers control the size of the review queue?
9. How should the system handle assignments with very few responses?
10. What should happen when the local model fails or produces invalid output?

## 5.3 Governance questions

1. Which data can remain entirely local?
2. What student information must be removed before analysis?
3. What permissions are required for classroom data?
4. How long should data and model outputs be retained?
5. How will performance be checked across student groups?
6. What decisions must always remain teacher-controlled?
7. What audit information must be preserved?
8. What data may be used for research publication?
9. How will teachers or institutions delete their data?
10. What human-subjects or institutional review is required?

---

# 6. Strategic Principles

1. **Human judgment remains authoritative.**
2. **Uncertainty should route attention, not automatically determine outcomes.**
3. **Every flag should have an understandable reason.**
4. **Synthetic data is for engineering validation.**
5. **Public data is for initial external validation.**
6. **Real teacher-rated data is required for educational claims.**
7. **The project should measure missed important cases, not only average accuracy.**
8. **Research claims and product capabilities must remain clearly separated.**
9. **The system should expose uncertainty rather than hide it.**
10. **The product should reduce unnecessary review without suppressing unusual valid reasoning.**

---

# 7. Product Requirements

## 7.1 Product problem

Teachers often spend similar amounts of time reviewing responses that differ greatly in how much judgment they require.

Some responses are:

- clearly correct;
- clearly incorrect;
- common examples of a familiar misconception;
- borderline under the rubric;
- unusual but potentially valid;
- internally inconsistent;
- difficult for automated systems to interpret;
- evidence that the rubric itself is unclear.

Conventional AI graders usually return a score, confidence value, or feedback draft. They do not reliably show how a response relates to patterns of human judgment across the rest of the class.

The proposed system will use disagreement-aware and relational methods to prioritize teacher review.

## 7.2 V1 product promise

> Show the teacher where closer human attention is likely to matter, and explain why.

The system will not promise perfect identification of every pedagogically important response. It will surface only categories that can be detected with acceptable reliability.

## 7.3 V1 non-goals

The first version will not:

1. Automatically submit grades.
2. Send feedback to students without teacher approval.
3. Make final high-stakes decisions.
4. Claim that model uncertainty is equivalent to teacher disagreement.
5. Infer personal characteristics from student work.
6. Replace professional teacher judgment.
7. Automatically retrain on teacher decisions.
8. Recommend disciplinary or placement decisions.

---

# 8. Teacher Workflow

## Step 1: Create an assignment

The teacher supplies:

- assignment title;
- problem statement;
- expected learning objective;
- rubric;
- optional reference solution;
- optional scored anchor responses.

Anchor responses should ideally represent a range:

- weak;
- borderline;
- adequate;
- strong;
- unusual but valid.

## Step 2: Import student responses

Responses may be entered manually or uploaded from a structured file.

The system should support:

- short constructed responses;
- multi-step reasoning;
- equations and plain-text mathematical work;
- final answers;
- optional teacher annotations.

## Step 3: Run local analysis

The system processes each response using:

- deterministic rule-based features;
- local Ollama model assessments;
- optional multiple prompts or models;
- uncertainty and disagreement measures;
- relational comparisons among responses.

## Step 4: View the class summary

The summary should show:

- major reasoning patterns;
- common misconceptions;
- clusters of similar responses;
- distribution of likely rubric performance;
- responses unlike the rest of the class;
- possible rubric ambiguity;
- number of responses requiring closer review.

This screen should answer:

> What happened across the class?

## Step 5: Review the attention queue

Each queued response should contain:

- student identifier or anonymous code;
- response text;
- tentative rubric evidence;
- attention priority;
- reason for the flag;
- similar responses;
- model disagreement or uncertainty;
- recommended teacher action.

Possible recommendations:

- review score boundary;
- verify unusual reasoning;
- inspect possible misconception;
- compare with similar responses;
- reconsider rubric wording;
- no immediate action.

This screen should answer:

> Where should I spend my time?

## Step 6: Record teacher decisions

The teacher can:

- confirm the flag;
- dismiss the flag;
- change its category;
- assign or revise a score;
- add notes;
- mark a reasoning pattern as valid;
- mark a rubric issue;
- identify a new misconception.

These decisions become evaluation data. They should not automatically retrain the model without an explicit research or update process.

---

# 9. Attention Signals

V1 should begin with measurable, conservative signals.

## 9.1 High-priority signals

1. **Model disagreement**  
   Multiple model conditions produce materially different evaluations.

2. **Borderline rubric evidence**  
   The response lies near a rubric threshold.

3. **Unusual reasoning**  
   The response differs substantially from common class patterns.

4. **Inconsistency**  
   The work and final answer do not support one another.

5. **Cluster-level misconception**  
   Multiple students show a similar incorrect reasoning pattern.

6. **Potentially valid alternative**  
   Reasoning differs from the reference solution but may remain defensible.

7. **Rubric ambiguity**  
   Similar responses appear to support different rubric outcomes.

8. **Relational instability**  
   The response's nearest comparison cases change substantially across prompts, models, or calibration conditions.

## 9.2 Signals requiring later validation

- fairness risk;
- copied or coordinated work;
- student effort;
- motivation;
- conceptual understanding beyond the submitted response;
- teacher-quality judgments.

---

# 10. Product Screens

## 10.1 Dashboard

Displays:

- assignment status;
- number of responses;
- analysis completion;
- review-queue size;
- major class patterns;
- unresolved teacher decisions.

## 10.2 Class Pattern Map

Displays:

- response clusters;
- representative examples;
- cluster size;
- estimated rubric distribution;
- average uncertainty;
- common reasoning features.

## 10.3 Review Queue

Displays:

- ranked responses;
- priority score;
- attention reason;
- brief evidence;
- teacher action controls.

## 10.4 Response Detail

Displays:

- complete student response;
- assignment and rubric;
- model assessments;
- neighboring responses;
- cluster membership;
- teacher notes and decisions;
- audit history.

## 10.5 Rubric Diagnostics

Displays:

- rubric criteria associated with disagreement;
- thresholds producing inconsistent classifications;
- examples near each boundary;
- suggested areas for teacher review—not automatic rubric rewriting.

---

# 11. Success Metrics

## 11.1 Technical metrics

- successful ingestion rate;
- valid local-inference rate;
- reproducibility;
- processing time;
- missing-data rate;
- failure recovery.

## 11.2 Attention-routing metrics

- recall of teacher-identified high-attention responses;
- precision of the review queue;
- false-negative rate;
- workload reduction;
- review-queue size;
- ranking quality;
- performance relative to confidence-only baselines.

## 11.3 Educational usefulness

- teacher agreement with flag reason;
- teacher-reported usefulness;
- number of newly discovered misconception patterns;
- number of valid alternative solutions surfaced;
- rubric issues identified;
- changes in teacher review time.

## 11.4 Trust and safety

- rate of unsupported explanations;
- differences in flag rates across relevant student groups;
- frequency of teacher overrides;
- severity of missed cases;
- teacher understanding of system limitations.

## 11.5 Pilot go/no-go criteria

Proceed to a teacher pilot only when:

- response ingestion is reliable;
- the review queue is reproducible;
- every flag has traceable evidence;
- simple baselines are implemented;
- synthetic recovery tests pass;
- the system does not silently omit responses;
- local data handling is documented.

Proceed beyond the pilot only when:

- high-attention recall is acceptable;
- false negatives are reviewed;
- teachers find the queue useful;
- teacher time is reduced or better allocated;
- no major subgroup failure is detected;
- the system remains advisory.

---

# 12. Technical Architecture

## 12.1 Architecture overview

The application should separate four concerns:

1. **Web application**  
   Flask routes, authentication, forms, dashboards, and review workflow.

2. **Persistent data layer**  
   SQLite and SQLAlchemy models.

3. **Inference layer**  
   Local Ollama requests, structured-output validation, prompt provenance, and resumable execution.

4. **Relational analytics layer**  
   Rust binaries or a shared Rust library for distances, graphs, nulls, clustering, ensemble analysis, and attention ranking.

The first integration between Flask and Rust should be simple:

- Flask writes an immutable run input file.
- Flask starts a Rust CLI subprocess.
- Rust writes versioned JSON or Parquet artifacts.
- Flask validates and imports the results.

Avoid building a networked microservice architecture in V1.

---

# 13. Core Data Model

## Assignment

- `id`
- `title`
- `description`
- `subject`
- `grade_band`
- `learning_objective`
- `created_at`

## Problem

- `id`
- `assignment_id`
- `prompt`
- `reference_solution`
- `response_format`

## Rubric

- `id`
- `assignment_id`
- `name`
- `version`
- `maximum_score`
- `status`

## RubricCriterion

- `id`
- `rubric_id`
- `name`
- `description`
- `score_levels`
- `weight`
- `display_order`

## StudentResponse

- `id`
- `problem_id`
- `anonymous_student_code`
- `response_text`
- `final_answer`
- `source`
- `submitted_at`
- `content_hash`

## AnchorResponse

- `id`
- `problem_id`
- `response_text`
- `score`
- `rationale`
- `anchor_type`
- `content_hash`

## HumanJudgment

- `id`
- `student_response_id`
- `rater_code`
- `rubric_scores`
- `overall_score`
- `confidence`
- `attention_priority`
- `reasoning_category`
- `rationale`
- `created_at`

## ModelRun

- `id`
- `assignment_id`
- `model_name`
- `model_digest`
- `prompt_version`
- `generation_parameters`
- `code_commit`
- `object_order_hash`
- `status`
- `started_at`
- `completed_at`

## ModelAssessment

- `id`
- `model_run_id`
- `student_response_id`
- `criterion_scores`
- `score_distribution`
- `confidence`
- `reasoning_features`
- `raw_response_path`
- `valid`

## AttentionFlag

- `id`
- `student_response_id`
- `model_run_id`
- `priority`
- `flag_type`
- `evidence`
- `cluster_id`
- `status`

## TeacherReview

- `id`
- `attention_flag_id`
- `teacher_action`
- `confirmed`
- `corrected_category`
- `final_score`
- `notes`
- `reviewed_at`

---

# 14. Analysis Pipeline

## Stage A: Ingestion

1. Validate assignment and rubric.
2. Import responses.
3. Remove direct identifiers.
4. Generate immutable response IDs.
5. Hash all input content.
6. Reject malformed or duplicate records.

## Stage B: Deterministic features

Extract:

- response length;
- equation count;
- numeric-answer presence;
- final-answer consistency;
- rubric-keyword evidence;
- lexical overlap with anchors;
- duplicate or near-duplicate responses.

These provide transparent baselines.

## Stage C: Local model assessments

Use Ollama to request structured assessments.

Each request should include:

- assignment prompt;
- rubric;
- response;
- optional anchor examples;
- explicit output schema;
- prompt version;
- fixed generation parameters;
- model digest.

Store raw responses immutably.

Model outputs may include:

- evidence for each rubric criterion;
- provisional score distribution;
- uncertainty;
- possible misconception;
- alternative reasoning;
- response validity;
- request for teacher review.

Model output must not be treated as ground truth.

## Stage D: Disagreement representation

For each response, construct one or more distributions:

- model score distribution;
- criterion-level distributions;
- ensemble distribution;
- repeated-sampling distribution;
- human-rating distribution when available.

Representations may be compared using:

- Hellinger distance;
- Jensen–Shannon divergence;
- score-margin difference;
- rubric-evidence similarity;
- reasoning-feature similarity.

## Stage E: Relational graph

Build a tie-aware nearest-neighbor graph over responses.

Possible graph targets:

- similarity in teacher score distributions;
- similarity in attention-priority judgments;
- similarity in misconception labels;
- similarity in reasoning explanations;
- combined multi-view similarity.

Calculate:

- neighbor stability;
- model-human relational support;
- local uncertainty;
- cluster persistence;
- outlier status;
- ensemble contribution.

## Stage F: Attention ranking

The initial attention score should combine distinct signals rather than hide them in one opaque number.

$$A_i = w_1 D_i + w_2 B_i + w_3 O_i + w_4 I_i + w_5 R_i$$

where:

- $D_i$: model disagreement;
- $B_i$: rubric-boundary proximity;
- $O_i$: response outlier score;
- $I_i$: reasoning/final-answer inconsistency;
- $R_i$: relational mismatch or instability.

The interface should show each component separately.

Weights should initially be manually configured and later learned only from teacher-reviewed pilot data.

---

# 15. Application Structure

## 15.1 Flask application

```text
app/
├── __init__.py
├── config.py
├── extensions.py
├── models/
│   ├── assignment.py
│   ├── rubric.py
│   ├── response.py
│   ├── judgment.py
│   ├── model_run.py
│   └── review.py
├── routes/
│   ├── assignments.py
│   ├── responses.py
│   ├── analysis.py
│   ├── dashboard.py
│   └── reviews.py
├── services/
│   ├── ingestion.py
│   ├── ollama_client.py
│   ├── run_manager.py
│   ├── artifact_validator.py
│   └── rust_runner.py
├── templates/
│   ├── dashboard.html
│   ├── assignment.html
│   ├── class_patterns.html
│   ├── review_queue.html
│   └── response_detail.html
├── static/
│   ├── css/
│   └── js/
└── schemas/
    ├── imports.py
    └── model_outputs.py
```

## 15.2 Rust analytics

```text
research_engine/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── distance.rs
    ├── topk.rs
    ├── support.rs
    ├── nulls.rs
    ├── clustering.rs
    ├── ensembles.rs
    ├── attention.rs
    ├── provenance.rs
    └── bin/
        ├── build_graph.rs
        ├── analyze_disagreement.rs
        ├── cluster_responses.rs
        └── rank_attention.rs
```

Every Rust output should include:

- input hashes;
- object-order hash;
- code commit;
- metric definitions;
- random seeds;
- thread count;
- model provenance;
- completion status.

## 15.3 Local Ollama requirements

Record for every run:

- Ollama version;
- model name;
- model digest;
- quantization;
- context length;
- prompt version;
- system prompt;
- sampling parameters;
- seed;
- parallelism;
- hardware information.

Use:

- resumable request IDs;
- append-only raw response files;
- bounded concurrency;
- explicit timeouts;
- validation and retry limits;
- no silent substitution for failed requests.

---

# 16. Privacy and Governance

V1 should assume educational data is sensitive.

Requirements:

- local storage by default;
- anonymous student codes;
- no unnecessary names or identifiers;
- encrypted backups where feasible;
- clear data-deletion process;
- access logging;
- immutable research artifacts separated from user-facing data;
- no external LLM API calls without explicit approval;
- no automatic student communication.

Any use of actual course data should receive appropriate institutional, district, contractual, and ethical review before research publication.

---

# 17. Baselines

The research prototype must compare disagreement-aware routing against simpler alternatives:

1. Random review queue.
2. Lowest model confidence.
3. Smallest score margin.
4. Distance from rubric threshold.
5. Response-length or outlier heuristic.
6. Single-model uncertainty.
7. Multi-model disagreement.
8. Full relational attention score.

The relational method is valuable only if it improves teacher-attention routing beyond these baselines.

---

# 18. Implementation Milestones

## Milestone 1: Project skeleton

- Flask application;
- SQLAlchemy schema;
- assignment and response ingestion;
- basic dashboard;
- test fixtures.

## Milestone 2: Synthetic dataset generator

- configurable STEM prompts;
- known misconception types;
- controlled ambiguity;
- known rubric boundaries;
- expected attention labels.

## Milestone 3: Local inference

- Ollama client;
- structured response schema;
- resumable runs;
- provenance capture;
- raw output viewer.

## Milestone 4: Rust analytics

- distance calculations;
- response graph;
- clustering;
- attention signals;
- JSON result import.

## Milestone 5: Teacher interface

- class summary;
- review queue;
- response-detail screen;
- teacher decision logging.

## Milestone 6: Public-data pilot

- dataset adapter;
- baseline evaluation;
- error analysis;
- internal pilot report.

## Milestone 7: Multi-rater study

- rater interface;
- judgment collection;
- disagreement graph;
- human-attention target;
- comparative evaluation.

---

# 19. Definition of V1 Complete

V1 is complete when a teacher or researcher can:

1. Create an assignment and rubric.
2. Import a class of STEM responses.
3. Run all analysis locally.
4. View class-level response patterns.
5. Open a prioritized review queue.
6. Understand why each response was flagged.
7. Record teacher decisions.
8. Export reviewed results and run provenance.
9. Reproduce the analysis from frozen artifacts.

V1 is not complete merely because it can generate scores.

---

# 20. Immediate Next Steps

1. Finish and freeze the current disagreement-geometry experiments.
2. Select candidate public STEM datasets.
3. Define the synthetic STEM response generator.
4. Establish the educational data schema.
5. Build the local Flask project skeleton.
6. Implement assignment, rubric, and response ingestion.
7. Build the Ollama structured-assessment runner.
8. Implement class summaries and a review queue.
9. Test simple attention-routing baselines.
10. Design the first multi-rater teacher study.
11. Establish privacy, consent, and data-retention requirements.
12. Keep the first product focused on teacher attention routing rather than automated grading.
