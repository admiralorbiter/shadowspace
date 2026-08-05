# Educational Counterfactual AI Audit Roadmap

**Status**: Active Applied Research Roadmap  
**Primary Goal**: Build an offline, reproducible audit harness that tests whether AI-generated educational and workforce materials change when identity cues change but qualifications, work, and accomplishments remain fixed.  
**Operating Principle**: Fail fast, preserve provenance, separate descriptive findings from supported conclusions, and do not use real student data during initial research phases.

---

## 1. Executive Summary & Product Scope

Holding student work, accomplishments, qualifications, and context constant, do identity cues alter the feedback, recommendation strength, opportunities, scores, or professional materials produced by an AI system?

The harness operates as an **offline evaluator and regression test**, NOT an autonomous decision-maker or production grader.

---

## 2. Application Sequence

| Order | Application | Scope & Purpose | Status |
| :---: | :--- | :--- | :---: |
| **1** | **Student Recommendation Letters** | Test for gender/identity bias in letter strength, lexical tone, and hallucinated claims. | **Build First (EDU-1)** |
| **2** | **Student Feedback Generation** | Test for differences in feedback rigor, encouragement, and opportunity recommendations. | **Build Second** |
| **3** | **Cover-Letter Generation** | Test workforce internship/scholarship application material bias. | **Add After Feedback** |
| **4** | **Rubric-Based Grading** | Requires human gold scores and psychometric calibration. | **Deferred** |
| **5** | **Applicant Ranking or Selection** | High-risk decision support requiring governance/legal review. | **Deferred Longest** |

---

## 3. Pilot 1 Specification: Recommendation-Letter Audit (EDU-1)

### Base Student Profiles:
8 synthetic student profiles across 4 domains (Technology, Math/Data, Humanities, Community Leadership) $\times$ 2 achievement levels (Qualified, Exceptional).

### 5 Identity Conditions per Profile:
1. **Anonymous Baseline**: Student A (they/them)
2. **Masculine Pronoun Cue**: Student A (he/him)
3. **Feminine Pronoun Cue**: Student A (she/her)
4. **Masculine Name Cue**: Historically male-associated name
5. **Feminine Name Cue**: Historically female-associated name

### Primary Outcome Metrics:
1. **Blinded Recommendation Strength**: 1–5 scale rated by blinded human or structured evaluator.
2. **Factual Fidelity**: Hallucinated claims per 100 words (unsupported claims beyond fact sheet).
3. **Opportunity-Strength Language**: Explicit recommendation rate for top-tier opportunities.

---

## 4. Technical & Governance Architecture

- **Fail-Fast Gates**:
  - Gate 0 (Validity): Qualifications remain 100% byte-identical.
  - Gate 1 (Provenance): Pinned model/tokenizer revisions, zero silent fallback.
  - Gate 2 (Reliability): Evaluator inter-rater reliability $\kappa \ge 0.70$.
  - Gate 3 (Signal vs Noise): Predeclared smallest effect size of interest ($\ge 0.25$ points on 5-point scale).
  - Gate 4 (Replication): Cross-model replication required before general claims.
  - Gate 5 (Privacy): 100% synthetic data; zero real student records submitted to external APIs.

---

## 5. Development Phases

- **Phase EDU-0**: NLI Milestone Freeze & Applied Roadmap Creation (Completed).
- **Phase EDU-1**: Mock Vertical Slice & Planted-Bias Suite (In Progress).
- **Phase EDU-2**: Single-Model Live Recommendation-Letter Audit (240 letters).
- **Phase EDU-3**: Multi-Model Confirmatory Study.
