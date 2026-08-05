# Project Handoff & Continuity Guide — Educational Counterfactual AI Audit

**Date**: 2026-08-05  
**Current Phase**: Phase EDU-2a-R1.2c (Engineering Frozen, Human Review Active)  
**Branch**: `research/education-counterfactual-audit`  
**Branch Head Commit**: [`3064ab0`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy)  
**Code Commit SHA**: [`be34446`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy)  
**Execution Status**: `COMPLETED` | `completion_integrity_status: PASSED` | `finding: AWAITING_MANUAL_REVIEW`  

---

## 1. Executive Summary & Project State

The live model generation pipeline (`gemma3:12b` via Ollama) has successfully generated all **60 complete counterfactual recommendation letters** with **100% complete generation integrity** (zero truncations, `done_reason: "stop"`, `num_predict: 650`). All prompt templates use native neutral placeholder framing (`[CANDIDATE]`, `they/them/their`), completely eliminating ungrammatical post-hoc prose redactions.

Engineering on the generation pipeline, blinding engine, rating packet generator, validator, and submission compiler is **100% complete and frozen**. The project is currently awaiting manual human ratings.

---

## 2. Key File Directory & Artifact Sitemap

| Resource / Artifact | Path | Description |
| :--- | :--- | :--- |
| **Human Review Rubric** | [`docs/education_audit/HUMAN_REVIEW_RUBRIC.md`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/docs/education_audit/HUMAN_REVIEW_RUBRIC.md) | Official 1-5 scale anchors & claim adjudication definitions. |
| **Decision Log** | [`docs/education_audit/DECISION_LOG.md`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/docs/education_audit/DECISION_LOG.md) | Chronological scientific decision log. |
| **Live Generations** | [`results/education_audit/edu_2a/generations.jsonl`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/results/education_audit/edu_2a/generations.jsonl) | 60 frozen live outputs from `gemma3:12b` (digest `f4031aab`). |
| **Analysis Manifest** | [`results/education_audit/edu_2a/analysis_manifest.json`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/results/education_audit/edu_2a/analysis_manifest.json) | Complete provenance & artifact-derived status manifest. |
| **Public Review Summary** | [`results/education_audit/edu_2a/review_design_summary.json`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/results/education_audit/edu_2a/review_design_summary.json) | Public summary of review quotas & packet hashes. |
| **Public Audit Report** | [`results/education_audit/edu_2a/report.md`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/results/education_audit/edu_2a/report.md) | Markdown audit report aligned 1-to-1 with status labels. |
| **Private Blinding Key** | `private_review/edu_2a_r1_blinding_key.json` | Secret key mapping letter IDs to conditions (uncommitted). |
| **Private Design Manifest** | `private_review/edu_2a_r1_review_design_manifest.json` | Secret design manifest with packet SHA256 hashes & R2 IDs. |
| **Practice Calibration Packet** | [`private_review/working/practice_calibration_5letters.csv`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/private_review/working/practice_calibration_5letters.csv) | 5 nonstudy practice letters for rater calibration. |
| **Reviewer 1 Working Packet** | [`private_review/working/reviewer1_pass1_working.csv`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/private_review/working/reviewer1_pass1_working.csv) | Reviewer 1 Pass 1 working copy (65 letters). |
| **Reviewer 2 Working Packet** | [`private_review/working/reviewer2_pass1_working.csv`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/private_review/working/reviewer2_pass1_working.csv) | Reviewer 2 Pass 1 working copy (20 shared letters). |

---

## 3. Step-by-Step Guide to Resuming Work

When jumping back into the project to conduct human review or process ratings, follow this exact sequence:

### Step 1: Rater Calibration (5 Nonstudy Practice Letters)
1. Have Reviewer 1 and Reviewer 2 independently complete [`private_review/working/practice_calibration_5letters.csv`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/private_review/working/practice_calibration_5letters.csv).
2. Compare scores, discuss discrepancies, and align on score anchors in [`HUMAN_REVIEW_RUBRIC.md`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/docs/education_audit/HUMAN_REVIEW_RUBRIC.md).
3. Do not include these 5 practice letters in official study results.

### Step 2: Pass 1 Shared 20-Letter Assessment
1. Reviewer 1 rates Pass 1 in [`private_review/working/reviewer1_pass1_working.csv`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/private_review/working/reviewer1_pass1_working.csv).
2. Reviewer 2 rates Pass 1 in [`private_review/working/reviewer2_pass1_working.csv`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/private_review/working/reviewer2_pass1_working.csv).
3. Calculate agreement on the 20 shared letters (Within-1-point $\ge 90\%$, MAD $\le 0.50$).

### Step 3: Complete & Lock Pass 1
1. If agreement passes, Reviewer 1 completes the remaining 45 Pass 1 letters.
2. Save immutable locked copies of Pass 1 working files under `private_review/locked/`.

### Step 4: Release & Complete Pass 2 (Factual Grounding Adjudication)
1. Release Pass 2 working files (`edu_2a_r1_reviewer1_pass2.csv` and `edu_2a_r1_reviewer2_pass2.csv`) showing target opportunity and verified facts.
2. Reviewers complete factual fidelity scores, unsupported positive/negative claim counts, and omission counts.

### Step 5: Submission Compilation & Final Analysis Gate
Once all 4 filled working files exist ($65 + 65 + 20 + 20 = 170$ records), run:
```powershell
$env:PYTHONPATH="."
python -c "from research.education_audit.evaluation.compile_review_submissions import compile_review_submissions; compile_review_submissions(valid_letter_ids=[f'LTR_R1_{i:03d}' for i in range(1, 66)])"
python research/education_audit/analysis/edu_2a_analysis.py
pytest --no-cov tests/education_audit
```

---

## 4. What NOT to Do

* **Do NOT rerun Gemma** or regenerate the 60 letters (generation is complete and 100% un-truncated).
* **Do NOT modify frozen private packet CSV files** in `private_review/` (edits must happen in `private_review/working/` to preserve SHA256 hashes).
* **Do NOT create new engineering phases** (all validators, compilers, and gates are complete and tested).
* **Do NOT expand to 240 letters** until human review validation passes.
