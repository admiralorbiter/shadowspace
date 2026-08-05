# Educational Counterfactual AI Audit - Decision Log

**Project Branch**: `research/education-counterfactual-audit`  
**Latest Synced Results Commit**: `186a4d2`  

---

## Chronological Decision History

### Decision 001: Pivot from NLI Holonomy to Applied Educational Audit (Phase EDU-0)
* **Date**: 2026-08-04
* **Context**: Phase E2-A1.2a-R1.2 established heavy-tailed pointwise sensitivity in pretrained NLI models, but global affine transport added no held-out predictive skill over identity ($R^2 \approx 0.992$) and did not reject the commuting null ($p = 0.9950$).
* **Decision**: Formally close NLI holonomy phase at R1.2 (`docs/holonomy/PHASE_E2_A1_2A_R1_2_CLOSEOUT.md`) and pivot to applied educational counterfactual AI auditing.
* **Target Task**: Recommendation letter generation (held-out student qualifications, counterfactual identity cues).

---

### Decision 002: Measurement Hardening & CI Consolidation (Phase EDU-1.1a)
* **Date**: 2026-08-04
* **Decisions**:
  1. **Consolidated CI**: Integrated education test suite into `.github/workflows/pytest_holonomy.yml` with `workflow_dispatch:`, `pytest-cov`, and `pip install -e .`.
  2. **Cryptographic Stable Seeds**: Replaced Python string `hash()` with SHA-256 `stable_seed(*parts)` for cross-process reproducibility.
  3. **Cue-Dose Balance**: Isolated identity headers (`Candidate Name:`, `Preferred Pronouns:`) and rendered `Verified Accomplishments:` bullet points raw without prepending student names.
  4. **Strict Zero-Disparity Null Contract**: Required zero disparity across strength, hallucinations, endorsements, word count, and lexical counts under `IndependentNullAdapter` (`INDEPENDENT_NULL_CONTRACT_PASSED`).
  5. **Evidence Level Tag**: Labeled automated rule-based rubric as `SCREENING_ONLY`.

---

### Decision 003: Successful Live Canary Execution (`gemma3:12b` via Ollama) (Phase EDU-2a)
* **Date**: 2026-08-04
* **Live Model**: `gemma3:12b` (digest `f4031aab637d1ffa37b42570452ae0e4fad0314754d17ded67322e4b95836f8a`, `Q4_K_M`, Ollama `0.32.5`, 100% GPU VRAM).
* **Canary Size**: 5 Preflight letters + 60 Live Canary letters ($2 \text{ profiles} \times 5 \text{ conditions} \times 2 \text{ prompts} \times 3 \text{ seeds}$).
* **Execution Performance**: ~12–15 seconds per letter at `num_predict = 250`. Total execution time ~10.5 minutes.
* **Results**:
  - `execution_status`: `COMPLETED`
  - `generation_integrity_status`: `PASSED` (0 fallback, 0 truncation, 0 errors)
  - `identity_leakage_count`: `0` (100% clean redaction to `[CANDIDATE]` and `they/them/their`)
  - `paired_contrasts`: `pronoun_masc_minus_fem` = `0.000`, `name_masc_minus_fem` = `-0.167` (minimal prompt) / `0.000` (structured prompt).
  - `go_to_full_pilot`: `true`

---

## Current Status & Next Steps for Tomorrow

1. **Human Manual Review**: The blinded rating packet is exported at [`results/education_audit/edu_2a/blinded_rating_packet.csv`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/results/education_audit/edu_2a/blinded_rating_packet.csv) and [`blinded_rating_packet.jsonl`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/results/education_audit/edu_2a/blinded_rating_packet.jsonl) (secret key in [`blinding_key.json`](file:///C:/Users/admir/Github/shadowspace-ambiguity-holonomy/results/education_audit/edu_2a/blinding_key.json)).
2. **Phase EDU-2 Pilot Expansion**: Once human ratings confirm rater reliability, expand from 2 profiles (60 letters) to all 8 profiles (240 letters).
