# Educational Counterfactual AI Audit - Project Charter

**Version**: 1.0.0  
**Status**: APPROVED  
**Target Scope**: Recommendation-letter generation counterfactual audit harness  

---

## 1. Project Mission & Purpose

The Educational Counterfactual AI Audit project develops an offline, reproducible research and audit harness designed to evaluate whether AI systems produce biased, unequal, or hallucinated outputs when identity cues (e.g. names, pronouns) change while student qualifications, achievements, and context remain 100% constant.

---

## 2. Core Operating Principles

1. **Synthetic Initial Data Only**: No real student records or PII will be submitted to external APIs during research pilots.
2. **Offline Audit Harness**: The tool operates strictly as an offline evaluator and regression test, NOT an autonomous grader or decision-maker.
3. **Strict Provenance & Reproducibility**: All runs require pinned model/tokenizer revisions, exact prompt/parameter hashes, and strict two-commit code/results manifest tracking.
4. **Blinded Evaluation**: Outcome raters evaluate identity-redacted outputs without seeing identity conditions or model metadata.
5. **No Unsupported Claims**: Simple lexical differences do not constitute proof of discrimination without human validation and pre-registered practical effect size thresholds.
