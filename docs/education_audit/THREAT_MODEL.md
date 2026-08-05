# Threat Model & Risk Analysis - Educational Counterfactual AI Audit

---

## 1. Research & Evaluation Threats

| Threat ID | Threat Description | Severity | Mitigation Strategy |
| :--- | :--- | :---: | :--- |
| **T-01** | **Name-Proxy Ambiguity**: Using proper names as sole demographic proxies conflates name frequency/tokenization with identity bias. | HIGH | Separate explicit pronoun cues (`he/him`, `she/her`) from name-only cues and anonymous baselines (`they/them`). |
| **T-02** | **Prompt Sensitivity Overinterpretation**: Findings driven by specific prompt phrasing rather than systematic model behavior. | HIGH | Test multiple distinct prompt templates (minimal vs structured); evaluate prompt interactions. |
| **T-03** | **Evaluator Model Bias**: Using an LLM judge that introduces its own demographic or stylistic preferences. | CRITICAL | Use blinded human review as primary gold standard; LLM judges are secondary screening tools only. |
| **T-04** | **Stochastic Generation Noise**: Ordinary LLM generation variance misidentified as demographic bias. | HIGH | Require 3+ repeated generations per condition; cluster analysis by base profile. |

---

## 2. Data & Governance Threats

| Threat ID | Threat Description | Severity | Mitigation Strategy |
| :--- | :--- | :---: | :--- |
| **T-05** | **FERPA / PII Leakage**: Real student information submitted to external API endpoints. | CRITICAL | Enforce 100% synthetic profile requirement; Gate 5 verification. |
| **T-06** | **Scope Creep into Autonomous Decisions**: Audit prototype mistakenly deployed to make real student admission or grading decisions. | CRITICAL | Scope restricted by Project Charter to offline regression testing; output labeled as audit evidence. |
