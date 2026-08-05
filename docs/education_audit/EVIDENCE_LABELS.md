# Evidence Status Taxonomy - Educational Counterfactual AI Audit

All exported metrics, manifests, and reports must tag results using the following standardized evidence status labels:

---

## Status Taxonomy

| Status Label | Definition | Usage Context |
| :--- | :--- | :--- |
| **`COMPLETED`** | The execution pipeline ran to completion with valid provenance and zero errors. | Execution status |
| **`OBSERVED`** | Direct counterfactual differences (e.g. word count, sentiment, score shifts) were measured. | Sensitivity status |
| **`PLANTED_SIGNAL_RECOVERED`** | A known planted bias in a mock/synthetic test suite was accurately detected by the evaluator. | Mock validation status |
| **`PLANTED_NULL_VERIFIED`** | A known planted null condition produced no false positive flag in the audit harness. | Mock validation status |
| **`DESCRIPTIVELY_SMALL`** | Observed differences are smaller than pre-registered practical significance thresholds. | Metric comparison |
| **`DESCRIPTIVE_ONLY`** | Reported metric describes sample properties but has not undergone a formal hypothesis test. | Metric comparison |
| **`NOT_REJECTED`** | A formal hypothesis test (bootstrap or permutation) did not reject the null hypothesis ($p > 0.05$). | Statistical hypothesis test |
| **`REJECTED`** | A formal hypothesis test rejected the null hypothesis ($p \le 0.05$). | Statistical hypothesis test |
