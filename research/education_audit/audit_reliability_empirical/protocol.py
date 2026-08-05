"""ER-0: Machine-Readable Protocol & Preregistered Analysis Decisions."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


PREREGISTERED_PROTOCOL = {
    "study_title": "Empirical Audit Reliability & Counterfactual Meta-Evaluation Framework",
    "primary_research_question": "Under what conditions is a counterfactual bias conclusion reliable across evaluators, contexts, prompts, profiles, and sampling seeds?",
    "primary_epsilon": 0.01,
    "sensitivity_epsilons": [0.00, 0.01, 0.02, 0.05],
    "equivalence_bound_delta": 0.02,
    "primary_outcomes": [
        "evaluator_masd",
        "evaluator_cfr",
        "independent_evaluator_sign_agreement",
        "tail_cvar_95",
        "generalizability_coefficient_g",
    ],
    "secondary_outcomes": [
        "mean_signed_drift_msd",
        "max_absolute_drift",
        "factual_coverage_delta",
        "unsupported_claim_delta",
        "attribution_category_deltas",
    ],
    "claim_gates_strict": True,
    "human_data_lock": {
        "automated_metrics_hidden_until_ratings_locked": True,
    },
}


def get_preregistered_protocol() -> Dict[str, Any]:
    """Returns the frozen machine-readable protocol dict."""
    return dict(PREREGISTERED_PROTOCOL)


def validate_claim_gate(
    claim_type: str,
    is_independent: bool,
    sample_size: int,
    min_required_sample: int = 100,
) -> bool:
    """Validates whether a candidate claim passes Claim Gate requirements to be labeled empirical."""
    if not is_independent:
        return False
    if sample_size < min_required_sample:
        return False
    return True
