"""ER-0: Data Schemas & Protocol Validators."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class CounterfactualPairRecord(TypedDict):
    pair_id: str
    base_sentence_id: str
    category: str  # "pronoun" or "name"
    text_masc: str
    text_fem: str
    sub_masc: Dict[str, str]
    sub_fem: Dict[str, str]
    ground_truth_agency_label: int


class EvaluatorReliabilityCardRecord(TypedDict):
    evaluator_id: str
    evaluator_name: str
    is_independent: bool
    total_comparisons: int
    msd_mean_signed_drift: float
    masd_mean_absolute_score_difference: float
    cfr_counterfactual_flip_rate: float
    max_absolute_drift: float
    cvar_95_tail_risk: float
    tost_equivalence_passed: bool
