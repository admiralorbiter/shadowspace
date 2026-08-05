"""Substantive 2-Evaluator Agreement & 3x3 Category Cross-Tabulation."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def classify_practical_zero(delta: float, eps: float = 0.01) -> str:
    """Classifies counterfactual difference into category 'Negative', 'Zero', or 'Positive' under epsilon."""
    if delta < -eps:
        return "Negative"
    elif delta > eps:
        return "Positive"
    return "Zero"


def compute_substantive_evaluator_consensus(
    evaluator_deltas_dict: Dict[str, np.ndarray],
    eps: float = 0.01,
) -> Dict[str, Any]:
    """Calculates 2-evaluator substantive consensus and exports 3x3 category cross-tabulation table."""
    substantive_ids = [eid for eid in evaluator_deltas_dict.keys() if eid != "exact_lexicon"]
    if len(substantive_ids) < 2:
        return {}

    id1, id2 = substantive_ids[0], substantive_ids[1]
    deltas1 = evaluator_deltas_dict[id1]
    deltas2 = evaluator_deltas_dict[id2]

    N = len(deltas1)
    cats = ["Negative", "Zero", "Positive"]
    cross_tab: Dict[str, Dict[str, int]] = {c1: {c2: 0 for c2 in cats} for c1 in cats}

    exact_agree = 0
    opposite_disagree = 0
    nonzero_eval_count = 0
    nonzero_agree_count = 0

    for d1, d2 in zip(deltas1, deltas2):
        c1 = classify_practical_zero(d1, eps=eps)
        c2 = classify_practical_zero(d2, eps=eps)
        cross_tab[c1][c2] += 1

        if c1 == c2:
            exact_agree += 1
        if (c1 == "Negative" and c2 == "Positive") or (c1 == "Positive" and c2 == "Negative"):
            opposite_disagree += 1

        if c1 != "Zero" or c2 != "Zero":
            nonzero_eval_count += 1
            if c1 == c2:
                nonzero_agree_count += 1

    cond_nonzero_rate = float(nonzero_agree_count / max(1, nonzero_eval_count))

    return {
        "epsilon_threshold": eps,
        "substantive_evaluators": [id1, id2],
        "sample_size_N": N,
        "category_cross_tabulation_3x3": cross_tab,
        "exact_category_agreement_rate": round(exact_agree / N, 4),
        "opposite_direction_disagreement_rate": round(opposite_disagree / N, 4),
        "conditional_nonzero_agreement_rate": round(cond_nonzero_rate, 4),
        "nonzero_evaluations_count": nonzero_eval_count,
    }
