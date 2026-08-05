"""Substantive 2-Evaluator Agreement & Consensus (Excludes Deterministic Negative Control)."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def classify_practical_zero(delta: float, eps: float = 0.01) -> int:
    """Classifies counterfactual difference into Z_je in {-1, 0, +1} under epsilon."""
    if delta < -eps:
        return -1
    elif delta > eps:
        return 1
    return 0


def compute_substantive_evaluator_consensus(
    evaluator_deltas_dict: Dict[str, np.ndarray],
    eps: float = 0.01,
) -> Dict[str, Any]:
    """Calculates 2-evaluator substantive consensus excluding deterministic exact lexicon control."""
    substantive_ids = [eid for eid in evaluator_deltas_dict.keys() if eid != "exact_lexicon"]
    E = len(substantive_ids)
    if E == 0:
        return {}

    N = len(evaluator_deltas_dict[substantive_ids[0]])
    stabilities = []
    exact_agree_count = 0
    opposite_count = 0

    for j in range(N):
        z_vals = [classify_practical_zero(evaluator_deltas_dict[eid][j], eps=eps) for eid in substantive_ids]
        counts = {-1: z_vals.count(-1), 0: z_vals.count(0), 1: z_vals.count(1)}
        max_count = max(counts.values())
        stability_j = max_count / float(E)
        stabilities.append(stability_j)

        if max_count == E:
            exact_agree_count += 1
        if counts[-1] > 0 and counts[1] > 0:
            opposite_count += 1

    mean_stability = float(np.mean(stabilities))

    return {
        "epsilon_threshold": eps,
        "substantive_evaluators": substantive_ids,
        "substantive_evaluators_count": E,
        "negative_control_excluded": True,
        "sample_size_N": N,
        "mean_substantive_consensus_stability": round(mean_stability, 4),
        "exact_category_agreement_rate": round(exact_agree_count / N, 4),
        "opposite_direction_disagreement_rate": round(opposite_count / N, 4),
    }
