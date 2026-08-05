"""Independent Evaluator Agreement and Practical-Zero Classification Stability."""

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


def compute_evaluator_consensus_stability(
    evaluator_deltas_dict: Dict[str, np.ndarray],
    eps: float = 0.01,
) -> Dict[str, Any]:
    """Calculates all-evaluator agreement, majority agreement, and average stability across independent evaluators."""
    eval_ids = list(evaluator_deltas_dict.keys())
    E = len(eval_ids)
    if E == 0:
        return {}

    N = len(evaluator_deltas_dict[eval_ids[0]])
    stabilities = []
    all_agree_count = 0
    maj_agree_count = 0
    opposite_count = 0

    for j in range(N):
        z_vals = [classify_practical_zero(evaluator_deltas_dict[eid][j], eps=eps) for eid in eval_ids]

        # Count frequencies of -1, 0, +1
        counts = { -1: z_vals.count(-1), 0: z_vals.count(0), 1: z_vals.count(1) }
        max_count = max(counts.values())
        stability_j = max_count / float(E)
        stabilities.append(stability_j)

        if max_count == E:
            all_agree_count += 1
        if max_count >= (E // 2 + 1):
            maj_agree_count += 1
        if counts[-1] > 0 and counts[1] > 0:
            opposite_count += 1

    mean_stability = float(np.mean(stabilities))

    return {
        "epsilon_threshold": eps,
        "evaluators_count": E,
        "sample_size_N": N,
        "mean_consensus_stability": round(mean_stability, 4),
        "all_evaluator_agreement_rate": round(all_agree_count / N, 4),
        "majority_agreement_rate": round(maj_agree_count / N, 4),
        "opposite_direction_disagreement_rate": round(opposite_count / N, 4),
    }
