"""Two One-Sided Equivalence Testing (TOST) on Per-Sentence Aggregate Cluster Means."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from scipy import stats


def run_tost_equivalence_test(
    pairs_data: List[Dict[str, Any]],
    raw_deltas: np.ndarray,
    evaluator_type: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Conducts TOST across independent per-sentence aggregate cluster means N_clusters."""
    if not pairs_data or len(raw_deltas) == 0:
        return {"tost_passed": False}

    # Aggregate deltas by base_sentence_id cluster
    cluster_sums: Dict[str, float] = {}
    cluster_counts: Dict[str, int] = {}
    for p, d in zip(pairs_data, raw_deltas):
        cid = p["base_sentence_id"]
        cluster_sums[cid] = cluster_sums.get(cid, 0.0) + float(d)
        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

    cluster_means = np.array([cluster_sums[cid] / cluster_counts[cid] for cid in cluster_sums.keys()])
    N_clusters = len(cluster_means)

    if N_clusters < 2:
        return {"tost_passed": False}

    # Evaluator-specific margins
    if evaluator_type == "exact_lexicon":
        bound_delta = 2.0  # Terms per 100 words
    else:
        bound_delta = 0.02  # Probability points

    mean_d = float(np.mean(cluster_means))
    std_d = float(np.std(cluster_means, ddof=1))
    se_d = std_d / np.sqrt(N_clusters)

    t_lower = (mean_d - (-bound_delta)) / max(1e-9, se_d)
    p_lower = float(1.0 - stats.t.cdf(t_lower, df=N_clusters - 1))

    t_upper = (mean_d - bound_delta) / max(1e-9, se_d)
    p_upper = float(stats.t.cdf(t_upper, df=N_clusters - 1))

    p_tost = max(p_lower, p_upper)
    tost_passed = bool(p_tost < alpha)

    return {
        "evaluator_type": evaluator_type,
        "evaluator_specific_margin_delta": bound_delta,
        "cluster_sample_size_N": N_clusters,
        "mean_signed_drift": round(mean_d, 4),
        "standard_error": round(se_d, 4),
        "p_value_tost": round(p_tost, 4),
        "mean_signed_drift_equivalence_passed": tost_passed,
        "status_label": "Mean Signed Drift Equivalence Passed" if tost_passed else "Mean Signed Drift Equivalence Failed",
    }
