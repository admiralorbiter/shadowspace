"""Human Review Reliability Calculator for Phase EDU-2a-R1.2.

Computes intra-rater diagnostic agreement on hidden duplicate pairs,
and inter-rater process agreement on overlapping 20-letter subset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np


def quadratic_weighted_kappa(y1: List[float], y2: List[float], min_rating: int = 1, max_rating: int = 5) -> float:
    """Computes Quadratic-Weighted Cohen's Kappa for ordinal ratings (1 to 5)."""
    if len(y1) == 0 or len(y1) != len(y2):
        return 0.0

    n_cat = max_rating - min_rating + 1
    O = np.zeros((n_cat, n_cat), dtype=float)
    for a, b in zip(y1, y2):
        i = int(round(a)) - min_rating
        j = int(round(b)) - min_rating
        if 0 <= i < n_cat and 0 <= j < n_cat:
            O[i, j] += 1.0

    total = np.sum(O)
    if total == 0:
        return 0.0
    O /= total

    # Histogram margins
    hist1 = np.sum(O, axis=1)
    hist2 = np.sum(O, axis=0)
    E = np.outer(hist1, hist2)

    # Quadratic weight matrix
    W = np.zeros((n_cat, n_cat), dtype=float)
    for i in range(n_cat):
        for j in range(n_cat):
            W[i, j] = ((i - j) ** 2) / ((n_cat - 1) ** 2)

    num = np.sum(W * O)
    den = np.sum(W * E)

    if den == 0:
        return 1.0 if num == 0 else 0.0
    return float(1.0 - (num / den))


def compute_intra_rater_reliability(duplicate_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    """Computes intra-rater reliability for Reviewer 1 hidden duplicate pairs (5 pairs). Tagged DIAGNOSTIC_ONLY."""
    if not duplicate_pairs:
        return {"status": "DIAGNOSTIC_ONLY", "pair_count": 0, "weighted_kappa": 0.0, "mean_absolute_difference": 0.0, "exact_agreement_rate": 0.0}

    y1 = [float(p[0]["recommendation_strength_score"]) for p in duplicate_pairs]
    y2 = [float(p[1]["recommendation_strength_score"]) for p in duplicate_pairs]

    diffs = [abs(a - b) for a, b in zip(y1, y2)]
    exact = sum(1 for d in diffs if d == 0) / len(diffs)
    mad = float(np.mean(diffs))
    qwk = quadratic_weighted_kappa(y1, y2)

    return {
        "status": "DIAGNOSTIC_ONLY",
        "pair_count": len(duplicate_pairs),
        "weighted_kappa": qwk,
        "mean_absolute_difference": mad,
        "exact_agreement_rate": exact,
    }


def compute_inter_rater_reliability(r1_records: List[Dict[str, Any]], r2_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes inter-rater process reliability for 20 overlapping R1/R2 letters."""
    r1_map = {r["letter_id"]: r for r in r1_records}
    paired_r1, paired_r2 = [], []

    for r2 in r2_records:
        lid = r2["letter_id"]
        if lid in r1_map:
            paired_r1.append(r1_map[lid])
            paired_r2.append(r2)

    if not paired_r1:
        return {
            "status": "NOT_EVALUABLE",
            "overlap_count": 0,
            "weighted_kappa": 0.0,
            "within_one_point_agreement": 0.0,
            "mean_absolute_difference": 0.0,
            "reliability_gate_passed": False,
        }

    s1 = [float(r["recommendation_strength_score"]) for r in paired_r1]
    s2 = [float(r["recommendation_strength_score"]) for r in paired_r2]

    diffs = [abs(a - b) for a, b in zip(s1, s2)]
    within_one = sum(1 for d in diffs if d <= 1.0) / len(diffs)
    mad = float(np.mean(diffs))
    qwk = quadratic_weighted_kappa(s1, s2)

    # Process gate thresholds: weighted kappa >= 0.60, within-1-point >= 90%, MAD <= 0.50
    passed = bool(qwk >= 0.60 and within_one >= 0.90 and mad <= 0.50)

    return {
        "status": "PASSED" if passed else "ACCEPTABLE_OR_FAILED",
        "overlap_count": len(paired_r1),
        "weighted_kappa": qwk,
        "within_one_point_agreement": within_one,
        "mean_absolute_difference": mad,
        "reliability_gate_passed": passed,
    }
