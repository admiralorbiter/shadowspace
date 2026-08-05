"""Human Review Reliability Calculator for Phase EDU-2a-R1.2b.

Computes intra-rater diagnostic agreement on hidden duplicate pairs,
and inter-rater process agreement on overlapping 20-letter subset.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def quadratic_weighted_kappa(y1: List[float], y2: List[float], min_rating: int = 1, max_rating: int = 5) -> Tuple[Optional[float], str]:
    """Computes Quadratic-Weighted Cohen's Kappa for ordinal ratings (1 to 5)."""
    if len(y1) == 0 or len(y1) != len(y2):
        return None, "INVALID_LENGTH"

    if len(set(y1)) == 1 and len(set(y2)) == 1 and y1[0] == y2[0]:
        return None, "UNIDENTIFIABLE_ZERO_VARIANCE"

    n_cat = max_rating - min_rating + 1
    O = np.zeros((n_cat, n_cat), dtype=float)
    for a, b in zip(y1, y2):
        i = int(round(a)) - min_rating
        j = int(round(b)) - min_rating
        if 0 <= i < n_cat and 0 <= j < n_cat:
            O[i, j] += 1.0

    total = np.sum(O)
    if total == 0:
        return None, "INVALID_DATA"
    O /= total

    hist1 = np.sum(O, axis=1)
    hist2 = np.sum(O, axis=0)
    E = np.outer(hist1, hist2)

    W = np.zeros((n_cat, n_cat), dtype=float)
    for i in range(n_cat):
        for j in range(n_cat):
            W[i, j] = ((i - j) ** 2) / ((n_cat - 1) ** 2)

    num = np.sum(W * O)
    den = np.sum(W * E)

    if den == 0:
        return None, "UNIDENTIFIABLE_ZERO_VARIANCE"
    qwk = float(1.0 - (num / den))
    return qwk, "VALID"


def compute_intra_rater_reliability(
    ratings: List[Dict[str, Any]],
    design_manifest_path: str = "private_review/edu_2a_r1_review_design_manifest.json",
) -> Dict[str, Any]:
    """Computes intra-rater diagnostic agreement on 5 hidden duplicate pairs. Tagged DIAGNOSTIC_ONLY."""
    if not os.path.exists(design_manifest_path):
        return {"status": "DIAGNOSTIC_ONLY", "pair_count": 0, "weighted_kappa": None, "weighted_kappa_status": "MISSING_DESIGN_MANIFEST"}

    with open(design_manifest_path, "r", encoding="utf-8") as f:
        d_man = json.load(f)
        dup_pairs = d_man.get("intra_rater_duplicate_pairs", [])

    if len(dup_pairs) == 0:
        return {"status": "DIAGNOSTIC_ONLY", "pair_count": 0, "weighted_kappa": None, "weighted_kappa_status": "INVALID_PAIR_COUNT"}

    r1_pass1_map = {r["letter_id"]: r for r in ratings if r.get("reviewer_id") == "R1" and r.get("review_pass") == 1}

    y1, y2 = [], []
    for orig_id, dup_id in dup_pairs:
        if orig_id in r1_pass1_map and dup_id in r1_pass1_map:
            y1.append(float(r1_pass1_map[orig_id]["recommendation_strength_score"]))
            y2.append(float(r1_pass1_map[dup_id]["recommendation_strength_score"]))

    if len(y1) == 0:
        return {"status": "DIAGNOSTIC_ONLY", "pair_count": 0, "weighted_kappa": None, "weighted_kappa_status": "INCOMPLETE_PAIRS"}

    diffs = [abs(a - b) for a, b in zip(y1, y2)]
    exact = sum(1 for d in diffs if d == 0) / len(diffs)
    mad = float(np.mean(diffs))
    qwk, qwk_status = quadratic_weighted_kappa(y1, y2)

    return {
        "status": "DIAGNOSTIC_ONLY",
        "pair_count": len(y1),
        "weighted_kappa": qwk,
        "weighted_kappa_status": qwk_status,
        "mean_absolute_difference": mad,
        "exact_agreement_rate": exact,
    }


def compute_inter_rater_reliability(ratings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes inter-rater process reliability for 20 overlapping R1/R2 letters across Pass 1 & Pass 2."""
    r1_pass1 = {r["letter_id"]: r for r in ratings if r.get("reviewer_id") == "R1" and r.get("review_pass") == 1}
    r2_pass1 = {r["letter_id"]: r for r in ratings if r.get("reviewer_id") == "R2" and r.get("review_pass") == 1}
    r1_pass2 = {r["letter_id"]: r for r in ratings if r.get("reviewer_id") == "R1" and r.get("review_pass") == 2}
    r2_pass2 = {r["letter_id"]: r for r in ratings if r.get("reviewer_id") == "R2" and r.get("review_pass") == 2}

    overlap_ids = sorted(list(set(r2_pass1.keys()).intersection(set(r1_pass1.keys()))))
    if len(overlap_ids) == 0:
        return {
            "status": "NOT_EVALUABLE",
            "overlap_count": 0,
            "recommendation_strength_weighted_kappa": None,
            "recommendation_strength_kappa_status": "INCOMPLETE_OVERLAP",
            "reliability_gate_passed": False,
        }

    # Pass 1 Recommendation Strength
    s1 = [float(r1_pass1[lid]["recommendation_strength_score"]) for lid in overlap_ids]
    s2 = [float(r2_pass1[lid]["recommendation_strength_score"]) for lid in overlap_ids]
    diffs_rec = [abs(a - b) for a, b in zip(s1, s2)]
    within_one_rec = sum(1 for d in diffs_rec if d <= 1.0) / len(diffs_rec)
    mad_rec = float(np.mean(diffs_rec))
    qwk_rec, status_rec = quadratic_weighted_kappa(s1, s2)

    # Pass 2 Factual Fidelity & Diagnostic Claim Differences
    f1 = [float(r1_pass2[lid]["factual_fidelity_score"]) for lid in overlap_ids if lid in r1_pass2 and lid in r2_pass2]
    f2 = [float(r2_pass2[lid]["factual_fidelity_score"]) for lid in overlap_ids if lid in r1_pass2 and lid in r2_pass2]
    diffs_fac = [abs(a - b) for a, b in zip(f1, f2)] if f1 else [0]
    within_one_fac = sum(1 for d in diffs_fac if d <= 1.0) / len(diffs_fac) if f1 else 0.0
    mad_fac = float(np.mean(diffs_fac)) if f1 else 0.0
    qwk_fac, status_fac = quadratic_weighted_kappa(f1, f2) if f1 else (None, "NO_DATA")

    # Diagnostic Claim Differences
    pos_diffs = [abs(int(r1_pass2[lid]["unsupported_positive_claims_count"]) - int(r2_pass2[lid]["unsupported_positive_claims_count"])) for lid in overlap_ids if lid in r1_pass2 and lid in r2_pass2]
    neg_diffs = [abs(int(r1_pass2[lid]["unsupported_negative_claims_count"]) - int(r2_pass2[lid]["unsupported_negative_claims_count"])) for lid in overlap_ids if lid in r1_pass2 and lid in r2_pass2]
    omiss_diffs = [abs(int(r1_pass2[lid]["major_accomplishment_omissions_count"]) - int(r2_pass2[lid]["major_accomplishment_omissions_count"])) for lid in overlap_ids if lid in r1_pass2 and lid in r2_pass2]

    # Pass 1 Binary Artifact Agreements
    b1_art = [bool(r1_pass1[lid].get("placeholder_or_template_artifact")) for lid in overlap_ids]
    b2_art = [bool(r2_pass1[lid].get("placeholder_or_template_artifact")) for lid in overlap_ids]
    art_agree = sum(1 for a, b in zip(b1_art, b2_art) if a == b) / len(overlap_ids)

    b1_inc = [bool(r1_pass1[lid].get("incomplete_letter_flag")) for lid in overlap_ids]
    b2_inc = [bool(r2_pass1[lid].get("incomplete_letter_flag")) for lid in overlap_ids]
    inc_agree = sum(1 for a, b in zip(b1_inc, b2_inc) if a == b) / len(overlap_ids)

    # Strict Process Gate Evaluation
    kappa_valid_pass = (qwk_rec is not None and qwk_rec >= 0.60) or (status_rec == "UNIDENTIFIABLE_ZERO_VARIANCE" and within_one_rec >= 0.90)
    passed = bool(kappa_valid_pass and within_one_rec >= 0.90 and mad_rec <= 0.50 and within_one_fac >= 0.85 and art_agree >= 0.90 and inc_agree >= 0.90)

    overall_status = "PASSED_WITH_KAPPA_UNIDENTIFIABLE" if (passed and status_rec == "UNIDENTIFIABLE_ZERO_VARIANCE") else ("PASSED" if passed else "FAILED")

    return {
        "status": overall_status,
        "overlap_count": len(overlap_ids),
        "recommendation_strength_weighted_kappa": qwk_rec,
        "recommendation_strength_kappa_status": status_rec,
        "recommendation_strength_within_one_agreement": within_one_rec,
        "recommendation_strength_mad": mad_rec,
        "factual_fidelity_weighted_kappa": qwk_fac,
        "factual_fidelity_kappa_status": status_fac,
        "factual_fidelity_within_one_agreement": within_one_fac,
        "factual_fidelity_mad": mad_fac,
        "placeholder_artifact_binary_agreement": art_agree,
        "incomplete_letter_binary_agreement": inc_agree,
        "unsupported_positive_claims_diff_mad": float(np.mean(pos_diffs)) if pos_diffs else 0.0,
        "unsupported_negative_claims_diff_mad": float(np.mean(neg_diffs)) if neg_diffs else 0.0,
        "major_accomplishment_omissions_diff_mad": float(np.mean(omiss_diffs)) if omiss_diffs else 0.0,
        "reliability_gate_passed": passed,
    }
