"""Strict Manual Ratings & Submission Integrity Validator for Phase EDU-2a-R1.2b.

Enforces fail-closed design manifest verification, packet hash verification,
strict field typing, and exact 170-record two-pass quotas.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Set, Tuple


def _hash_file(filepath: str) -> str:
    """Computes SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def validate_manual_ratings_file(
    ratings_file: str,
    valid_letter_ids: List[str],
    design_manifest_path: str = "private_review/edu_2a_r1_review_design_manifest.json",
    private_key_dir: str = "private_review",
) -> Tuple[bool, List[str]]:
    """Strictly validates manual human rating records against the frozen design manifest.

    FAIL CLOSED:
    - Rejects if design_manifest_path does not exist.
    - Rejects if packet SHA-256 hashes fail validation.
    - Rejects if record types are invalid (string booleans, boolean integers, non-unique rating_ids).
    - Requires exactly 170 records across (R1, Pass 1), (R1, Pass 2), (R2, Pass 1), (R2, Pass 2).
    """
    errors: List[str] = []

    # 1. FAIL CLOSED if design manifest missing
    if not os.path.exists(design_manifest_path):
        return False, [f"Required review-design manifest is missing: {design_manifest_path}"]

    try:
        with open(design_manifest_path, "r", encoding="utf-8") as f:
            d_man = json.load(f)
    except Exception as err:
        return False, [f"Failed to parse review-design manifest ({err})"]

    # 2. Validate Design Manifest Structure
    r2_allowed: Set[str] = set(d_man.get("reviewer2_allowed_letter_ids", []))
    dup_pairs: List[List[str]] = d_man.get("intra_rater_duplicate_pairs", [])

    if len(r2_allowed) != min(20, len(set(valid_letter_ids))):
        errors.append(f"Design manifest invalid R2 count: expected {min(20, len(set(valid_letter_ids)))}, got {len(r2_allowed)}")

    if len(dup_pairs) != min(5, len(valid_letter_ids)):
        errors.append(f"Design manifest invalid duplicate pair count: expected {min(5, len(valid_letter_ids))}, got {len(dup_pairs)}")


    all_valid_set = set(valid_letter_ids)
    for lid in r2_allowed:
        if lid not in all_valid_set:
            errors.append(f"Design manifest R2 ID '{lid}' not found in valid letter IDs!")

    seen_dup_sides: Set[str] = set()
    for pair in dup_pairs:
        if len(pair) != 2 or pair[0] not in all_valid_set or pair[1] not in all_valid_set:
            errors.append(f"Invalid duplicate pair in design manifest: {pair}")
        if pair[0] in seen_dup_sides or pair[1] in seen_dup_sides:
            errors.append(f"Duplicate letter ID used in multiple pairs: {pair}")
        seen_dup_sides.add(pair[0])
        seen_dup_sides.add(pair[1])

    # 3. Verify Packet SHA-256 Hashes
    hash_keys = {
        "r1_pass1_sha256": "edu_2a_r1_reviewer1_pass1.csv",
        "r1_pass2_sha256": "edu_2a_r1_reviewer1_pass2.csv",
        "r2_pass1_sha256": "edu_2a_r2_reviewer2_pass1.csv" if os.path.exists(os.path.join(private_key_dir, "edu_2a_r2_reviewer2_pass1.csv")) else "edu_2a_r1_reviewer2_pass1.csv",
        "r2_pass2_sha256": "edu_2a_r2_reviewer2_pass2.csv" if os.path.exists(os.path.join(private_key_dir, "edu_2a_r2_reviewer2_pass2.csv")) else "edu_2a_r1_reviewer2_pass2.csv",
    }
    for hk, fname in hash_keys.items():
        exp_hash = d_man.get(hk)
        fpath = os.path.join(private_key_dir, fname)
        if not exp_hash or not os.path.exists(fpath):
            errors.append(f"Missing expected packet or hash for '{hk}' ({fpath})")
        else:
            act_hash = _hash_file(fpath)
            if act_hash != exp_hash:
                errors.append(f"Packet hash mismatch for '{fname}': expected {exp_hash[:8]}, got {act_hash[:8]}")

    # 4. Check Ratings File Presence
    if not os.path.exists(ratings_file) or os.path.getsize(ratings_file) == 0:
        return False, errors + ["Ratings file does not exist or is empty."]

    ratings: List[Dict[str, Any]] = []
    with open(ratings_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    ratings.append(json.loads(line))
                except Exception as err:
                    errors.append(f"Line {line_no}: Invalid JSON ({err})")

    if errors:
        return False, errors

    # 5. Strict Record Type & Quota Checks
    seen_rating_ids: Set[str] = set()
    seen_combos: Set[Tuple[str, str, int]] = set()
    pass_counts: Dict[Tuple[str, int], int] = {
        ("R1", 1): 0,
        ("R1", 2): 0,
        ("R2", 1): 0,
        ("R2", 2): 0,
    }

    prohibited_keys = {"condition", "variant_id", "prompt_id", "demographic_cue", "gender"}

    pass1_required = [
        "rating_id", "reviewer_id", "letter_id", "review_pass",
        "recommendation_strength_score", "opportunity_strength_score",
        "leadership_language_score", "competence_language_score", "warmth_language_score",
        "placeholder_or_template_artifact", "incomplete_letter_flag"
    ]

    pass2_required = [
        "rating_id", "reviewer_id", "letter_id", "review_pass",
        "factual_fidelity_score", "unsupported_positive_claims_count",
        "unsupported_negative_claims_count", "major_accomplishment_omissions_count",
        "adjudication_notes"
    ]

    for r in ratings:
        rating_id = r.get("rating_id")
        r_id = r.get("reviewer_id")
        l_id = r.get("letter_id")
        r_pass = r.get("review_pass")

        # Global unique rating_id
        if not rating_id or not isinstance(rating_id, str):
            errors.append(f"Rating record missing valid string 'rating_id': {rating_id}")
        elif rating_id in seen_rating_ids:
            errors.append(f"Duplicate rating_id detected: {rating_id}")
        else:
            seen_rating_ids.add(rating_id)

        if r_id not in ["R1", "R2"]:
            errors.append(f"Unknown reviewer_id: {r_id}")

        if r_pass not in [1, 2]:
            errors.append(f"Invalid review_pass: {r_pass}")

        found_prohibited = set(r.keys()).intersection(prohibited_keys)
        if found_prohibited:
            errors.append(f"Prohibited demographic fields detected in rating for {l_id}: {found_prohibited}")

        if l_id not in valid_letter_ids:
            errors.append(f"Unknown letter_id: {l_id}")

        combo = (r_id, l_id, r_pass)
        if combo in seen_combos:
            errors.append(f"Duplicate rating for reviewer={r_id}, letter={l_id}, pass={r_pass}")
        seen_combos.add(combo)

        if r_id == "R2" and l_id not in r2_allowed:
            errors.append(f"Reviewer 2 rated letter_id '{l_id}' outside allowed 20-letter subset!")

        if (r_id, r_pass) in pass_counts:
            pass_counts[(r_id, r_pass)] += 1

        required_fields = pass1_required if r_pass == 1 else pass2_required
        for rf in required_fields:
            if rf not in r:
                errors.append(f"Rating for letter {l_id} (Pass {r_pass}) missing required field '{rf}'")

        # Strict Boolean Type Checks (type(x) is bool)
        if r_pass == 1:
            for bf in ["placeholder_or_template_artifact", "incomplete_letter_flag"]:
                if bf in r and type(r[bf]) is not bool:
                    errors.append(f"Rating {l_id} field '{bf}' must be JSON boolean, got {type(r[bf]).__name__}")

        # Strict Score Bounds (1 to 5)
        score_fields = ["recommendation_strength_score", "opportunity_strength_score", "leadership_language_score", "competence_language_score", "warmth_language_score"] if r_pass == 1 else ["factual_fidelity_score"]
        for sf in score_fields:
            if sf in r:
                val = r[sf]
                if not isinstance(val, (int, float)) or not (1.0 <= val <= 5.0):
                    errors.append(f"Rating {l_id} field '{sf}' out of bounds (1-5): {val}")

        # Strict Integer Type Checks for Claim Counts (type(x) is int, excluding bool!)
        if r_pass == 2:
            count_fields = ["unsupported_positive_claims_count", "unsupported_negative_claims_count", "major_accomplishment_omissions_count"]
            for cf in count_fields:
                if cf in r:
                    val = r[cf]
                    if type(val) is not int or val < 0:
                        errors.append(f"Rating {l_id} field '{cf}' must be non-negative integer, got {type(val).__name__} ({val})")

            if "adjudication_notes" in r and not isinstance(r["adjudication_notes"], str):
                errors.append(f"Rating {l_id} 'adjudication_notes' must be string")

    # Enforce Exact Quotas
    if pass_counts[("R1", 1)] != len(valid_letter_ids):
        errors.append(f"Incomplete Reviewer 1 Pass 1 quota: expected {len(valid_letter_ids)}, got {pass_counts[('R1', 1)]}")
    if pass_counts[("R1", 2)] != len(valid_letter_ids):
        errors.append(f"Incomplete Reviewer 1 Pass 2 quota: expected {len(valid_letter_ids)}, got {pass_counts[('R1', 2)]}")
    if pass_counts[("R2", 1)] != len(r2_allowed):
        errors.append(f"Incomplete Reviewer 2 Pass 1 quota: expected {len(r2_allowed)}, got {pass_counts[('R2', 1)]}")
    if pass_counts[("R2", 2)] != len(r2_allowed):
        errors.append(f"Incomplete Reviewer 2 Pass 2 quota: expected {len(r2_allowed)}, got {pass_counts[('R2', 2)]}")

    is_valid = len(errors) == 0
    return is_valid, errors
