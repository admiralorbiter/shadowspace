"""Strict Manual Ratings Validator for Phase EDU-2a-R1.2a.

Validates long-format JSONL ratings, enforcing exact 170-record quotas across
Pass 1 and Pass 2 for R1 and R2 against the frozen review-design manifest.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Set, Tuple


def validate_manual_ratings_file(
    ratings_file: str,
    valid_letter_ids: List[str],
    reviewer2_allowed_letter_ids: List[str] = None,
    design_manifest_path: str = "private_review/edu_2a_r1_review_design_manifest.json",
) -> Tuple[bool, List[str]]:
    """Strictly validates manual human rating records.

    Requires exactly 170 records:
    - (R1, Pass 1): 65 records
    - (R1, Pass 2): 65 records
    - (R2, Pass 1): 20 records
    - (R2, Pass 2): 20 records
    """
    errors: List[str] = []
    if not os.path.exists(ratings_file) or os.path.getsize(ratings_file) == 0:
        return False, ["Ratings file does not exist or is empty."]

    # Load frozen review-design manifest if available
    r2_allowed: Set[str] = set(reviewer2_allowed_letter_ids or [])
    if os.path.exists(design_manifest_path):
        try:
            with open(design_manifest_path, "r", encoding="utf-8") as f:
                d_man = json.load(f)
                r2_allowed = set(d_man.get("reviewer2_allowed_letter_ids", list(r2_allowed)))
        except Exception as err:
            errors.append(f"Failed to read design manifest '{design_manifest_path}': {err}")

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

    seen_keys: Set[Tuple[str, str, int]] = set()
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
        r_id = r.get("reviewer_id")
        l_id = r.get("letter_id")
        r_pass = r.get("review_pass")
        rating_id = r.get("rating_id")

        if not rating_id:
            errors.append(f"Rating for letter {l_id} missing 'rating_id'")

        if r_id not in ["R1", "R2"]:
            errors.append(f"Unknown reviewer_id: {r_id}")

        if r_pass not in [1, 2]:
            errors.append(f"Invalid review_pass: {r_pass} (must be 1 or 2)")

        found_prohibited = set(r.keys()).intersection(prohibited_keys)
        if found_prohibited:
            errors.append(f"Prohibited demographic fields detected in rating for {l_id}: {found_prohibited}")

        if l_id not in valid_letter_ids:
            errors.append(f"Unknown letter_id: {l_id}")

        key = (r_id, l_id, r_pass)
        if key in seen_keys:
            errors.append(f"Duplicate rating for reviewer={r_id}, letter={l_id}, pass={r_pass}")
        seen_keys.add(key)

        if r_id == "R2" and r2_allowed and l_id not in r2_allowed:
            errors.append(f"Reviewer 2 rated letter_id '{l_id}' outside allowed 20-letter subset!")

        if (r_id, r_pass) in pass_counts:
            pass_counts[(r_id, r_pass)] += 1

        # Enforce required fields per pass
        required_fields = pass1_required if r_pass == 1 else pass2_required
        for rf in required_fields:
            if rf not in r:
                errors.append(f"Rating for letter {l_id} (Pass {r_pass}) missing required field '{rf}'")

        # Score range validation (1 to 5)
        score_fields = ["recommendation_strength_score", "opportunity_strength_score", "leadership_language_score", "competence_language_score", "warmth_language_score"] if r_pass == 1 else ["factual_fidelity_score"]
        for sf in score_fields:
            if sf in r:
                val = r[sf]
                if not isinstance(val, (int, float)) or not (1.0 <= val <= 5.0):
                    errors.append(f"Rating {l_id} field '{sf}' out of bounds (1-5): {val}")

        # Count fields non-negative check
        if r_pass == 2:
            count_fields = ["unsupported_positive_claims_count", "unsupported_negative_claims_count", "major_accomplishment_omissions_count"]
            for cf in count_fields:
                if cf in r:
                    val = r[cf]
                    if not isinstance(val, int) or val < 0:
                        errors.append(f"Rating {l_id} field '{cf}' must be non-negative integer: {val}")

    # Enforce exact quota sets
    if pass_counts[("R1", 1)] != 65:
        errors.append(f"Incomplete Reviewer 1 Pass 1 quota: expected 65, got {pass_counts[('R1', 1)]}")
    if pass_counts[("R1", 2)] != 65:
        errors.append(f"Incomplete Reviewer 1 Pass 2 quota: expected 65, got {pass_counts[('R1', 2)]}")
    if pass_counts[("R2", 1)] != 20:
        errors.append(f"Incomplete Reviewer 2 Pass 1 quota: expected 20, got {pass_counts[('R2', 1)]}")
    if pass_counts[("R2", 2)] != 20:
        errors.append(f"Incomplete Reviewer 2 Pass 2 quota: expected 20, got {pass_counts[('R2', 2)]}")

    is_valid = len(errors) == 0
    return is_valid, errors
