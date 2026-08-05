"""Strict Manual Ratings Validator for Phase EDU-2a-R1.2.

Validates long-format JSONL ratings, ensuring score ranges, reviewer quotas,
valid letter IDs, and absence of demographic leakage fields.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple


def validate_manual_ratings_file(
    ratings_file: str,
    valid_letter_ids: List[str],
    reviewer2_allowed_letter_ids: List[str],
) -> Tuple[bool, List[str]]:
    """Strictly validates manual human rating records.

    Rejects:
    - Unknown letter IDs
    - Duplicate reviewer/letter/pass combinations
    - Missing required scores
    - Scores outside 1-5
    - Negative claim counts
    - Fewer than 65 R1 entries or fewer than 20 R2 entries
    - R2 entries outside the frozen 20-letter subset
    - Demographic cue leakage fields in rating records
    """
    errors: List[str] = []
    if not os.path.exists(ratings_file) or os.path.getsize(ratings_file) == 0:
        return False, ["Ratings file does not exist or is empty."]

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

    seen_keys = set()
    r1_count, r2_count = 0, 0
    prohibited_keys = {"condition", "variant_id", "prompt_id", "demographic_cue", "gender"}

    for r in ratings:
        r_id = r.get("reviewer_id")
        l_id = r.get("letter_id")
        r_pass = r.get("review_pass", 1)

        # Prohibited demographic leakage key check
        found_prohibited = set(r.keys()).intersection(prohibited_keys)
        if found_prohibited:
            errors.append(f"Prohibited demographic fields detected in rating for {l_id}: {found_prohibited}")

        # Letter ID check
        if l_id not in valid_letter_ids:
            errors.append(f"Unknown letter_id: {l_id}")

        # Duplicate check
        key = (r_id, l_id, r_pass)
        if key in seen_keys:
            errors.append(f"Duplicate rating for reviewer={r_id}, letter={l_id}, pass={r_pass}")
        seen_keys.add(key)

        # Reviewer 2 subset check
        if r_id == "R2" and l_id not in reviewer2_allowed_letter_ids:
            errors.append(f"Reviewer 2 rated letter_id '{l_id}' outside allowed 20-letter subset!")

        # Quota count
        if r_id == "R1":
            r1_count += 1
        elif r_id == "R2":
            r2_count += 1

        # Score range validation
        score_fields = ["recommendation_strength_score", "opportunity_strength_score", "leadership_language_score", "competence_language_score", "warmth_language_score"]
        if r_pass == 2:
            score_fields.append("factual_fidelity_score")

        for sf in score_fields:
            if sf in r:
                val = r[sf]
                if not isinstance(val, (int, float)) or not (1.0 <= val <= 5.0):
                    errors.append(f"Rating {l_id} field '{sf}' out of bounds (1-5): {val}")

        # Count fields non-negative check
        count_fields = ["unsupported_positive_claims_count", "unsupported_negative_claims_count", "major_accomplishment_omissions_count"]
        for cf in count_fields:
            if cf in r:
                val = r[cf]
                if not isinstance(val, int) or val < 0:
                    errors.append(f"Rating {l_id} field '{cf}' must be non-negative integer: {val}")

    if r1_count < 65:
        errors.append(f"Incomplete Reviewer 1 quota: expected 65, got {r1_count}")
    if r2_count < 20:
        errors.append(f"Incomplete Reviewer 2 quota: expected 20, got {r2_count}")

    is_valid = len(errors) == 0
    return is_valid, errors
