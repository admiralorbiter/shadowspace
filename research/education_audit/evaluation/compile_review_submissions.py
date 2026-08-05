"""Reviewer Submission Compiler for Phase EDU-2a-R1.2c.

Reads completed working copy CSV packets for Reviewer 1 and Reviewer 2,
transforms them into standardized long-format JSONL records, and exports
to manual_ratings.jsonl without modifying original frozen private CSV packets.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Tuple

from research.education_audit.evaluation.validate_manual_ratings import validate_manual_ratings_file


def compile_review_submissions(
    working_dir: str = "private_review",
    out_dir: str = "results/education_audit/edu_2a",
    valid_letter_ids: List[str] = None,
) -> Tuple[str, List[Dict[str, Any]], bool, List[str]]:
    """Compiles working copy CSV submissions into manual_ratings.jsonl.

    Expects working copy files:
    - private_review/r1_pass1_submission.csv (or edu_2a_r1_reviewer1_pass1.csv)
    - private_review/r1_pass2_submission.csv (or edu_2a_r1_reviewer1_pass2.csv)
    - private_review/r2_pass1_submission.csv (or edu_2a_r1_reviewer2_pass1.csv)
    - private_review/r2_pass2_submission.csv (or edu_2a_r1_reviewer2_pass2.csv)
    """
    os.makedirs(out_dir, exist_ok=True)
    out_ratings_path = os.path.join(out_dir, "manual_ratings.jsonl")

    # Map possible filenames
    file_map = [
        ("R1", 1, ["r1_pass1_submission.csv", "edu_2a_r1_reviewer1_pass1.csv"]),
        ("R1", 2, ["r1_pass2_submission.csv", "edu_2a_r1_reviewer1_pass2.csv"]),
        ("R2", 1, ["r2_pass1_submission.csv", "edu_2a_r1_reviewer2_pass1.csv", "edu_2a_r2_reviewer2_pass1.csv"]),
        ("R2", 2, ["r2_pass2_submission.csv", "edu_2a_r1_reviewer2_pass2.csv", "edu_2a_r2_reviewer2_pass2.csv"]),
    ]

    compiled_records: List[Dict[str, Any]] = []

    def parse_val(v: str) -> Any:
        v_str = str(v).strip()
        if v_str.lower() == "true":
            return True
        if v_str.lower() == "false":
            return False
        try:
            if "." in v_str:
                return float(v_str)
            return int(v_str)
        except ValueError:
            return v_str

    for r_id, r_pass, candidates in file_map:
        found_file = None
        for cand in candidates:
            p = os.path.join(working_dir, cand)
            if os.path.exists(p) and os.path.getsize(p) > 0:
                found_file = p
                break

        if not found_file:
            return out_ratings_path, [], False, [f"Missing working copy submission for {r_id} Pass {r_pass}"]

        with open(found_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                l_id = row.get("letter_id", "").strip()
                if not l_id:
                    continue

                rec: Dict[str, Any] = {
                    "rating_id": f"{r_id}_P{r_pass}_{l_id}",
                    "reviewer_id": r_id,
                    "letter_id": l_id,
                    "review_pass": r_pass,
                }

                for k, v in row.items():
                    if k in ["letter_id", "blinded_text", "target_opportunity", "verified_accomplishments", "identity_leakage_flag"]:
                        continue
                    if v is not None and str(v).strip() != "":
                        rec[k] = parse_val(v)

                compiled_records.append(rec)

    # Write compiled long-format JSONL
    with open(out_ratings_path, "w", encoding="utf-8") as f:
        for rec in compiled_records:
            f.write(json.dumps(rec) + "\n")

    # Validate compiled output if valid_letter_ids provided
    if valid_letter_ids:
        valid, errs = validate_manual_ratings_file(out_ratings_path, valid_letter_ids, private_key_dir=working_dir)
        return out_ratings_path, compiled_records, valid, errs

    return out_ratings_path, compiled_records, True, []
