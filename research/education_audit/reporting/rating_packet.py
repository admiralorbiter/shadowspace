"""Two-Pass Blinded Rating Packet Generator for Phase EDU-2a-R1.2a."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Tuple

from research.education_audit.evaluation.blinding import blind_generation_text
from research.education_audit.schemas import GenerationRecord


def _hash_file(filepath: str) -> str:
    """Computes SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def generate_blinded_rating_packet(
    gen_records: List[GenerationRecord],
    variants_map: Dict[str, Any],
    cases_map: Dict[str, Any] = None,
    out_dir: str = "results/education_audit/edu_2a",
    private_key_dir: str = "private_review",
    seed: int = 888,
) -> Tuple[str, str, str]:
    """Generates randomized two-pass blinded rating packets (Pass 1 & Pass 2) for R1 & R2."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(private_key_dir, exist_ok=True)
    rng = random.Random(seed)

    records_shuffled = list(gen_records)
    rng.shuffle(records_shuffled)

    # 1. Select 5 duplicate letters under new IDs for Reviewer 1 intra-rater consistency
    duplicates = rng.sample(records_shuffled, k=min(5, len(records_shuffled)))
    r1_combined = [(g, False) for g in records_shuffled] + [(g, True) for g in duplicates]
    rng.shuffle(r1_combined)

    # 2. Select 20 unique letters for Reviewer 2 (2 profiles x 2 prompts x 5 conditions = 20)
    # Rotate seeds deterministically across cells: seed 101, 202, 303
    r2_subset: List[GenerationRecord] = []
    cell_groups: Dict[Tuple[str, str, str], List[GenerationRecord]] = {}
    for g in records_shuffled:
        cond = g.condition
        cell_key = (g.case_id, g.prompt_id, cond)
        cell_groups.setdefault(cell_key, []).append(g)

    sorted_cells = sorted(cell_groups.keys())
    seed_choices = [101, 202, 303]
    for idx, cell_k in enumerate(sorted_cells):
        target_seed = seed_choices[idx % len(seed_choices)]
        cell_gens = cell_groups[cell_k]
        matched = [g for g in cell_gens if g.parameters.get("requested_seed") == target_seed]
        chosen = matched[0] if matched else cell_gens[0]
        r2_subset.append(chosen)

    expected_r2_count = min(20, len(cell_groups))
    assert len(r2_subset) == expected_r2_count, f"Reviewer 2 subset size error: expected {expected_r2_count}, got {len(r2_subset)}"

    # 3. Define Packet Field Schema
    pass1_fields = [
        "letter_id",
        "blinded_text",
        "identity_leakage_flag",
        "recommendation_strength_score",
        "opportunity_strength_score",
        "leadership_language_score",
        "competence_language_score",
        "warmth_language_score",
        "placeholder_or_template_artifact",
        "incomplete_letter_flag",
        "reviewer_notes",
    ]

    pass2_fields = [
        "letter_id",
        "blinded_text",
        "target_opportunity",
        "verified_accomplishments",
        "factual_fidelity_score",
        "unsupported_positive_claims_count",
        "unsupported_negative_claims_count",
        "major_accomplishment_omissions_count",
        "adjudication_notes",
    ]

    packet_rows_public = []
    blinding_key = {}
    gen_id_to_letter_id: Dict[str, str] = {}
    duplicate_pairs: List[Tuple[str, str]] = []

    r1_pass1_rows, r1_pass2_rows = [], []
    r2_pass1_rows, r2_pass2_rows = [], []

    # Map original letters first
    for idx, (gen, is_duplicate) in enumerate(r1_combined, start=1):
        letter_id = f"LTR_R1_{idx:03d}"
        var = variants_map.get(gen.variant_id)
        c_obj = cases_map.get(gen.case_id) if cases_map else None

        student_name = var.student_name if var else "Student A"
        subj = var.pronoun_subject if var else "they"
        obj = var.pronoun_object if var else "them"
        poss = var.pronoun_possessive if var else "their"

        blinded_text, leakage, details = blind_generation_text(
            gen.output_text, student_name, subj, obj, poss
        )

        blinding_key[letter_id] = {
            "generation_id": gen.generation_id,
            "case_id": gen.case_id,
            "variant_id": gen.variant_id,
            "condition": gen.condition,
            "prompt_id": gen.prompt_id,
            "repeat_index": gen.repeat_index,
            "requested_seed": gen.parameters.get("requested_seed"),
            "identity_leakage_detected": leakage,
            "identity_leakage_details": details,
            "is_intra_rater_duplicate": is_duplicate,
        }

        if not is_duplicate and gen.generation_id not in gen_id_to_letter_id:
            gen_id_to_letter_id[gen.generation_id] = letter_id

        # Pass 1 Row
        r1_p1 = {fn: "" for fn in pass1_fields}
        r1_p1["letter_id"] = letter_id
        r1_p1["blinded_text"] = blinded_text
        r1_p1["identity_leakage_flag"] = leakage
        r1_pass1_rows.append(r1_p1)

        # Pass 2 Row
        r1_p2 = {fn: "" for fn in pass2_fields}
        r1_p2["letter_id"] = letter_id
        r1_p2["blinded_text"] = blinded_text
        r1_p2["target_opportunity"] = c_obj.target_opportunity if c_obj else "Opportunity"
        r1_p2["verified_accomplishments"] = " | ".join(c_obj.facts) if c_obj else ""
        r1_pass2_rows.append(r1_p2)

        pub_row = {
            "letter_id": letter_id,
            "blinded_text": blinded_text,
            "identity_leakage_flag": leakage,
        }
        packet_rows_public.append(pub_row)

    # Reconstruct duplicate pairs
    for lid, info in blinding_key.items():
        if info["is_intra_rater_duplicate"]:
            orig_lid = gen_id_to_letter_id[info["generation_id"]]
            duplicate_pairs.append((orig_lid, lid))

    # Build R2 Pass 1 and Pass 2 Rows (20 unique letters)
    r2_shuffled = list(r2_subset)
    rng.shuffle(r2_shuffled)
    r2_allowed_lids = []

    for g in r2_shuffled:
        r1_match = gen_id_to_letter_id[g.generation_id]
        r2_allowed_lids.append(r1_match)

        var = variants_map.get(g.variant_id)
        c_obj = cases_map.get(g.case_id) if cases_map else None

        blinded_text, leakage, _ = blind_generation_text(
            g.output_text, var.student_name, var.pronoun_subject, var.pronoun_object, var.pronoun_possessive
        )

        r2_p1 = {fn: "" for fn in pass1_fields}
        r2_p1["letter_id"] = r1_match
        r2_p1["blinded_text"] = blinded_text
        r2_p1["identity_leakage_flag"] = leakage
        r2_pass1_rows.append(r2_p1)

        r2_p2 = {fn: "" for fn in pass2_fields}
        r2_p2["letter_id"] = r1_match
        r2_p2["blinded_text"] = blinded_text
        r2_p2["target_opportunity"] = c_obj.target_opportunity if c_obj else "Opportunity"
        r2_p2["verified_accomplishments"] = " | ".join(c_obj.facts) if c_obj else ""
        r2_pass2_rows.append(r2_p2)

    # Export Public CSV & JSONL Packets
    csv_path = os.path.join(out_dir, "blinded_rating_packet.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["letter_id", "blinded_text", "identity_leakage_flag"])
        writer.writeheader()
        writer.writerows(packet_rows_public)

    jsonl_path = os.path.join(out_dir, "blinded_rating_packet.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in packet_rows_public:
            f.write(json.dumps(r) + "\n")

    key_path = os.path.join(private_key_dir, "edu_2a_r1_blinding_key.json")
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(blinding_key, f, indent=2)

    # Export Private Two-Pass Packets for Reviewer 1 and Reviewer 2
    def write_csv(p: str, fields: List[str], rows: List[Dict[str, Any]]) -> str:
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        return p

    p_r1_p1 = write_csv(os.path.join(private_key_dir, "edu_2a_r1_reviewer1_pass1.csv"), pass1_fields, r1_pass1_rows)
    p_r1_p2 = write_csv(os.path.join(private_key_dir, "edu_2a_r1_reviewer1_pass2.csv"), pass2_fields, r1_pass2_rows)
    p_r2_p1 = write_csv(os.path.join(private_key_dir, "edu_2a_r1_reviewer2_pass1.csv"), pass1_fields, r2_pass1_rows)
    p_r2_p2 = write_csv(os.path.join(private_key_dir, "edu_2a_r1_reviewer2_pass2.csv"), pass2_fields, r2_pass2_rows)

    # Export Frozen Review-Design Manifest
    design_manifest = {
        "reviewer2_allowed_letter_ids": r2_allowed_lids,
        "intra_rater_duplicate_pairs": duplicate_pairs,
        "r1_pass1_sha256": _hash_file(p_r1_p1),
        "r1_pass2_sha256": _hash_file(p_r1_p2),
        "r2_pass1_sha256": _hash_file(p_r2_p1),
        "r2_pass2_sha256": _hash_file(p_r2_p2),
    }

    manifest_path = os.path.join(private_key_dir, "edu_2a_r1_review_design_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(design_manifest, f, indent=2)

    print(f"Two-Pass Rating Packets & Design Manifest exported to {private_key_dir}.")
    return csv_path, jsonl_path, key_path
