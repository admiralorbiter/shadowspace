"""Blinded Rating Packet Generator for Phase EDU-2a Canary."""

from __future__ import annotations

import csv
import json
import os
import random
from typing import Dict, List, Tuple

from research.education_audit.evaluation.blinding import blind_generation_text
from research.education_audit.schemas import GenerationRecord


def generate_blinded_rating_packet(
    gen_records: List[GenerationRecord],
    variants_map: Dict[str, Any],
    out_dir: str = "results/education_audit/edu_2a",
    private_key_dir: str = "private_review",
    seed: int = 888,
) -> Tuple[str, str, str]:
    """Generates randomized blinded rating packet CSV, JSONL, and private uncommitted blinding key."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(private_key_dir, exist_ok=True)
    rng = random.Random(seed)

    records_shuffled = list(gen_records)
    rng.shuffle(records_shuffled)

    # Include 5 duplicate letters under new IDs to measure intra-rater consistency
    duplicates = rng.sample(records_shuffled, k=min(5, len(records_shuffled)))
    combined_records = [(g, False) for g in records_shuffled] + [(g, True) for g in duplicates]
    rng.shuffle(combined_records)

    packet_rows = []
    blinding_key = {}

    fieldnames = [
        "letter_id",
        "blinded_text",
        "identity_leakage_flag",
        "reviewer_id",
        "recommendation_strength_score",
        "factual_fidelity_score",
        "unsupported_positive_claims_count",
        "unsupported_negative_claims_count",
        "major_accomplishment_omissions_count",
        "explicit_endorsement_flag",
        "opportunity_strength_score",
        "leadership_language_score",
        "competence_language_score",
        "warmth_language_score",
        "placeholder_or_template_artifact",
        "incomplete_letter_flag",
        "reviewer_notes",
    ]

    for idx, (gen, is_duplicate) in enumerate(combined_records, start=1):
        letter_id = f"LTR_R1_{idx:03d}"
        var = variants_map.get(gen.variant_id)

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

        row = {fn: "" for fn in fieldnames}
        row["letter_id"] = letter_id
        row["blinded_text"] = blinded_text
        row["identity_leakage_flag"] = leakage
        packet_rows.append(row)

    # Export CSV
    csv_path = os.path.join(out_dir, "blinded_rating_packet.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(packet_rows)

    # Export JSONL
    jsonl_path = os.path.join(out_dir, "blinded_rating_packet.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in packet_rows:
            f.write(json.dumps(r) + "\n")

    # Export secret private blinding key (outside Git)
    key_path = os.path.join(private_key_dir, "edu_2a_r1_blinding_key.json")
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(blinding_key, f, indent=2)

    print(f"Blinded Rating Packet exported to {out_dir} ({len(packet_rows)} letters including 5 intra-rater duplicates). Private blinding key exported to {key_path}")
    return csv_path, jsonl_path, key_path

