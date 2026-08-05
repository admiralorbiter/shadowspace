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
    seed: int = 777,
) -> Tuple[str, str, str]:
    """Generates randomized blinded rating packet CSV, JSONL, and secret blinding_key.json."""
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    # Shuffle generation records
    records_shuffled = list(gen_records)
    rng.shuffle(records_shuffled)

    packet_rows = []
    blinding_key = {}

    for idx, gen in enumerate(records_shuffled, start=1):
        letter_id = f"LTR_{idx:03d}"
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
        }

        packet_rows.append({
            "letter_id": letter_id,
            "blinded_text": blinded_text,
            "identity_leakage_flag": leakage,
        })

    # Export CSV
    csv_path = os.path.join(out_dir, "blinded_rating_packet.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["letter_id", "blinded_text", "identity_leakage_flag"])
        writer.writeheader()
        writer.writerows(packet_rows)

    # Export JSONL
    jsonl_path = os.path.join(out_dir, "blinded_rating_packet.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in packet_rows:
            f.write(json.dumps(r) + "\n")

    # Export secret blinding key
    key_path = os.path.join(out_dir, "blinding_key.json")
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(blinding_key, f, indent=2)

    print(f"Blinded Rating Packet exported to {out_dir} (CSV & JSONL). Blinding key exported to {key_path}")
    return csv_path, jsonl_path, key_path
