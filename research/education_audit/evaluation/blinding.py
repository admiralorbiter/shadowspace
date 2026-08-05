"""Identity Blinding & Leakage Detection Engine for Manual Rating Packets (Phase EDU-2a)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BlindedGenerationRecord:
    letter_id: str                          # Blinded randomized letter ID (e.g. LTR_849201)
    original_generation_id: str
    blinded_text: str
    identity_leakage_detected: bool
    identity_leakage_details: str


def blind_generation_text(
    output_text: str,
    student_name: str,
    pronoun_subject: str,
    pronoun_object: str,
    pronoun_possessive: str,
) -> Tuple[str, bool, str]:
    """Inspects generated recommendation letter text for explicit identity leakage.

    Preserves raw model output text to avoid post-hoc grammatical corruption,
    while detecting and flagging any unredacted names or gendered pronouns.
    """
    leakage = False
    details = []

    # Check explicit name leakage
    if student_name and student_name != "Student A":
        pattern = re.compile(re.escape(student_name), re.IGNORECASE)
        if pattern.search(output_text):
            leakage = True
            details.append(f"Explicit name '{student_name}' detected in output")

    if re.search(r"\bStudent A\b", output_text, re.IGNORECASE):
        leakage = True
        details.append("Explicit name 'Student A' detected in output")

    # Check gendered pronoun / term leakage
    gender_terms = re.findall(
        r"\b(he|she|him|her|his|hers|himself|herself|man|woman|boy|girl|mr|ms|mrs)\b",
        output_text,
        re.IGNORECASE,
    )
    if gender_terms:
        leakage = True
        details.append(f"Gendered terms detected: {set(gender_terms)}")

    return output_text, leakage, "; ".join(details) if details else "None"

