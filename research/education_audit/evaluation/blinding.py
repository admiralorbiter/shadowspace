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
    """Redacts identity cues from generated recommendation letter text.

    Replaces explicit names with [CANDIDATE] and pronouns with they/them/their.
    Detects if model generated unexpected gendered/cultural identity terms.
    """
    blinded = output_text
    leakage = False
    details = []

    # 1. Redact Name
    if student_name and student_name != "Student A":
        pattern = re.compile(re.escape(student_name), re.IGNORECASE)
        if pattern.search(blinded):
            blinded = pattern.sub("[CANDIDATE]", blinded)

    # Replace "Student A"
    blinded = re.sub(r"\bStudent A\b", "[CANDIDATE]", blinded, flags=re.IGNORECASE)

    # 2. Redact Gendered Pronouns if explicit
    if pronoun_subject in ["he", "she"]:
        # Subject: He / She -> They
        blinded = re.sub(r"\bHe\b", "They", blinded)
        blinded = re.sub(r"\bhe\b", "they", blinded)
        blinded = re.sub(r"\bShe\b", "They", blinded)
        blinded = re.sub(r"\bshe\b", "they", blinded)

    if pronoun_object in ["him", "her"]:
        # Object: him / her -> them
        blinded = re.sub(r"\bhim\b", "them", blinded)
        blinded = re.sub(r"\bher\b", "them", blinded)

    if pronoun_possessive in ["his", "her"]:
        # Possessive: His / Her -> Their / their
        blinded = re.sub(r"\bHis\b", "Their", blinded)
        blinded = re.sub(r"\bhis\b", "their", blinded)
        blinded = re.sub(r"\bHer\b", "Their", blinded)
        blinded = re.sub(r"\bher\b", "their", blinded)

    # 3. Check for residual unredacted gender leakage
    residual_gender_terms = re.findall(r"\b(he|she|him|her|his|hers|himself|herself|man|woman|boy|girl|mr|ms|mrs)\b", blinded, re.IGNORECASE)
    if residual_gender_terms:
        leakage = True
        details.append(f"Residual gender terms detected: {set(residual_gender_terms)}")

    return blinded, leakage, "; ".join(details) if details else "None"
