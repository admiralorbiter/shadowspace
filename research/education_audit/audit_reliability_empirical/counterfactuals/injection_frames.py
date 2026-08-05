"""Controlled Neutral Identity-Injection Framing Module."""

from __future__ import annotations

import re
from typing import Dict, Tuple

IDENTITY_NAMES = ["Michael", "Sarah", "Joseph", "Kelly", "David", "Emily"]
IDENTITY_PRONOUNS = ["he", "she", "him", "her", "his", "hers"]


def neutralize_existing_identity_terms(text: str) -> str:
    """Neutralizes pre-existing identity names and pronouns to prevent conflicting term collisions."""
    clean = text
    for name in IDENTITY_NAMES:
        clean = re.sub(rf"\b{name}\b", "the candidate", clean, flags=re.IGNORECASE)
    for pron in ["he", "she"]:
        clean = re.sub(rf"\b{pron}\b", "they", clean, flags=re.IGNORECASE)
    for pron in ["him", "her"]:
        clean = re.sub(rf"\b{pron}\b", "them", clean, flags=re.IGNORECASE)
    for pron in ["his", "hers"]:
        clean = re.sub(rf"\b{pron}\b", "their", clean, flags=re.IGNORECASE)
    return clean


def apply_controlled_injection_framing(text: str, target_masc: str, target_fem: str, category: str) -> Tuple[str, str]:
    """Injects identity framing into neutralized text."""
    clean_text = neutralize_existing_identity_terms(text)
    if category == "name":
        text_masc = f"{target_masc} was described as follows: {clean_text}"
        text_fem = f"{target_fem} was described as follows: {clean_text}"
    else:
        text_masc = f"The candidate demonstrated that {target_masc} {clean_text}"
        text_fem = f"The candidate demonstrated that {target_fem} {clean_text}"
    return (text_masc, text_fem)
