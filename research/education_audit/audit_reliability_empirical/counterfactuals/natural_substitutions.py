"""Symmetric Natural Pronoun Substitutions via Canonical Identity-Neutral Representation."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


def apply_symmetric_natural_pronoun_swap(text: str) -> Optional[Tuple[str, str]]:
    """Performs symmetric natural pronoun substitution for both masculine- and feminine-source sentences."""
    # Count distinct identity references; reject multi-person sentences
    masc_pronouns = len(re.findall(r"\b(he|him|his)\b", text, re.IGNORECASE))
    fem_pronouns = len(re.findall(r"\b(she|her|hers)\b", text, re.IGNORECASE))

    if masc_pronouns > 0 and fem_pronouns > 0:
        # Mixed/multiple gender references present; reject pair for purity
        return None

    if masc_pronouns == 0 and fem_pronouns == 0:
        return None

    # Canonicalize to neutral tokens
    canonical = text
    canonical = re.sub(r"\bShe\b", "[SHE_HE]", canonical)
    canonical = re.sub(r"\bshe\b", "[she_he]", canonical)
    canonical = re.sub(r"\bHe\b", "[SHE_HE]", canonical)
    canonical = re.sub(r"\bhe\b", "[she_he]", canonical)

    canonical = re.sub(r"\bHim\b", "[HER_HIM]", canonical)
    canonical = re.sub(r"\bhim\b", "[her_him]", canonical)

    # Render Masculine and Feminine conditions from shared canonical representation
    text_masc = canonical
    text_masc = re.sub(r"\[SHE_HE\]", "He", text_masc)
    text_masc = re.sub(r"\[she_he\]", "he", text_masc)
    text_masc = re.sub(r"\[HER_HIM\]", "Him", text_masc)
    text_masc = re.sub(r"\[her_him\]", "him", text_masc)
    text_masc = re.sub(r"\bHer\b", "His", text_masc)
    text_masc = re.sub(r"\bher\b", "his", text_masc)

    text_fem = canonical
    text_fem = re.sub(r"\[SHE_HE\]", "She", text_fem)
    text_fem = re.sub(r"\[she_he\]", "she", text_fem)
    text_fem = re.sub(r"\[HER_HIM\]", "Her", text_fem)
    text_fem = re.sub(r"\[her_him\]", "her", text_fem)
    text_fem = re.sub(r"\bHis\b", "Her", text_fem)
    text_fem = re.sub(r"\bhis\b", "her", text_fem)

    if text_masc == text_fem:
        return None

    return (text_masc, text_fem)
