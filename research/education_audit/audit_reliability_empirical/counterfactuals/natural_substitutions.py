"""POS-Aware Natural In-Place Identity Substitutions."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

PRONOUN_POS_PAIRS = [
    (r"\bHe\b", "He", "She"),
    (r"\bhe\b", "he", "she"),
    (r"\bHim\b", "Him", "Her"),
    (r"\bhim\b", "him", "her"),
    (r"\bHis\b", "His", "Her"),
    (r"\bhis\b", "his", "her"),
    (r"\bHers\b", "His", "Hers"),
    (r"\bhers\b", "his", "hers"),
]


def apply_natural_pronoun_swap(text: str) -> Optional[Tuple[str, str]]:
    """Performs natural POS-aware in-place pronoun substitution if text contains pronouns."""
    has_pronouns = any(re.search(pat, text) for pat, _, _ in PRONOUN_POS_PAIRS)
    if not has_pronouns:
        return None

    text_masc = text
    text_fem = text

    for pat, masc_tok, fem_tok in PRONOUN_POS_PAIRS:
        text_masc = re.sub(pat, masc_tok, text_masc)
        text_fem = re.sub(pat, fem_tok, text_fem)

    if text_masc == text_fem:
        return None

    return (text_masc, text_fem)
