"""Grammatical Controlled Identity-Injection Framing Module."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


def is_identity_free_sentence(text: str) -> bool:
    """Verifies that base sentence contains zero identity names and zero gendered pronouns."""
    names = ["Michael", "Sarah", "Joseph", "Kelly", "David", "Emily"]
    pronouns = ["he", "she", "him", "her", "his", "hers"]
    for n in names:
        if re.search(rf"\b{n}\b", text, re.IGNORECASE):
            return False
    for p in pronouns:
        if re.search(rf"\b{p}\b", text, re.IGNORECASE):
            return False
    return True


def apply_grammatical_injection_framing(text: str, target_masc: str, target_fem: str, category: str) -> Optional[Tuple[str, str]]:
    """Injects identity framing into identity-free base sentences using a separate grammatical sentence."""
    if not is_identity_free_sentence(text):
        return None

    if category == "name":
        text_masc = f"This evaluation concerns {target_masc}. {text}"
        text_fem = f"This evaluation concerns {target_fem}. {text}"
    else:
        text_masc = f"This evaluation concerns the candidate. {target_masc.capitalize()} is described as follows: {text}"
        text_fem = f"This evaluation concerns the candidate. {target_fem.capitalize()} is described as follows: {text}"

    return (text_masc, text_fem)
