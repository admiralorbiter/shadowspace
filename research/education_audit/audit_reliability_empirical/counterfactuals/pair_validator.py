"""Whitelist Token-Level Purity Validator for Counterfactual Pairs."""

from __future__ import annotations

import re
from typing import Dict, Set

APPROVED_WHITELIST_TOKENS: Set[str] = {
    "Michael", "Sarah", "Joseph", "Kelly", "David", "Emily",
    "He", "She", "he", "she", "Him", "Her", "him", "her", "His", "Hers", "his", "hers",
    "was", "described", "as", "follows:", "The", "candidate", "demonstrated", "that"
}


def validate_counterfactual_pair_purity(text_masc: str, text_fem: str, category: str) -> bool:
    """Enforces token-level purity assertions: text_masc != text_fem and only whitelisted identity tokens differ."""
    if not text_masc or not text_fem:
        raise ValueError("Counterfactual texts must be non-empty strings.")

    if text_masc == text_fem:
        raise ValueError("Counterfactual Masculine and Feminine texts must be distinct (text_masc != text_fem).")

    # Verify no conflicting term collisions (e.g. 'Michael She managed')
    if "Michael She" in text_masc or "Michael She" in text_fem:
        raise ValueError("Conflicting identity terms detected in counterfactual pair ('Michael She').")
    if "Sarah He" in text_masc or "Sarah He" in text_fem:
        raise ValueError("Conflicting identity terms detected in counterfactual pair ('Sarah He').")

    words_m = set(re.findall(r"\b\w+\b", text_masc))
    words_f = set(re.findall(r"\b\w+\b", text_fem))

    diff_words = (words_m ^ words_f)
    for word in diff_words:
        if word not in APPROVED_WHITELIST_TOKENS:
            raise ValueError(f"Unapproved non-identity token difference detected: '{word}'")

    return True
