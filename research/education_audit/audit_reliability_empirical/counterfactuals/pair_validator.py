"""Sequence-Level Token Purity Validator and Changed-Span Extractor."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

APPROVED_IDENTITY_TOKENS: Set[str] = {
    "Michael", "Sarah", "Joseph", "Kelly", "David", "Emily",
    "He", "She", "he", "she", "Him", "Her", "him", "her", "His", "Hers", "his", "hers",
    "concerns", "described", "follows:", "candidate."
}


def validate_sequence_token_purity(text_masc: str, text_fem: str, category: str) -> Dict[str, Any]:
    """Validates sequence-level token purity and extracts aligned changed spans."""
    if not text_masc or not text_fem:
        raise ValueError("Counterfactual texts must be non-empty strings.")

    if text_masc == text_fem:
        raise ValueError("Counterfactual Masculine and Feminine texts must be distinct.")

    tokens_m = text_masc.split()
    tokens_f = text_fem.split()

    if len(tokens_m) != len(tokens_f):
        raise ValueError(f"Token length mismatch between masculine ({len(tokens_m)}) and feminine ({len(tokens_f)}) strings.")

    changed_spans = []
    for idx, (tm, tf) in enumerate(zip(tokens_m, tokens_f)):
        if tm != tf:
            clean_m = re.sub(r"[^\w]", "", tm)
            clean_f = re.sub(r"[^\w]", "", tf)
            if clean_m not in APPROVED_IDENTITY_TOKENS and clean_f not in APPROVED_IDENTITY_TOKENS:
                raise ValueError(f"Unapproved token difference at index {idx}: '{tm}' vs '{tf}'")
            changed_spans.append({
                "token_index": idx,
                "masculine": tm,
                "feminine": tf,
                "replacement_class": category,
            })

    if not changed_spans:
        raise ValueError("No token differences detected between counterfactual texts.")

    return {
        "purity_passed": True,
        "changed_spans_count": len(changed_spans),
        "changed_spans": changed_spans,
    }
