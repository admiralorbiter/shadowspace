"""Grammar and Purity Validation for Counterfactual Pairs."""

from __future__ import annotations

from typing import Dict


def validate_counterfactual_pair(text_masc: str, text_fem: str, category: str) -> bool:
    """Enforces strict purity assertions: text_masc != text_fem and non-empty strings."""
    if not text_masc or not text_fem:
        raise ValueError("Counterfactual texts must be non-empty strings.")

    if text_masc == text_fem:
        raise ValueError("Counterfactual Masculine and Feminine texts must be distinct (text_masc != text_fem).")

    return True
