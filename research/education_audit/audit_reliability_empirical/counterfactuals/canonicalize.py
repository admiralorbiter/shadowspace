"""Canonicalization utilities for token-level identity substitution."""

from __future__ import annotations

import re
from typing import Dict, Tuple


def apply_identity_swap(text: str, sub_dict: Dict[str, str]) -> str:
    """Applies exact dictionary substitution to text."""
    res = text
    for k, v in sub_dict.items():
        res = re.sub(rf"\b{k}\b", v, res)
    return res
