"""Tier 1 Reversible Entity Renaming Transformation for Phase E2."""

from __future__ import annotations

import re
from typing import Tuple
from research.holonomy.natural_language.transforms.base import BaseTransform


class ReversibleEntityRenameTransform(BaseTransform):
    """Tier 1 Reversible Entity Renaming: e1 <-> e2 preserving text path closure."""

    def __init__(self, name: str, entity1: str, entity2: str) -> None:
        self._name = name
        self.entity1 = entity1
        self.entity2 = entity2

    @property
    def name(self) -> str:
        return self._name

    def _swap_text(self, text: str) -> str:
        placeholder = "___ENT_PLACEHOLDER___"
        # Match whole words
        pattern1 = re.compile(rf"\b{re.escape(self.entity1)}\b", re.IGNORECASE)
        pattern2 = re.compile(rf"\b{re.escape(self.entity2)}\b", re.IGNORECASE)

        s = pattern1.sub(placeholder, text)
        s = pattern2.sub(self.entity1, s)
        s = s.replace(placeholder, self.entity2)
        return s

    def apply(self, premise: str, hypothesis: str) -> Tuple[str, str]:
        return self._swap_text(premise), self._swap_text(hypothesis)

    def invert(self, premise: str, hypothesis: str) -> Tuple[str, str]:
        return self.apply(premise, hypothesis)
