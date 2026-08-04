"""Tier 1 Reversible Entity Renaming Transformation for Phase E2."""

from __future__ import annotations

import re
from typing import Tuple
from research.holonomy.natural_language.transforms.base import BaseTransform


class ReversibleEntityRenameTransform(BaseTransform):
    """Tier 1 Reversible Entity Renaming: e1 <-> e2 with case preservation."""

    def __init__(self, name: str, entity1: str, entity2: str) -> None:
        self._name = name
        self.entity1 = entity1
        self.entity2 = entity2

    @property
    def name(self) -> str:
        return self._name

    def _preserve_case(self, source_match: str, replacement_target: str) -> str:
        """Preserves casing of matched text."""
        if source_match.isupper():
            return replacement_target.upper()
        elif source_match.istitle():
            return replacement_target.capitalize()
        elif source_match.islower():
            return replacement_target.lower()
        return replacement_target

    def _swap_text(self, text: str) -> Tuple[str, bool]:
        """Swaps entity1 and entity2 simultaneously with case preservation."""
        if self.entity1 == self.entity2:
            return text, False

        pattern1 = re.compile(rf"\b{re.escape(self.entity1)}\b", re.IGNORECASE)
        pattern2 = re.compile(rf"\b{re.escape(self.entity2)}\b", re.IGNORECASE)

        m1 = list(pattern1.finditer(text))
        m2 = list(pattern2.finditer(text))

        if not m1 and not m2:
            return text, False

        # Simultaneous replacement
        placeholder1 = "___ENT_SWAP_1___"
        placeholder2 = "___ENT_SWAP_2___"

        cased_repl1 = self._preserve_case(m1[0].group(0), self.entity2) if m1 else self.entity2
        cased_repl2 = self._preserve_case(m2[0].group(0), self.entity1) if m2 else self.entity1

        s = pattern1.sub(placeholder1, text)
        s = pattern2.sub(placeholder2, s)

        s = s.replace(placeholder1, cased_repl1)
        s = s.replace(placeholder2, cased_repl2)

        return s, True

    def apply(self, premise: str, hypothesis: str) -> Tuple[str, str]:
        p_new, _ = self._swap_text(premise)
        h_new, _ = self._swap_text(hypothesis)
        return p_new, h_new

    def invert(self, premise: str, hypothesis: str) -> Tuple[str, str]:
        return self.apply(premise, hypothesis)

    def is_active_on_pair(self, premise: str, hypothesis: str) -> bool:
        """Returns True if at least one entity match is present in premise or hypothesis."""
        combined = premise + " " + hypothesis
        p1 = re.compile(rf"\b{re.escape(self.entity1)}\b", re.IGNORECASE)
        p2 = re.compile(rf"\b{re.escape(self.entity2)}\b", re.IGNORECASE)
        return bool(p1.search(combined) or p2.search(combined))
