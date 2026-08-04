"""Base class for natural language semantic transformation generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


class BaseTransform(ABC):
    """Abstract Base Class for Natural Language Semantic Transformations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the transformation generator."""
        pass

    @abstractmethod
    def apply(self, premise: str, hypothesis: str) -> Tuple[str, str]:
        """Applies transformation: (premise, hypothesis) -> (new_premise, new_hypothesis)."""
        pass

    @abstractmethod
    def invert(self, premise: str, hypothesis: str) -> Tuple[str, str]:
        """Inverts transformation: (new_premise, new_hypothesis) -> (premise, hypothesis)."""
        pass
