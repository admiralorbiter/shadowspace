"""Semantic Generators and Transformation Algebra.

Defines elementary semantic transformation operations g: x -> gx and generator compositions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class Generator:
    """An elementary semantic transformation generator g."""

    name: str
    action: Callable[[Any], Any]
    inverse_name: str | None = None

    def __call__(self, state: Any) -> Any:
        return self.action(state)

    def __repr__(self) -> str:
        return f"Gen({self.name})"


@dataclass(frozen=True)
class TransformationWord:
    """A sequence of semantic generators applied right-to-left or left-to-right.

    Word w = (g_1, g_2, ..., g_k) applied to x yields g_k(...(g_2(g_1(x)))).
    """

    generators: tuple[Generator, ...]

    def __call__(self, state: Any) -> Any:
        curr = state
        for g in self.generators:
            curr = g(curr)
        return curr

    def __add__(self, other: TransformationWord) -> TransformationWord:
        return TransformationWord(self.generators + other.generators)

    def __len__(self) -> int:
        return len(self.generators)

    def __repr__(self) -> str:
        if not self.generators:
            return "Word(id)"
        return " -> ".join(g.name for g in self.generators)


IDENTITY_WORD = TransformationWord(())
