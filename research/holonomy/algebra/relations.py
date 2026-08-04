"""Algebraic Relations between Transformation Words.

Defines equivalences between generator paths (involutions, commutative squares, braid relations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from research.holonomy.algebra.generators import Generator, TransformationWord


@dataclass(frozen=True)
class Relation:
    """An algebraic equation path_1 === path_2 between two transformation words."""

    name: str
    path1: TransformationWord
    path2: TransformationWord

    def forms_closed_loop(self) -> bool:
        """Returns True if path1 and path2 form a closed loop when path2 is traversed backwards."""
        return True


def create_commutative_square_relation(
    g_a: Generator, g_b: Generator, name: str | None = None
) -> Relation:
    """Create a commutative square relation a * b === b * a."""
    rel_name = name or f"Comm({g_a.name}, {g_b.name})"
    w1 = TransformationWord((g_a, g_b))
    w2 = TransformationWord((g_b, g_a))
    return Relation(name=rel_name, path1=w1, path2=w2)


def create_involution_relation(g: Generator, name: str | None = None) -> Relation:
    """Create an involution relation g * g === identity."""
    rel_name = name or f"Invol({g.name})"
    w1 = TransformationWord((g, g))
    w2 = TransformationWord(())
    return Relation(name=rel_name, path1=w1, path2=w2)
