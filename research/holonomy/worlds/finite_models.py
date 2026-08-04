"""Finite First-Order Model Structure for Phase E0 Simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple


@dataclass
class Entity:
    """An individual constant in the finite domain."""

    name: str


@dataclass
class FiniteModel:
    """A finite first-order model M = (D, I).

    Attributes:
        domain: Set of entity names (e.g. {'e1', 'e2', 'e3'})
        unary_predicates: Map from predicate name to subset of domain where it holds.
        binary_predicates: Map from predicate name to set of (e_i, e_j) pairs where it holds.
    """

    domain: Set[str] = field(default_factory=set)
    unary_predicates: Dict[str, Set[str]] = field(default_factory=dict)
    binary_predicates: Dict[str, Set[Tuple[str, str]]] = field(default_factory=dict)

    def evaluate_unary(self, pred: str, entity: str) -> bool:
        return entity in self.unary_predicates.get(pred, set())

    def evaluate_binary(self, pred: str, entity1: str, entity2: str) -> bool:
        return (entity1, entity2) in self.binary_predicates.get(pred, set())
