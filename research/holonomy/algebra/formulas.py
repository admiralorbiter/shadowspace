"""Canonical First-Order Logic Formula AST for Phase E0.5.

Implements canonical formula representation with exact simplification rules:
- Not(Not(phi)) -> phi
- Swap(Swap(phi)) -> phi
State identity and equality are strictly based on canonical semantic structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple


class Formula(ABC):
    """Abstract Base Class for First-Order Logic Formulas."""

    @abstractmethod
    def canonical_id(self) -> str:
        """Returns unique canonical string representation of the formula."""
        pass

    @abstractmethod
    def simplify(self) -> Formula:
        """Applies canonical simplification rules (e.g. double negation elimination)."""
        pass

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Formula):
            return False
        return self.simplify().canonical_id() == other.simplify().canonical_id()

    def __hash__(self) -> int:
        return hash(self.simplify().canonical_id())


@dataclass(frozen=True)
class Atom(Formula):
    """Atomic predicate P(x1, ..., xn)."""

    predicate: str
    arguments: Tuple[str, ...]

    def canonical_id(self) -> str:
        args_str = ", ".join(self.arguments)
        return f"{self.predicate}({args_str})"

    def simplify(self) -> Formula:
        return self


@dataclass(frozen=True)
class Not(Formula):
    """Negation ~phi."""

    operand: Formula

    def canonical_id(self) -> str:
        simp = self.operand.simplify()
        return f"NOT({simp.canonical_id()})"

    def simplify(self) -> Formula:
        simp_op = self.operand.simplify()
        # Rule: Not(Not(phi)) -> phi
        if isinstance(simp_op, Not):
            return simp_op.operand.simplify()
        return Not(simp_op)


@dataclass(frozen=True)
class Swap(Formula):
    """Predicate Swap operation (e.g., swap P and Q)."""

    operand: Formula

    def canonical_id(self) -> str:
        simp = self.operand.simplify()
        return f"SWAP({simp.canonical_id()})"

    def simplify(self) -> Formula:
        simp_op = self.operand.simplify()
        # Rule: Swap(Swap(phi)) -> phi
        if isinstance(simp_op, Swap):
            return simp_op.operand.simplify()
        return Swap(simp_op)


@dataclass(frozen=True)
class QuantifiedFormula(Formula):
    """Quantified formula (FORALL x phi or EXISTS x phi)."""

    quantifier: str  # "FORALL" or "EXISTS"
    variable: str
    body: Formula

    def canonical_id(self) -> str:
        simp_body = self.body.simplify()
        return f"{self.quantifier} {self.variable}.({simp_body.canonical_id()})"

    def simplify(self) -> Formula:
        return QuantifiedFormula(self.quantifier, self.variable, self.body.simplify())


@dataclass(frozen=True)
class FormalState:
    """Formal semantic state represented by premise and hypothesis canonical formulas."""

    premise: Formula
    hypothesis: Formula
    context_features: Tuple[float, ...] = ()

    @property
    def id(self) -> str:
        p_id = self.premise.simplify().canonical_id()
        h_id = self.hypothesis.simplify().canonical_id()
        return f"[{p_id} |= {h_id}]"

    def swap_predicates(self) -> FormalState:
        """Applies predicate swap transformation."""
        new_p = Swap(self.premise).simplify()
        new_h = Swap(self.hypothesis).simplify()
        return FormalState(premise=new_p, hypothesis=new_h, context_features=self.context_features)

    def negate_hypothesis(self) -> FormalState:
        """Applies hypothesis negation transformation."""
        new_h = Not(self.hypothesis).simplify()
        return FormalState(premise=self.premise, hypothesis=new_h, context_features=self.context_features)
