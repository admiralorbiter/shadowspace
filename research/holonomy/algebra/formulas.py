"""Canonical First-Order Logic Formula AST and Predicate Permutations for Phase E0.6.

Implements full FOL AST (Atom, Not, And, Or, Implies, QuantifiedFormula) and recursive
PredicatePermutation mapping P(x) <-> Q(x) commuting with all logical operators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Tuple


class Formula(ABC):
    """Abstract Base Class for First-Order Logic Formulas."""

    @abstractmethod
    def canonical_id(self) -> str:
        """Returns unique canonical string representation of the formula."""
        pass

    @abstractmethod
    def simplify(self) -> Formula:
        """Applies canonical simplification rules."""
        pass

    @abstractmethod
    def permute_predicates(self, mapping: Mapping[str, str]) -> Formula:
        """Recursively permutes predicate symbols according to mapping."""
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

    def permute_predicates(self, mapping: Mapping[str, str]) -> Formula:
        new_pred = mapping.get(self.predicate, self.predicate)
        return Atom(predicate=new_pred, arguments=self.arguments)


@dataclass(frozen=True)
class Not(Formula):
    """Negation ~phi."""

    operand: Formula

    def canonical_id(self) -> str:
        simp = self.operand.simplify()
        return f"NOT({simp.canonical_id()})"

    def simplify(self) -> Formula:
        simp_op = self.operand.simplify()
        if isinstance(simp_op, Not):
            return simp_op.operand.simplify()
        return Not(simp_op)

    def permute_predicates(self, mapping: Mapping[str, str]) -> Formula:
        return Not(self.operand.permute_predicates(mapping)).simplify()


@dataclass(frozen=True)
class And(Formula):
    """Conjunction phi ^ psi."""

    left: Formula
    right: Formula

    def canonical_id(self) -> str:
        s_left = self.left.simplify().canonical_id()
        s_right = self.right.simplify().canonical_id()
        return f"({s_left} AND {s_right})"

    def simplify(self) -> Formula:
        return And(self.left.simplify(), self.right.simplify())

    def permute_predicates(self, mapping: Mapping[str, str]) -> Formula:
        return And(
            self.left.permute_predicates(mapping),
            self.right.permute_predicates(mapping),
        ).simplify()


@dataclass(frozen=True)
class Or(Formula):
    """Disjunction phi v psi."""

    left: Formula
    right: Formula

    def canonical_id(self) -> str:
        s_left = self.left.simplify().canonical_id()
        s_right = self.right.simplify().canonical_id()
        return f"({s_left} OR {s_right})"

    def simplify(self) -> Formula:
        return Or(self.left.simplify(), self.right.simplify())

    def permute_predicates(self, mapping: Mapping[str, str]) -> Formula:
        return Or(
            self.left.permute_predicates(mapping),
            self.right.permute_predicates(mapping),
        ).simplify()


@dataclass(frozen=True)
class Implies(Formula):
    """Implication phi -> psi."""

    left: Formula
    right: Formula

    def canonical_id(self) -> str:
        s_left = self.left.simplify().canonical_id()
        s_right = self.right.simplify().canonical_id()
        return f"({s_left} IMPLIES {s_right})"

    def simplify(self) -> Formula:
        return Implies(self.left.simplify(), self.right.simplify())

    def permute_predicates(self, mapping: Mapping[str, str]) -> Formula:
        return Implies(
            self.left.permute_predicates(mapping),
            self.right.permute_predicates(mapping),
        ).simplify()


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

    def permute_predicates(self, mapping: Mapping[str, str]) -> Formula:
        return QuantifiedFormula(
            self.quantifier, self.variable, self.body.permute_predicates(mapping)
        ).simplify()


@dataclass(frozen=True)
class Swap(Formula):
    """Predicate Swap wrapper for P <-> Q."""

    operand: Formula

    def canonical_id(self) -> str:
        simp = self.operand.simplify()
        return f"SWAP({simp.canonical_id()})"

    def simplify(self) -> Formula:
        simp_op = self.operand.simplify()
        if isinstance(simp_op, Swap):
            return simp_op.operand.simplify()
        # Evaluate recursive predicate permutation P <-> Q
        mapping = {"P": "Q", "Q": "P"}
        return simp_op.permute_predicates(mapping)

    def permute_predicates(self, mapping: Mapping[str, str]) -> Formula:
        return self.simplify().permute_predicates(mapping)


@dataclass(frozen=True)
class PredicatePermutation:
    """Explicit recursive predicate permutation mapping."""

    mapping: Mapping[str, str]

    def apply(self, formula: Formula) -> Formula:
        return formula.permute_predicates(self.mapping)


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
        """Applies predicate swap transformation P <-> Q."""
        mapping = {"P": "Q", "Q": "P"}
        perm = PredicatePermutation(mapping)
        new_p = perm.apply(self.premise)
        new_h = perm.apply(self.hypothesis)
        return FormalState(premise=new_p, hypothesis=new_h, context_features=self.context_features)

    def negate_hypothesis(self) -> FormalState:
        """Applies hypothesis negation transformation."""
        new_h = Not(self.hypothesis).simplify()
        return FormalState(premise=self.premise, hypothesis=new_h, context_features=self.context_features)
