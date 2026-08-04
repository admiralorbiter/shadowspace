"""Finite First-Order Model Structure & Recursive Formula Evaluator for Phase E0.6.

Implements model-theoretic satisfaction M, rho |= phi for arbitrary finite domain models M.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple

from research.holonomy.algebra.formulas import And, Atom, Formula, Implies, Not, Or, QuantifiedFormula, Swap


@dataclass
class Entity:
    """An individual constant in the finite domain."""

    name: str


@dataclass
class FiniteModel:
    """A finite first-order model M = (D, I)."""

    domain: Set[str] = field(default_factory=set)
    unary_predicates: Dict[str, Set[str]] = field(default_factory=dict)
    binary_predicates: Dict[str, Set[Tuple[str, str]]] = field(default_factory=dict)

    def evaluate_unary(self, pred: str, entity: str) -> bool:
        return entity in self.unary_predicates.get(pred, set())

    def evaluate_binary(self, pred: str, entity1: str, entity2: str) -> bool:
        return (entity1, entity2) in self.binary_predicates.get(pred, set())


def evaluate_formula(
    formula: Formula, model: FiniteModel, assignment: Dict[str, str] | None = None
) -> bool:
    """Evaluates recursive satisfaction M, rho |= phi."""
    rho = assignment or {}
    simp = formula.simplify()

    if isinstance(simp, Atom):
        eval_args = tuple(rho.get(arg, arg) for arg in simp.arguments)
        if len(eval_args) == 1:
            return model.evaluate_unary(simp.predicate, eval_args[0])
        elif len(eval_args) == 2:
            return model.evaluate_binary(simp.predicate, eval_args[0], eval_args[1])
        return False

    elif isinstance(simp, Not):
        return not evaluate_formula(simp.operand, model, rho)

    elif isinstance(simp, And):
        return evaluate_formula(simp.left, model, rho) and evaluate_formula(simp.right, model, rho)

    elif isinstance(simp, Or):
        return evaluate_formula(simp.left, model, rho) or evaluate_formula(simp.right, model, rho)

    elif isinstance(simp, Implies):
        return (not evaluate_formula(simp.left, model, rho)) or evaluate_formula(simp.right, model, rho)

    elif isinstance(simp, QuantifiedFormula):
        var = simp.variable
        if simp.quantifier == "FORALL":
            return all(
                evaluate_formula(simp.body, model, {**rho, var: e}) for e in model.domain
            )
        elif simp.quantifier == "EXISTS":
            return any(
                evaluate_formula(simp.body, model, {**rho, var: e}) for e in model.domain
            )

    return False


def evaluate_nli_label_over_models(
    premise: Formula, hypothesis: Formula, models: Sequence[FiniteModel]
) -> int:
    """Evaluates NLI label {0: E, 1: N, 2: C} across a suite of finite models.

    - Entailment (0): P |= H (In every model where P holds, H holds)
    - Contradiction (2): P |= ~H (In every model where P holds, H fails)
    - Neutral (1): Otherwise
    """
    valid_p_count = 0
    h_holds_count = 0
    not_h_holds_count = 0

    for model in models:
        if evaluate_formula(premise, model):
            valid_p_count += 1
            if evaluate_formula(hypothesis, model):
                h_holds_count += 1
            if evaluate_formula(Not(hypothesis), model):
                not_h_holds_count += 1

    if valid_p_count == 0:
        return 1  # Neutral

    if h_holds_count == valid_p_count:
        return 0  # Entailment
    if not_h_holds_count == valid_p_count:
        return 2  # Contradiction
    return 1     # Neutral
