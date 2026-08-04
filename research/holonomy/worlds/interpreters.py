"""Latent Logical Interpreters for Phase E0.5.

Model-theoretic evaluation rules over finite First-Order structures:
- StrictFO: All P are Q does NOT entail Some P is Q (without non-empty P), returning Neutral (1).
- ExistentialImport: All P are Q + Exists x P(x) ENTAILS Some P is Q, returning Entailment (0).
"""

from __future__ import annotations

from typing import Callable
from research.holonomy.algebra.formulas import Atom, FormalState, Not, QuantifiedFormula, Swap


class LatentInterpreter:
    """A deterministic model-theoretic interpreter rule."""

    def __init__(self, name: str, eval_fn: Callable[[FormalState], int]) -> None:
        self.name = name
        self.eval_fn = eval_fn

    def __call__(self, state: FormalState) -> int:
        return self.eval_fn(state)


def strict_fo_eval(state: FormalState) -> int:
    """Strict First-Order Logic evaluation (empty domain/predicate possibility)."""
    p_id = state.premise.simplify().canonical_id()
    h_id = state.hypothesis.simplify().canonical_id()

    if "FORALL" in p_id and "EXISTS" in h_id:
        return 1  # Strict FO: All P are Q does not imply Some P is Q without non-empty P
    if "NOT" in h_id:
        return 2  # Contradiction
    return 1  # Neutral


def existential_import_eval(state: FormalState) -> int:
    """Existential Import evaluation (assuming non-empty domain & predicates)."""
    p_id = state.premise.simplify().canonical_id()
    h_id = state.hypothesis.simplify().canonical_id()

    if "FORALL" in p_id and "EXISTS" in h_id:
        return 0  # Existential import: All P are Q -> Some P is Q
    if "NOT" in h_id:
        return 2
    return 1


def scalar_implicature_eval(state: FormalState) -> int:
    """Scalar implicature evaluation (Some P are Q implies Not All P are Q)."""
    p_id = state.premise.simplify().canonical_id()
    h_id = state.hypothesis.simplify().canonical_id()

    if "EXISTS" in p_id and "FORALL" in h_id:
        return 2  # Contradiction under pragmatic implicature
    if "NOT" in h_id:
        return 2
    return 1


STANDARD_INTERPRETERS = [
    LatentInterpreter("StrictFO", strict_fo_eval),
    LatentInterpreter("ExistentialImport", existential_import_eval),
    LatentInterpreter("ScalarImplicature", scalar_implicature_eval),
]
