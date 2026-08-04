"""Latent Interpreters for Phase E0 Simulator.

Multiple deterministic interpretation rules (e.g. Strict FO, Existential Import,
Scalar Implicature, Prior Bias) mapping a formal NLI state to label {0: E, 1: N, 2: C}.
"""

from __future__ import annotations

from typing import Callable
from research.holonomy.worlds.formulas import FormalState


class LatentInterpreter:
    """A deterministic interpreter rule."""

    def __init__(self, name: str, eval_fn: Callable[[FormalState], int]) -> None:
        self.name = name
        self.eval_fn = eval_fn

    def __call__(self, state: FormalState) -> int:
        return self.eval_fn(state)


# Standard interpreter definitions:
def strict_fo_eval(state: FormalState) -> int:
    """Strict First-Order interpretation."""
    if state.premise_type == "ALL_P_ARE_Q" and state.hypothesis_type == "EXISTS_P_Q":
        return 0  # Under strict FO with non-empty domain, entailment
    if state.hypothesis_type.startswith("NOT_"):
        return 2  # Contradiction
    return 1  # Neutral


def existential_import_eval(state: FormalState) -> int:
    """Existential import assumption interpretation."""
    if state.premise_type == "ALL_P_ARE_Q" and state.hypothesis_type == "EXISTS_P_Q":
        return 0  # Always entailment under existential import
    if state.hypothesis_type.startswith("NOT_"):
        return 2
    return 1


def scalar_implicature_eval(state: FormalState) -> int:
    """Scalar implicature interpretation (SOME implies NOT ALL)."""
    if state.premise_type == "SOME_P_ARE_Q" and state.hypothesis_type == "FORALL_P_Q":
        return 2  # Contradiction under pragmatic implicature
    if state.hypothesis_type.startswith("NOT_"):
        return 2
    return 1


def prior_biased_eval(state: FormalState) -> int:
    """Prior-driven interpretation (biased towards neutral unless explicit contradiction)."""
    if state.hypothesis_type.startswith("NOT_"):
        return 2
    return 1


STANDARD_INTERPRETERS = [
    LatentInterpreter("StrictFO", strict_fo_eval),
    LatentInterpreter("ExistentialImport", existential_import_eval),
    LatentInterpreter("ScalarImplicature", scalar_implicature_eval),
    LatentInterpreter("PriorBiased", prior_biased_eval),
]
