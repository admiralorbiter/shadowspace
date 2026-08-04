"""Latent Logical Interpreters for Phase E0.7.

Model-theoretic evaluation rules over the 64 finite First-Order structures:
- StrictFO: Evaluates premise and hypothesis over all 64 finite models (including 8 models where P is empty).
  Premise ALL P ARE Q: \forall x (P(x) -> Q(x))
  Hypothesis SOME P IS Q: \exists x (P(x) \land Q(x))
  StrictFO returns Neutral (1) because P is empty in 8 models.
- ExistentialImport: Evaluates premise + \exists x P(x) over 56 non-empty P models.
  With existential import \exists x P(x), ALL P ARE Q entails SOME P IS Q, returning Entailment (0).
"""

from __future__ import annotations

from typing import Callable, List
from research.holonomy.algebra.formulas import And, Atom, FormalState, Formula, Implies, Not, QuantifiedFormula
from research.holonomy.worlds.finite_models import ALL_64_MODELS, FiniteModel, evaluate_nli_label_over_models


def get_non_empty_p_models() -> List[FiniteModel]:
    """Filters ALL_64_MODELS to the 56 models where predicate P is non-empty."""
    return [m for m in ALL_64_MODELS if len(m.unary_predicates.get("P", set())) > 0]


NON_EMPTY_P_MODELS = get_non_empty_p_models()


class LatentInterpreter:
    """A deterministic model-theoretic interpreter rule."""

    def __init__(self, name: str, eval_fn: Callable[[FormalState], int]) -> None:
        self.name = name
        self.eval_fn = eval_fn

    def __call__(self, state: FormalState) -> int:
        return self.eval_fn(state)


def strict_fo_eval(state: FormalState) -> int:
    """Strict First-Order Logic evaluation over all 64 models."""
    return evaluate_nli_label_over_models(state.premise, state.hypothesis, ALL_64_MODELS)


def existential_import_eval(state: FormalState) -> int:
    """Existential Import evaluation over 56 non-empty P models."""
    return evaluate_nli_label_over_models(state.premise, state.hypothesis, NON_EMPTY_P_MODELS)


def scalar_implicature_eval(state: FormalState) -> int:
    """Scalar Implicature evaluation (Some P is Q implies Not All P are Q)."""
    h_id = state.hypothesis.simplify().canonical_id()
    if "FORALL" in h_id:
        return 2  # Contradiction under pragmatic implicature
    return evaluate_nli_label_over_models(state.premise, state.hypothesis, ALL_64_MODELS)


STANDARD_INTERPRETERS = [
    LatentInterpreter("StrictFO", strict_fo_eval),
    LatentInterpreter("ExistentialImport", existential_import_eval),
    LatentInterpreter("ScalarImplicature", scalar_implicature_eval),
]
