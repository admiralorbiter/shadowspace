r"""Latent Logical Interpreters for Phase E0.6.

Model-theoretic evaluation rules over finite First-Order structures:
- StrictFO: Evaluates premise and hypothesis over finite models including empty-predicate models.
  Premise ALL P ARE Q: \forall x (P(x) -> Q(x))
  Hypothesis SOME P IS Q: \exists x (P(x) \land Q(x))
  StrictFO returns Neutral (1) because P may be empty in some model.
- ExistentialImport: Evaluates premise + \exists x P(x) over finite models.
  With existential import \exists x P(x), ALL P ARE Q entails SOME P IS Q, returning Entailment (0).
"""

from __future__ import annotations

from typing import Callable, List
from research.holonomy.algebra.formulas import And, Atom, FormalState, Formula, Implies, Not, QuantifiedFormula
from research.holonomy.worlds.finite_models import FiniteModel, evaluate_nli_label_over_models


def generate_canonical_finite_models() -> List[FiniteModel]:
    """Generates canonical finite models over domain D = {e1, e2, e3}."""
    D = {"e1", "e2", "e3"}
    m1 = FiniteModel(domain=D, unary_predicates={"P": {"e1"}, "Q": {"e1"}})        # P non-empty, P & Q overlap
    m2 = FiniteModel(domain=D, unary_predicates={"P": set(), "Q": {"e1", "e2"}})   # P empty
    m3 = FiniteModel(domain=D, unary_predicates={"P": {"e1", "e2"}, "Q": set()})   # Q empty (Contradiction if P non-empty)
    m4 = FiniteModel(domain=D, unary_predicates={"P": {"e1"}, "Q": {"e2"}})        # Disjoint P and Q
    return [m1, m2, m3, m4]


CANONICAL_MODELS = generate_canonical_finite_models()


class LatentInterpreter:
    """A deterministic model-theoretic interpreter rule."""

    def __init__(self, name: str, eval_fn: Callable[[FormalState], int]) -> None:
        self.name = name
        self.eval_fn = eval_fn

    def __call__(self, state: FormalState) -> int:
        return self.eval_fn(state)


def strict_fo_eval(state: FormalState) -> int:
    """Strict First-Order Logic evaluation over finite models."""
    return evaluate_nli_label_over_models(state.premise, state.hypothesis, CANONICAL_MODELS)


def existential_import_eval(state: FormalState) -> int:
    """Existential Import evaluation (restricting to models where P is non-empty)."""
    p_non_empty_models = [m for m in CANONICAL_MODELS if len(m.unary_predicates.get("P", set())) > 0]
    return evaluate_nli_label_over_models(state.premise, state.hypothesis, p_non_empty_models)


def scalar_implicature_eval(state: FormalState) -> int:
    """Scalar Implicature evaluation (Some P is Q implies Not All P are Q)."""
    h_id = state.hypothesis.simplify().canonical_id()
    if "FORALL" in h_id:
        return 2  # Contradiction under pragmatic implicature
    return evaluate_nli_label_over_models(state.premise, state.hypothesis, CANONICAL_MODELS)


STANDARD_INTERPRETERS = [
    LatentInterpreter("StrictFO", strict_fo_eval),
    LatentInterpreter("ExistentialImport", existential_import_eval),
    LatentInterpreter("ScalarImplicature", scalar_implicature_eval),
]
