"""Unit tests for Phase E0.6 formula AST worlds and logic interpreters."""

import numpy as np
import pytest

from research.holonomy.algebra.formulas import And, Atom, FormalState, Implies, QuantifiedFormula
from research.holonomy.geometry.simplex_bundle import ilr_inverse, ilr_transform
from research.holonomy.worlds.generative_world import CurvedWorld, FlatWorld
from research.holonomy.worlds.interpreters import existential_import_eval, strict_fo_eval


def test_strict_fo_vs_existential_import_semantics():
    # Premise: All P are Q -> \forall x (P(x) -> Q(x))
    premise = QuantifiedFormula("FORALL", "x", Implies(Atom("P", ("x",)), Atom("Q", ("x",))))
    # Hypothesis: Some P is Q -> \exists x (P(x) \land Q(x))
    hypothesis = QuantifiedFormula("EXISTS", "x", And(Atom("P", ("x",)), Atom("Q", ("x",))))
    s0 = FormalState(premise=premise, hypothesis=hypothesis)

    # Strict FO yields Neutral (1), Existential Import yields Entailment (0)
    assert strict_fo_eval(s0) == 1
    assert existential_import_eval(s0) == 0


def test_curved_world_weight_modulation():
    curved_world = CurvedWorld(rotation_angle=np.pi / 4)
    premise = QuantifiedFormula("FORALL", "x", Implies(Atom("P", ("x",)), Atom("Q", ("x",))))
    hypothesis = QuantifiedFormula("EXISTS", "x", And(Atom("P", ("x",)), Atom("Q", ("x",))))
    s1 = FormalState(premise=premise, hypothesis=hypothesis)
    s2 = s1.swap_predicates()

    w1 = curved_world.get_weights(s1)
    w2 = curved_world.get_weights(s2)

    assert not np.allclose(w1, w2)
