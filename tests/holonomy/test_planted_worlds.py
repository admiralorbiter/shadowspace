"""Unit tests for Phase E0.7.1 formula AST worlds, non-commuting dynamics, and logic interpreters."""

import numpy as np
import pytest

from research.holonomy.algebra.formulas import And, Atom, FormalState, Implies, QuantifiedFormula
from research.holonomy.geometry.simplex_bundle import ilr_transform
from research.holonomy.worlds.generative_world import CurvedWorld, FlatWorld
from research.holonomy.worlds.interpreters import existential_import_eval, strict_fo_eval


def test_strict_fo_vs_existential_import_semantics():
    premise = QuantifiedFormula("FORALL", "x", Implies(Atom("P", ("x",)), Atom("Q", ("x",))))
    hypothesis = QuantifiedFormula("EXISTS", "x", And(Atom("P", ("x",)), Atom("Q", ("x",))))
    s0 = FormalState(premise=premise, hypothesis=hypothesis)

    assert strict_fo_eval(s0) == 1
    assert existential_import_eval(s0) == 0


def test_curved_world_non_commuting_weight_dynamics():
    curved_world = CurvedWorld(rotation_angle=np.pi / 4)
    u_base = ilr_transform(curved_world.base_weights)

    # u_a_b = K_b (K_a u + c_a) + c_b
    u_a = curved_world.transform_weights_along_edge("swap", u_base)
    u_a_b = curved_world.transform_weights_along_edge("neg", u_a)

    # u_b_a = K_a (K_b u + c_b) + c_a
    u_b = curved_world.transform_weights_along_edge("neg", u_base)
    u_b_a = curved_world.transform_weights_along_edge("swap", u_b)

    # Non-commutativity K_a K_b != K_b K_a
    assert not np.allclose(u_a_b, u_b_a)
