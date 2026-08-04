"""Unit tests for Phase E0.5 canonical formula AST and Cayley complex."""

import pytest
from research.holonomy.algebra.cayley_complex import CayleyComplex, ClosedLoop, Edge
from research.holonomy.algebra.formulas import Atom, FormalState, Not, QuantifiedFormula, Swap
from research.holonomy.algebra.generators import Generator, TransformationWord
from research.holonomy.algebra.relations import create_commutative_square_relation


def test_canonical_formula_ast_simplification():
    atom = Atom("P", ("x",))
    not_atom = Not(atom)
    double_not = Not(not_atom)

    assert double_not.simplify() == atom
    assert double_not.simplify().canonical_id() == "P(x)"


def test_swap_simplification():
    atom = Atom("P", ("x",))
    swap_atom = Swap(atom)
    double_swap = Swap(swap_atom)

    assert double_swap.simplify() == atom
    assert double_swap.simplify().canonical_id() == "P(x)"


def test_formal_state_exact_involutions():
    premise = QuantifiedFormula("FORALL", "x", Atom("P", ("x",)))
    hypothesis = QuantifiedFormula("EXISTS", "x", Atom("Q", ("x",)))
    s0 = FormalState(premise=premise, hypothesis=hypothesis)

    s_swap_swap = s0.swap_predicates().swap_predicates()
    assert s_swap_swap == s0
    assert s_swap_swap.id == s0.id

    s_neg_neg = s0.negate_hypothesis().negate_hypothesis()
    assert s_neg_neg == s0
    assert s_neg_neg.id == s0.id
