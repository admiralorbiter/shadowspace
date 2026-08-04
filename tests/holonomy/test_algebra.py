"""Unit tests for holonomy transformation algebra and Cayley complex."""

import pytest
from research.holonomy.algebra.cayley_complex import CayleyComplex, ClosedLoop, Edge
from research.holonomy.algebra.generators import Generator, TransformationWord
from research.holonomy.algebra.relations import create_commutative_square_relation, create_involution_relation


def test_generator_composition():
    g1 = Generator("add1", action=lambda x: x + 1)
    g2 = Generator("mul2", action=lambda x: x * 2)

    word = TransformationWord((g1, g2))
    assert len(word) == 2
    # g2(g1(5)) = (5 + 1) * 2 = 12
    assert word(5) == 12


def test_commutative_square_relation():
    g_a = Generator("a", action=lambda s: f"a({s})")
    g_b = Generator("b", action=lambda s: f"b({s})")

    rel = create_commutative_square_relation(g_a, g_b)
    assert rel.name == "Comm(a, b)"
    assert len(rel.path1) == 2
    assert len(rel.path2) == 2


def test_cayley_complex_building():
    cc = CayleyComplex()
    g_swap = Generator("swap", action=lambda x: f"swap({x})")
    cc.add_vertex("x0", "item0")
    cc.add_edge("x0", "x1", g_swap)

    assert len(cc.vertices) == 1
    assert len(cc.edges) == 1
    assert cc.edges[0].source_id == "x0"
    assert cc.edges[0].target_id == "x1"
