"""Unit tests for Phase E0 finite ambiguity laboratory worlds."""

import numpy as np
import pytest

from research.holonomy.geometry.simplex_bundle import HELMERT_V3, ilr_inverse, ilr_transform
from research.holonomy.worlds.formulas import FormalState
from research.holonomy.worlds.generative_world import CurvedWorld, FlatWorld


def test_ilr_simplex_bijection():
    p = np.array([0.6, 0.3, 0.1])
    z = ilr_transform(p)
    assert z.shape == (2,)

    p_recon = ilr_inverse(z)
    assert p_recon.shape == (3,)
    assert np.allclose(p, p_recon, atol=1e-6)


def test_flat_world_distribution_constancy():
    flat_world = FlatWorld()
    s1 = FormalState("s1", "ALL_P_ARE_Q", "EXISTS_P_Q", "P", "Q")
    s2 = s1.swap_predicates()

    p1 = flat_world.generate_distribution(s1, flat_world.get_weights(s1))
    p2 = flat_world.generate_distribution(s2, flat_world.get_weights(s2))

    assert np.allclose(p1, p2, atol=1e-5)


def test_curved_world_weight_modulation():
    curved_world = CurvedWorld(rotation_angle=np.pi / 4)
    s1 = FormalState("s1", "ALL_P_ARE_Q", "EXISTS_P_Q", "P", "Q")
    s2 = s1.swap_predicates()

    w1 = curved_world.get_weights(s1)
    w2 = curved_world.get_weights(s2)

    assert not np.allclose(w1, w2)
