"""Experiment E000: Flat World Holonomy Verification (Phase E0.5).

Validates Phase E0 null hypothesis: in a Flat World with canonical AST formulas,
parallel transport around a closed semantic loop gamma yields exact zero curvature (H_gamma = I_2).
"""

from __future__ import annotations

import numpy as np

from research.holonomy.algebra.formulas import Atom, FormalState, QuantifiedFormula
from research.holonomy.algebra.generators import Generator
from research.holonomy.geometry.connection import ConnectionEstimator
from research.holonomy.geometry.holonomy import evaluate_holonomy
from research.holonomy.geometry.parallel_transport import PathTransport
from research.holonomy.geometry.simplex_bundle import ilr_transform
from research.holonomy.worlds.generative_world import FlatWorld


def run_e000_flat_world_experiment() -> bool:
    """Runs E000 experiment and returns True if flat world yields zero holonomy."""
    flat_world = FlatWorld()

    g_swap = Generator("swap", action=lambda s: s.swap_predicates())
    g_neg = Generator("neg", action=lambda s: s.negate_hypothesis())

    premise = QuantifiedFormula("FORALL", "x", Atom("P", ("x",)))
    hypothesis = QuantifiedFormula("EXISTS", "x", Atom("Q", ("x",)))
    x0 = FormalState(premise=premise, hypothesis=hypothesis)

    x1 = g_swap(x0)
    x2 = g_neg(x1)
    x3 = g_neg(x0)

    np.random.seed(42)
    sample_size = 100
    noise_level = 0.05

    p_x0 = flat_world.generate_distribution(x0, flat_world.get_weights(x0))
    z_x0 = ilr_transform(p_x0)

    deltas = np.random.normal(0, noise_level, (sample_size, 2))

    orbit_x0 = z_x0 + deltas
    orbit_x1 = z_x0 + deltas

    reflection_neg = np.array([1.0, -1.0])
    orbit_x2 = z_x0 + deltas * reflection_neg
    orbit_x3 = z_x0 + deltas * reflection_neg

    estimator = ConnectionEstimator()
    T_01 = estimator.estimate_linear_transport("swap", "x0", "x1", orbit_x0, orbit_x1)
    T_12 = estimator.estimate_linear_transport("neg", "x1", "x2", orbit_x1, orbit_x2)
    T_23 = estimator.estimate_linear_transport("swap_inv", "x2", "x3", orbit_x2, orbit_x3)
    T_30 = estimator.estimate_linear_transport("neg_inv", "x3", "x0", orbit_x3, orbit_x0)

    path_transport = PathTransport([T_01, T_12, T_23, T_30])
    res = evaluate_holonomy("CommutativeSquare_Flat", path_transport)

    is_flat = np.allclose(res.matrix, np.eye(2), atol=1e-3) and np.isclose(res.rotation_angle, 0.0, atol=1e-3)
    return bool(is_flat)


if __name__ == "__main__":
    success = run_e000_flat_world_experiment()
    print(f"E000 Flat World Experiment Passed: {success}")
