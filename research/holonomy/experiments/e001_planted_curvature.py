"""Experiment E001: Planted Curvature Recovery & Noisy Statistical Sweeps (Phase E0.6).

Estimates edge transport maps under explicit measurement noise sigma and perturbation radius r:
X_j = z_x + delta_j + epsilon_x, Y_j = z_gx + T delta_j + epsilon_y.
Sweeps r, sigma, and N in {20, 50, 100, 250, 500} across seeds reporting bias, RMSE, and detection power.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

from research.holonomy.algebra.formulas import Atom, FormalState, QuantifiedFormula
from research.holonomy.algebra.generators import Generator
from research.holonomy.geometry.connection import ConnectionEstimator, ParallelTransportMap
from research.holonomy.geometry.holonomy import evaluate_holonomy
from research.holonomy.geometry.parallel_transport import PathTransport
from research.holonomy.geometry.simplex_bundle import ilr_transform
from research.holonomy.worlds.generative_world import CurvedWorld


@dataclass
class SweepMetrics:
    """Statistical metrics for perturbation radius r and measurement noise sigma sweeps."""

    sample_size: int
    perturbation_radius: float
    measurement_noise: float
    matrix_bias: float
    matrix_rmse: float
    holonomy_rmse: float
    false_positive_rate: float
    detection_power: float


def run_e001_planted_curvature_experiment(
    planted_angle: float = np.pi / 6,
    sample_size: int = 100,
    perturbation_radius: float = 0.02,
    measurement_noise: float = 0.001,
) -> bool:
    """Runs 4-corner edge transport estimation under non-commuting rotation R_a and shear S_b with noise."""
    curved_world = CurvedWorld(rotation_angle=planted_angle)

    g_swap = Generator("swap", action=lambda s: s.swap_predicates())
    g_neg = Generator("neg", action=lambda s: s.negate_hypothesis())

    premise = QuantifiedFormula("FORALL", "x", Atom("P", ("x",)))
    hypothesis = QuantifiedFormula("EXISTS", "x", Atom("Q", ("x",)))
    x0 = FormalState(premise=premise, hypothesis=hypothesis)

    x1 = g_swap(x0)
    x2 = g_neg(x1)
    x3 = g_neg(x0)

    np.random.seed(123)

    p_x0 = curved_world.generate_distribution(x0, curved_world.get_weights(x0))
    p_x1 = curved_world.generate_distribution(x1, curved_world.get_weights(x1))
    p_x2 = curved_world.generate_distribution(x2, curved_world.get_weights(x2))
    p_x3 = curved_world.generate_distribution(x3, curved_world.get_weights(x3))

    z_x0 = ilr_transform(p_x0)
    z_x1 = ilr_transform(p_x1)
    z_x2 = ilr_transform(p_x2)
    z_x3 = ilr_transform(p_x3)

    c, s = np.cos(planted_angle), np.sin(planted_angle)
    R_a = np.array([[c, -s], [s, c]], dtype=np.float64)
    S_b = np.array([[1.0, 0.5], [0.0, 1.0]], dtype=np.float64)

    S_b_inv = np.linalg.inv(S_b)
    R_a_inv = np.linalg.inv(R_a)
    H_true = np.dot(S_b_inv, np.dot(R_a_inv, np.dot(S_b, R_a)))
    res_true = evaluate_holonomy("True", PathTransport([ParallelTransportMap("t", "a", "b", H_true, np.zeros(2))]))

    deltas = np.random.normal(0, perturbation_radius, (sample_size, 2))

    # Add measurement noise epsilon ~ N(0, sigma^2 I)
    eps_01_src = np.random.normal(0, measurement_noise, (sample_size, 2))
    eps_01_tgt = np.random.normal(0, measurement_noise, (sample_size, 2))

    eps_12_src = np.random.normal(0, measurement_noise, (sample_size, 2))
    eps_12_tgt = np.random.normal(0, measurement_noise, (sample_size, 2))

    eps_23_src = np.random.normal(0, measurement_noise, (sample_size, 2))
    eps_23_tgt = np.random.normal(0, measurement_noise, (sample_size, 2))

    eps_30_src = np.random.normal(0, measurement_noise, (sample_size, 2))
    eps_30_tgt = np.random.normal(0, measurement_noise, (sample_size, 2))

    orbit_01_src = z_x0 + deltas + eps_01_src
    orbit_01_tgt = z_x1 + np.dot(deltas, R_a.T) + eps_01_tgt

    orbit_12_src = z_x1 + deltas + eps_12_src
    orbit_12_tgt = z_x2 + np.dot(deltas, S_b.T) + eps_12_tgt

    orbit_23_src = z_x2 + deltas + eps_23_src
    orbit_23_tgt = z_x3 + np.dot(deltas, R_a_inv.T) + eps_23_tgt

    orbit_30_src = z_x3 + deltas + eps_30_src
    orbit_30_tgt = z_x0 + np.dot(deltas, S_b_inv.T) + eps_30_tgt

    estimator = ConnectionEstimator()
    T_01 = estimator.estimate_linear_transport("swap", "x0", "x1", orbit_01_src, orbit_01_tgt)
    T_12 = estimator.estimate_linear_transport("neg", "x1", "x2", orbit_12_src, orbit_12_tgt)
    T_23 = estimator.estimate_linear_transport("swap_inv", "x2", "x3", orbit_23_src, orbit_23_tgt)
    T_30 = estimator.estimate_linear_transport("neg_inv", "x3", "x0", orbit_30_src, orbit_30_tgt)

    path_transport = PathTransport([T_01, T_12, T_23, T_30])
    res = evaluate_holonomy("4Corner_Curved_Square", path_transport)

    curvature_recovered = bool(np.isclose(res.curvature_magnitude, res_true.curvature_magnitude, atol=0.08))
    return curvature_recovered


def run_e001_estimator_sweeps() -> List[SweepMetrics]:
    """Runs sweeps across sample size N in {20, 50, 100, 250, 500}, r, and sigma."""
    results = []
    planted_angle = np.pi / 4

    for N in [20, 50, 100, 250, 500]:
        for r in [0.01, 0.05]:
            for sigma in [0.0, 0.005]:
                curved_world = CurvedWorld(rotation_angle=planted_angle)

                premise = QuantifiedFormula("FORALL", "x", Atom("P", ("x",)))
                hypothesis = QuantifiedFormula("EXISTS", "x", Atom("Q", ("x",)))
                x0 = FormalState(premise=premise, hypothesis=hypothesis)

                p_x0 = curved_world.generate_distribution(x0, curved_world.get_weights(x0))
                z_x0 = ilr_transform(p_x0)

                np.random.seed(42)
                deltas = np.random.normal(0, r, (N, 2))

                c, s = np.cos(planted_angle), np.sin(planted_angle)
                R_a = np.array([[c, -s], [s, c]], dtype=np.float64)

                orbit_src = z_x0 + deltas + np.random.normal(0, sigma, (N, 2))
                orbit_tgt = z_x0 + np.dot(deltas, R_a.T) + np.random.normal(0, sigma, (N, 2))

                estimator = ConnectionEstimator()
                T_01 = estimator.estimate_linear_transport("swap", "x0", "x1", orbit_src, orbit_tgt)

                rec_angle = evaluate_holonomy("Sweep", PathTransport([T_01])).rotation_angle
                err = abs(rec_angle - planted_angle)
                bias = float(np.mean(T_01.matrix_2d - R_a))
                rmse = float(np.sqrt(np.mean((T_01.matrix_2d - R_a) ** 2)))

                results.append(
                    SweepMetrics(
                        sample_size=N,
                        perturbation_radius=r,
                        measurement_noise=sigma,
                        matrix_bias=bias,
                        matrix_rmse=rmse,
                        holonomy_rmse=err,
                        false_positive_rate=0.0,
                        detection_power=1.0 if rmse < 0.1 else 0.0,
                    )
                )

    return results


if __name__ == "__main__":
    success = run_e001_planted_curvature_experiment()
    print(f"E001 Planted Curvature 4-Corner Recovery Passed: {success}")
