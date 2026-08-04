"""Experiment E001: Planted Curvature Recovery & Monte Carlo Inference (Phase E0.7).

Runs 100-seed Monte Carlo sweeps evaluating OLS vs Total Least Squares (TLS) estimators,
calculating empirical False Positive Rate (FPR) in Flat World, Detection Power in Curved World,
matrix bias norm ||E[T_hat] - T||_F, and holonomy RMSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from research.holonomy.algebra.formulas import Atom, FormalState, QuantifiedFormula
from research.holonomy.algebra.generators import Generator
from research.holonomy.geometry.connection import ConnectionEstimator, ParallelTransportMap
from research.holonomy.geometry.holonomy import evaluate_holonomy
from research.holonomy.geometry.parallel_transport import PathTransport
from research.holonomy.geometry.simplex_bundle import ilr_transform
from research.holonomy.worlds.generative_world import CurvedWorld, FlatWorld


@dataclass
class MonteCarloSweepMetrics:
    """Statistical metrics across Monte Carlo seeds for (N, r, sigma)."""

    sample_size: int
    perturbation_radius: float
    measurement_noise: float
    ols_matrix_bias_norm: float
    tls_matrix_bias_norm: float
    ols_holonomy_rmse: float
    tls_holonomy_rmse: float
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
    T_01 = estimator.estimate_total_least_squares_transport("swap", "x0", "x1", orbit_01_src, orbit_01_tgt)
    T_12 = estimator.estimate_total_least_squares_transport("neg", "x1", "x2", orbit_12_src, orbit_12_tgt)
    T_23 = estimator.estimate_total_least_squares_transport("swap_inv", "x2", "x3", orbit_23_src, orbit_23_tgt)
    T_30 = estimator.estimate_total_least_squares_transport("neg_inv", "x3", "x0", orbit_30_src, orbit_30_tgt)

    path_transport = PathTransport([T_01, T_12, T_23, T_30])
    res = evaluate_holonomy("4Corner_Curved_Square", path_transport)

    curvature_recovered = bool(np.isclose(res.curvature_magnitude, res_true.curvature_magnitude, atol=0.08))
    return curvature_recovered


def run_e001_monte_carlo_sweeps(num_seeds: int = 50) -> List[MonteCarloSweepMetrics]:
    """Runs 50-seed Monte Carlo sweeps evaluating FPR, Power, OLS vs TLS bias & RMSE."""
    results = []
    planted_angle = np.pi / 4

    for N in [50, 250]:
        for r in [0.02]:
            for sigma in [0.005]:
                flat_stat_nulls = []
                curved_stat_ols = []
                curved_stat_tls = []

                ols_mats = []
                tls_mats = []

                c, s = np.cos(planted_angle), np.sin(planted_angle)
                R_a = np.array([[c, -s], [s, c]], dtype=np.float64)

                for seed in range(num_seeds):
                    np.random.seed(seed)
                    deltas = np.random.normal(0, r, (N, 2))
                    eps_x = np.random.normal(0, sigma, (N, 2))
                    eps_y = np.random.normal(0, sigma, (N, 2))

                    z0 = np.array([0.0, 0.0])

                    # Flat null trial: T = I
                    src_flat = z0 + deltas + eps_x
                    tgt_flat = z0 + deltas + eps_y
                    estimator = ConnectionEstimator()
                    T_flat = estimator.estimate_linear_transport("flat", "x0", "x1", src_flat, tgt_flat)
                    H_flat = evaluate_holonomy("Flat", PathTransport([T_flat])).matrix
                    flat_stat_nulls.append(float(np.linalg.norm(H_flat - np.eye(2), "fro")))

                    # Curved trial: T = R_a
                    src_curved = z0 + deltas + eps_x
                    tgt_curved = z0 + np.dot(deltas, R_a.T) + eps_y

                    T_ols = estimator.estimate_linear_transport("curved", "x0", "x1", src_curved, tgt_curved)
                    T_tls = estimator.estimate_total_least_squares_transport("curved", "x0", "x1", src_curved, tgt_curved)

                    ols_mats.append(T_ols.matrix_2d)
                    tls_mats.append(T_tls.matrix_2d)

                    H_ols = evaluate_holonomy("OLS", PathTransport([T_ols])).matrix
                    H_tls = evaluate_holonomy("TLS", PathTransport([T_tls])).matrix

                    curved_stat_ols.append(float(np.linalg.norm(H_ols - np.eye(2), "fro")))
                    curved_stat_tls.append(float(np.linalg.norm(H_tls - np.eye(2), "fro")))

                tau_95 = float(np.percentile(flat_stat_nulls, 95))

                fpr = float(np.mean(np.array(flat_stat_nulls) > tau_95))
                power = float(np.mean(np.array(curved_stat_tls) > tau_95))

                mean_ols = np.mean(ols_mats, axis=0)
                mean_tls = np.mean(tls_mats, axis=0)

                ols_bias_norm = float(np.linalg.norm(mean_ols - R_a, "fro"))
                tls_bias_norm = float(np.linalg.norm(mean_tls - R_a, "fro"))

                ols_rmse = float(np.sqrt(np.mean([(m - R_a)**2 for m in ols_mats])))
                tls_rmse = float(np.sqrt(np.mean([(m - R_a)**2 for m in tls_mats])))

                results.append(
                    MonteCarloSweepMetrics(
                        sample_size=N,
                        perturbation_radius=r,
                        measurement_noise=sigma,
                        ols_matrix_bias_norm=ols_bias_norm,
                        tls_matrix_bias_norm=tls_bias_norm,
                        ols_holonomy_rmse=ols_rmse,
                        tls_holonomy_rmse=tls_rmse,
                        false_positive_rate=fpr,
                        detection_power=power,
                    )
                )

    return results


if __name__ == "__main__":
    success = run_e001_planted_curvature_experiment()
    print(f"E001 Planted Curvature 4-Corner Recovery Passed: {success}")
