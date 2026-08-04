"""Experiment E001: Planted Curvature Recovery & 3-Group Monte Carlo Loop Inference (Phase E0.7.1).

Runs independent 3-group Monte Carlo loop holonomy experiments (500 seeds each):
1. Group 1 (Calibration): 500 flat trials calibrate null thresholds tau_OLS and tau_TLS at 95th percentile.
2. Group 2 (Validation): 500 independent flat trials evaluate empirical FPR under tau.
3. Group 3 (Power): 500 independent curved trials evaluate empirical Detection Power under tau.
Computes OLS vs TLS edge matrix RMSE and 4-edge loop holonomy RMSE ||H_gamma - H_true||_F.
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
class MonteCarloLoopMetrics:
    """3-group Monte Carlo statistical loop holonomy metrics."""

    sample_size: int
    perturbation_radius: float
    measurement_noise: float
    ols_edge_matrix_rmse: float
    tls_edge_matrix_rmse: float
    ols_loop_holonomy_rmse: float
    tls_loop_holonomy_rmse: float
    ols_matrix_bias_norm: float
    tls_matrix_bias_norm: float
    ols_false_positive_rate: float
    tls_false_positive_rate: float
    ols_detection_power: float
    tls_detection_power: float


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


def _run_single_4corner_trial(
    curved: bool,
    N: int,
    r: float,
    sigma: float,
    seed: int,
    planted_angle: float = np.pi / 4,
) -> Tuple[ParallelTransportMap, PathTransport, ParallelTransportMap, PathTransport]:
    """Generates 4-corner trial and returns (T_01_ols, Path_ols, T_01_tls, Path_tls)."""
    np.random.seed(seed)
    c, s = np.cos(planted_angle), np.sin(planted_angle)

    if curved:
        R_a = np.array([[c, -s], [s, c]], dtype=np.float64)
        S_b = np.array([[1.0, 0.5], [0.0, 1.0]], dtype=np.float64)
    else:
        # Flat null: Nontrivial edge maps R_a, but second edge is R_a^-1 so loop H_gamma = I
        R_a = np.array([[c, -s], [s, c]], dtype=np.float64)
        S_b = np.linalg.inv(R_a)

    S_b_inv = np.linalg.inv(S_b)
    R_a_inv = np.linalg.inv(R_a)

    z0 = np.array([0.0, 0.0])
    deltas = np.random.normal(0, r, (N, 2))

    # Add measurement noise to source and target
    e01_src, e01_tgt = np.random.normal(0, sigma, (N, 2)), np.random.normal(0, sigma, (N, 2))
    e12_src, e12_tgt = np.random.normal(0, sigma, (N, 2)), np.random.normal(0, sigma, (N, 2))
    e23_src, e23_tgt = np.random.normal(0, sigma, (N, 2)), np.random.normal(0, sigma, (N, 2))
    e30_src, e30_tgt = np.random.normal(0, sigma, (N, 2)), np.random.normal(0, sigma, (N, 2))

    o01_s, o01_t = z0 + deltas + e01_src, z0 + np.dot(deltas, R_a.T) + e01_tgt
    o12_s, o12_t = z0 + deltas + e12_src, z0 + np.dot(deltas, S_b.T) + e12_tgt
    o23_s, o23_t = z0 + deltas + e23_src, z0 + np.dot(deltas, R_a_inv.T) + e23_tgt
    o30_s, o30_t = z0 + deltas + e30_src, z0 + np.dot(deltas, S_b_inv.T) + e30_tgt

    est = ConnectionEstimator()

    # OLS Transport Maps
    t01_ols = est.estimate_linear_transport("a", "x0", "x1", o01_s, o01_t)
    t12_ols = est.estimate_linear_transport("b", "x1", "x2", o12_s, o12_t)
    t23_ols = est.estimate_linear_transport("a_inv", "x2", "x3", o23_s, o23_t)
    t30_ols = est.estimate_linear_transport("b_inv", "x3", "x0", o30_s, o30_t)
    p_ols = PathTransport([t01_ols, t12_ols, t23_ols, t30_ols])

    # TLS Transport Maps
    t01_tls = est.estimate_total_least_squares_transport("a", "x0", "x1", o01_s, o01_t)
    t12_tls = est.estimate_total_least_squares_transport("b", "x1", "x2", o12_s, o12_t)
    t23_tls = est.estimate_total_least_squares_transport("a_inv", "x2", "x3", o23_s, o23_t)
    t30_tls = est.estimate_total_least_squares_transport("b_inv", "x3", "x0", o30_s, o30_t)
    p_tls = PathTransport([t01_tls, t12_tls, t23_tls, t30_tls])

    return t01_ols, p_ols, t01_tls, p_tls


def run_e001_monte_carlo_sweeps(num_seeds: int = 100) -> List[MonteCarloLoopMetrics]:
    """Runs independent 3-group Monte Carlo loop holonomy experiments (num_seeds each)."""
    results = []
    planted_angle = np.pi / 4

    c, s = np.cos(planted_angle), np.sin(planted_angle)
    R_a = np.array([[c, -s], [s, c]], dtype=np.float64)
    S_b = np.array([[1.0, 0.5], [0.0, 1.0]], dtype=np.float64)
    H_true = np.dot(np.linalg.inv(S_b), np.dot(np.linalg.inv(R_a), np.dot(S_b, R_a)))
    H_hom_true = np.eye(3, dtype=np.float64)
    H_hom_true[:2, :2] = H_true

    for N in [50, 250]:
        for r in [0.02]:
            for sigma in [0.005]:
                # Group 1 (Calibration): 100 seeds for threshold tau
                ols_calib_stats = []
                tls_calib_stats = []

                for seed in range(num_seeds):
                    _, p_ols, _, p_tls = _run_single_4corner_trial(
                        curved=False, N=N, r=r, sigma=sigma, seed=seed, planted_angle=planted_angle
                    )
                    H_ols = p_ols.compute_homogeneous_matrix()
                    H_tls = p_tls.compute_homogeneous_matrix()

                    ols_calib_stats.append(float(np.linalg.norm(H_ols - np.eye(3), "fro")))
                    tls_calib_stats.append(float(np.linalg.norm(H_tls - np.eye(3), "fro")))

                tau_ols = float(np.percentile(ols_calib_stats, 95))
                tau_tls = float(np.percentile(tls_calib_stats, 95))

                # Group 2 (Validation): 100 new seeds for empirical FPR
                ols_val_stats = []
                tls_val_stats = []

                for seed in range(num_seeds, 2 * num_seeds):
                    _, p_ols, _, p_tls = _run_single_4corner_trial(
                        curved=False, N=N, r=r, sigma=sigma, seed=seed, planted_angle=planted_angle
                    )
                    H_ols = p_ols.compute_homogeneous_matrix()
                    H_tls = p_tls.compute_homogeneous_matrix()

                    ols_val_stats.append(float(np.linalg.norm(H_ols - np.eye(3), "fro")))
                    tls_val_stats.append(float(np.linalg.norm(H_tls - np.eye(3), "fro")))

                fpr_ols = float(np.mean(np.array(ols_val_stats) > tau_ols))
                fpr_tls = float(np.mean(np.array(tls_val_stats) > tau_tls))

                # Group 3 (Power & Bias): 100 new seeds for Detection Power, Bias, & Loop RMSE
                ols_mats = []
                tls_mats = []
                ols_loop_h = []
                tls_loop_h = []

                for seed in range(2 * num_seeds, 3 * num_seeds):
                    t01_ols, p_ols, t01_tls, p_tls = _run_single_4corner_trial(
                        curved=True, N=N, r=r, sigma=sigma, seed=seed, planted_angle=planted_angle
                    )
                    ols_mats.append(t01_ols.matrix_2d)
                    tls_mats.append(t01_tls.matrix_2d)

                    H_ols = p_ols.compute_homogeneous_matrix()
                    H_tls = p_tls.compute_homogeneous_matrix()

                    ols_loop_h.append(float(np.linalg.norm(H_ols - np.eye(3), "fro")))
                    tls_loop_h.append(float(np.linalg.norm(H_tls - np.eye(3), "fro")))

                power_ols = float(np.mean(np.array(ols_loop_h) > tau_ols))
                power_tls = float(np.mean(np.array(tls_loop_h) > tau_tls))

                mean_ols = np.mean(ols_mats, axis=0)
                mean_tls = np.mean(tls_mats, axis=0)

                ols_bias_norm = float(np.linalg.norm(mean_ols - R_a, "fro"))
                tls_bias_norm = float(np.linalg.norm(mean_tls - R_a, "fro"))

                ols_edge_rmse = float(np.sqrt(np.mean([(m - R_a)**2 for m in ols_mats])))
                tls_edge_rmse = float(np.sqrt(np.mean([(m - R_a)**2 for m in tls_mats])))

                ols_loop_rmse = float(np.sqrt(np.mean([(h - float(np.linalg.norm(H_hom_true - np.eye(3), "fro")))**2 for h in ols_loop_h])))
                tls_loop_rmse = float(np.sqrt(np.mean([(h - float(np.linalg.norm(H_hom_true - np.eye(3), "fro")))**2 for h in tls_loop_h])))

                results.append(
                    MonteCarloLoopMetrics(
                        sample_size=N,
                        perturbation_radius=r,
                        measurement_noise=sigma,
                        ols_edge_matrix_rmse=ols_edge_rmse,
                        tls_edge_matrix_rmse=tls_edge_rmse,
                        ols_loop_holonomy_rmse=ols_loop_rmse,
                        tls_loop_holonomy_rmse=tls_loop_rmse,
                        ols_matrix_bias_norm=ols_bias_norm,
                        tls_matrix_bias_norm=tls_bias_norm,
                        ols_false_positive_rate=fpr_ols,
                        tls_false_positive_rate=fpr_tls,
                        ols_detection_power=power_ols,
                        tls_detection_power=power_tls,
                    )
                )

    return results


if __name__ == "__main__":
    success = run_e001_planted_curvature_experiment()
    print(f"E001 Planted Curvature 4-Corner Recovery Passed: {success}")
