"""Experiment E001: End-to-End Derived Curvature & 3-Group Monte Carlo Loop Inference (Phase E0.7.2).

Derives target observations strictly by propagating ILR weights sequentially around the 4-corner loop:
u_0 -> u_1 -> u_2 -> u_3 -> u_0 via CurvedWorld.transform_weights_along_edge().

Evaluates independent 3-group Monte Carlo loop holonomy experiments (50 seeds per group):
- Group 1 (Calibration): 50 flat trials calibrate null thresholds tau_OLS and tau_TLS at 95th percentile.
- Group 2 (Validation): 50 independent flat trials evaluate empirical FPR under tau.
- Group 3 (Power): 50 independent curved trials evaluate empirical Detection Power under tau.
Computes true matrix holonomy RMSE ||H_gamma,b - H_hom_true||_F against 3x3 homogeneous ground truth.
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
    ols_matrix_holonomy_rmse: float
    tls_matrix_holonomy_rmse: float
    ols_curvature_stat_rmse: float
    tls_curvature_stat_rmse: float
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
    """Runs 4-corner end-to-end derived transport estimation under non-commuting weight dynamics."""
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
    u_x0 = ilr_transform(curved_world.base_weights)

    # Edge 01 (swap)
    deltas = np.random.normal(0, perturbation_radius, (sample_size, 2))
    u_01_src = u_x0 + deltas
    u_01_tgt = np.array([curved_world.transform_weights_along_edge("swap", u) for u in u_01_src])

    z_01_src = np.array([ilr_transform(curved_world.generate_distribution_from_weights(ilr_transform.__globals__["ilr_inverse"](u))) for u in u_01_src]) + np.random.normal(0, measurement_noise, (sample_size, 2))
    z_01_tgt = np.array([ilr_transform(curved_world.generate_distribution_from_weights(ilr_transform.__globals__["ilr_inverse"](u))) for u in u_01_tgt]) + np.random.normal(0, measurement_noise, (sample_size, 2))

    # Edge 12 (neg)
    u_x1 = curved_world.transform_weights_along_edge("swap", u_x0)
    u_12_src = u_x1 + deltas
    u_12_tgt = np.array([curved_world.transform_weights_along_edge("neg", u) for u in u_12_src])

    z_12_src = np.array([ilr_transform(curved_world.generate_distribution_from_weights(ilr_transform.__globals__["ilr_inverse"](u))) for u in u_12_src]) + np.random.normal(0, measurement_noise, (sample_size, 2))
    z_12_tgt = np.array([ilr_transform(curved_world.generate_distribution_from_weights(ilr_transform.__globals__["ilr_inverse"](u))) for u in u_12_tgt]) + np.random.normal(0, measurement_noise, (sample_size, 2))

    # Edge 23 (swap_inv)
    u_x2 = curved_world.transform_weights_along_edge("neg", u_x1)
    u_23_src = u_x2 + deltas
    u_23_tgt = np.array([curved_world.transform_weights_along_edge("swap_inv", u) for u in u_23_src])

    z_23_src = np.array([ilr_transform(curved_world.generate_distribution_from_weights(ilr_transform.__globals__["ilr_inverse"](u))) for u in u_23_src]) + np.random.normal(0, measurement_noise, (sample_size, 2))
    z_23_tgt = np.array([ilr_transform(curved_world.generate_distribution_from_weights(ilr_transform.__globals__["ilr_inverse"](u))) for u in u_23_tgt]) + np.random.normal(0, measurement_noise, (sample_size, 2))

    # Edge 30 (neg_inv)
    u_x3 = curved_world.transform_weights_along_edge("swap_inv", u_x2)
    u_30_src = u_x3 + deltas
    u_30_tgt = np.array([curved_world.transform_weights_along_edge("neg_inv", u) for u in u_30_src])

    z_30_src = np.array([ilr_transform(curved_world.generate_distribution_from_weights(ilr_transform.__globals__["ilr_inverse"](u))) for u in u_30_src]) + np.random.normal(0, measurement_noise, (sample_size, 2))
    z_30_tgt = np.array([ilr_transform(curved_world.generate_distribution_from_weights(ilr_transform.__globals__["ilr_inverse"](u))) for u in u_30_tgt]) + np.random.normal(0, measurement_noise, (sample_size, 2))

    estimator = ConnectionEstimator()
    T_01 = estimator.estimate_total_least_squares_transport("swap", "x0", "x1", z_01_src, z_01_tgt)
    T_12 = estimator.estimate_total_least_squares_transport("neg", "x1", "x2", z_12_src, z_12_tgt)
    T_23 = estimator.estimate_total_least_squares_transport("swap_inv", "x2", "x3", z_23_src, z_23_tgt)
    T_30 = estimator.estimate_total_least_squares_transport("neg_inv", "x3", "x0", z_30_src, z_30_tgt)

    path_transport = PathTransport([T_01, T_12, T_23, T_30])
    res = evaluate_holonomy("4Corner_Derived_Square", path_transport)

    curvature_recovered = not res.affine_is_flat
    return curvature_recovered


def _run_single_derived_trial(
    curved: bool,
    N: int,
    r: float,
    sigma: float,
    seed: int,
    curved_world: CurvedWorld,
) -> Tuple[ParallelTransportMap, PathTransport, ParallelTransportMap, PathTransport]:
    """Generates 4-corner trial derived from CurvedWorld and returns (T_01_ols, Path_ols, T_01_tls, Path_tls)."""
    np.random.seed(seed)
    u_curr = ilr_transform(curved_world.base_weights)
    deltas = np.random.normal(0, r, (N, 2))

    generators = ["swap", "neg", "swap_inv", "neg_inv" if curved else "flat_close"]

    ols_maps = []
    tls_maps = []
    estimator = ConnectionEstimator()

    for g_name in generators:
        u_src = u_curr + deltas
        u_tgt = np.array([curved_world.transform_weights_along_edge(g_name, u) for u in u_src])

        z_src = np.array([ilr_transform(curved_world.generate_distribution_from_weights(ilr_transform.__globals__["ilr_inverse"](u))) for u in u_src]) + np.random.normal(0, sigma, (N, 2))
        z_tgt = np.array([ilr_transform(curved_world.generate_distribution_from_weights(ilr_transform.__globals__["ilr_inverse"](u))) for u in u_tgt]) + np.random.normal(0, sigma, (N, 2))

        t_ols = estimator.estimate_linear_transport(g_name, "src", "tgt", z_src, z_tgt)
        t_tls = estimator.estimate_total_least_squares_transport(g_name, "src", "tgt", z_src, z_tgt)

        ols_maps.append(t_ols)
        tls_maps.append(t_tls)

        u_curr = curved_world.transform_weights_along_edge(g_name, u_curr)

    p_ols = PathTransport(ols_maps)
    p_tls = PathTransport(tls_maps)

    return ols_maps[0], p_ols, tls_maps[0], p_tls


def run_e001_monte_carlo_sweeps(num_seeds: int = 50) -> List[MonteCarloLoopMetrics]:
    """Runs independent 3-group Monte Carlo loop holonomy experiments (num_seeds each)."""
    results = []
    curved_world = CurvedWorld(rotation_angle=np.pi / 4)

    K_a, c_a = curved_world.K_a, curved_world.c_a
    S_b, c_b = curved_world.S_b, curved_world.c_b
    K_a_inv = np.linalg.inv(K_a)
    S_b_inv = np.linalg.inv(S_b)

    # Build exact 3x3 homogeneous ground truth ParallelTransportMap objects
    T01_true = ParallelTransportMap("swap", "0", "1", K_a, c_a)
    T12_true = ParallelTransportMap("neg", "1", "2", S_b, c_b)
    T23_true = ParallelTransportMap("swap_inv", "2", "3", K_a_inv, -np.dot(K_a_inv, c_a))
    T30_true = ParallelTransportMap("neg_inv", "3", "0", S_b_inv, -np.dot(S_b_inv, c_b))

    H_hom_true = PathTransport([T01_true, T12_true, T23_true, T30_true]).compute_homogeneous_matrix()
    H_true_linear = H_hom_true[:2, :2]

    for N in [50, 250]:
        for r in [0.02]:
            for sigma in [0.005]:
                # Group 1 (Calibration): 50 seeds for threshold tau
                ols_calib_stats = []
                tls_calib_stats = []

                for seed in range(num_seeds):
                    _, p_ols, _, p_tls = _run_single_derived_trial(
                        curved=False, N=N, r=r, sigma=sigma, seed=seed, curved_world=curved_world
                    )
                    A_ols = p_ols.compute_composite_matrix()
                    A_tls = p_tls.compute_composite_matrix()

                    ols_calib_stats.append(float(np.linalg.norm(A_ols - np.eye(2), "fro")))
                    tls_calib_stats.append(float(np.linalg.norm(A_tls - np.eye(2), "fro")))

                tau_ols = float(np.percentile(ols_calib_stats, 95))
                tau_tls = float(np.percentile(tls_calib_stats, 95))

                # Group 2 (Validation): 50 new seeds for empirical FPR
                ols_val_stats = []
                tls_val_stats = []

                for seed in range(num_seeds, 2 * num_seeds):
                    _, p_ols, _, p_tls = _run_single_derived_trial(
                        curved=False, N=N, r=r, sigma=sigma, seed=seed, curved_world=curved_world
                    )
                    A_ols = p_ols.compute_composite_matrix()
                    A_tls = p_tls.compute_composite_matrix()

                    ols_val_stats.append(float(np.linalg.norm(A_ols - np.eye(2), "fro")))
                    tls_val_stats.append(float(np.linalg.norm(A_tls - np.eye(2), "fro")))

                fpr_ols = float(np.mean(np.array(ols_val_stats) > tau_ols))
                fpr_tls = float(np.mean(np.array(tls_val_stats) > tau_tls))

                # Group 3 (Power & Bias): 50 new seeds for Detection Power, Matrix Holonomy RMSE & Curvature Stat RMSE
                ols_mats = []
                tls_mats = []

                ols_h_mats = []
                tls_h_mats = []

                ols_loop_stat = []
                tls_loop_stat = []

                for seed in range(2 * num_seeds, 3 * num_seeds):
                    t01_ols, p_ols, t01_tls, p_tls = _run_single_derived_trial(
                        curved=True, N=N, r=r, sigma=sigma, seed=seed, curved_world=curved_world
                    )
                    ols_mats.append(t01_ols.matrix_2d)
                    tls_mats.append(t01_tls.matrix_2d)

                    H_ols = p_ols.compute_homogeneous_matrix()
                    H_tls = p_tls.compute_homogeneous_matrix()

                    ols_h_mats.append(H_ols)
                    tls_h_mats.append(H_tls)

                    A_ols = p_ols.compute_composite_matrix()
                    A_tls = p_tls.compute_composite_matrix()

                    ols_loop_stat.append(float(np.linalg.norm(A_ols - np.eye(2), "fro")))
                    tls_loop_stat.append(float(np.linalg.norm(A_tls - np.eye(2), "fro")))

                power_ols = float(np.mean(np.array(ols_loop_stat) > tau_ols))
                power_tls = float(np.mean(np.array(tls_loop_stat) > tau_tls))

                mean_ols = np.mean(ols_mats, axis=0)
                mean_tls = np.mean(tls_mats, axis=0)

                ols_bias_norm = float(np.linalg.norm(mean_ols - K_a, "fro"))
                tls_bias_norm = float(np.linalg.norm(mean_tls - K_a, "fro"))

                ols_edge_rmse = float(np.sqrt(np.mean([(m - K_a)**2 for m in ols_mats])))
                tls_edge_rmse = float(np.sqrt(np.mean([(m - K_a)**2 for m in tls_mats])))

                # True 3x3 homogeneous matrix holonomy RMSE ||H_b - H_hom_true||_F
                ols_mat_h_rmse = float(np.sqrt(np.mean([np.linalg.norm(h - H_hom_true, "fro")**2 for h in ols_h_mats])))
                tls_mat_h_rmse = float(np.sqrt(np.mean([np.linalg.norm(h - H_hom_true, "fro")**2 for h in tls_h_mats])))

                # Scalar curvature statistic RMSE
                true_stat = float(np.linalg.norm(H_true_linear - np.eye(2), "fro"))
                ols_stat_rmse = float(np.sqrt(np.mean([(s - true_stat)**2 for s in ols_loop_stat])))
                tls_stat_rmse = float(np.sqrt(np.mean([(s - true_stat)**2 for s in tls_loop_stat])))

                results.append(
                    MonteCarloLoopMetrics(
                        sample_size=N,
                        perturbation_radius=r,
                        measurement_noise=sigma,
                        ols_edge_matrix_rmse=ols_edge_rmse,
                        tls_edge_matrix_rmse=tls_edge_rmse,
                        ols_matrix_holonomy_rmse=ols_mat_h_rmse,
                        tls_matrix_holonomy_rmse=tls_mat_h_rmse,
                        ols_curvature_stat_rmse=ols_stat_rmse,
                        tls_curvature_stat_rmse=tls_stat_rmse,
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
