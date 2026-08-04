"""Experiment E003: Calibration-Holonomy Invariance & Proposition 1B Bounds (Phase E0.5).

Verifies Theorem 1A (exact similarity conjugation under Jacobians) and Proposition 1B
(finite-sample estimated transport convergence under smooth nonlinear recalibrations f(z)).
"""

from __future__ import annotations

import numpy as np

from research.holonomy.geometry.connection import ConnectionEstimator
from research.holonomy.geometry.gauge_invariants import verify_calibration_holonomy_invariance


def nonlinear_recalibration_map(z: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Nonlinear smooth invertible recalibration map f(z) = z + alpha * tanh(z)."""
    return z + alpha * np.tanh(z)


def nonlinear_recalibration_jacobian(z: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Jacobian Df(z) = I + alpha * diag(1 - tanh^2(z))."""
    z_1d = np.atleast_1d(z)
    diag_terms = 1.0 - np.tanh(z_1d) ** 2
    return np.eye(2) + alpha * np.diag(diag_terms)


def run_e003_calibration_invariance_experiment() -> bool:
    """Runs E003 experiment testing Theorem 1A (exact) and Proposition 1B (finite-sample nonlinear)."""
    np.random.seed(123)

    # 1. Test Theorem 1A: Exact Similarity Conjugation
    for _ in range(10):
        angle = np.random.uniform(0.1, np.pi / 2)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        scale = np.diag([np.random.uniform(0.5, 2.0), np.random.uniform(0.5, 2.0)])
        H_orig = np.dot(R, scale)

        A = np.random.normal(0, 1, (2, 2))
        while abs(np.linalg.det(A)) < 0.1:
            A = np.random.normal(0, 1, (2, 2))

        if not verify_calibration_holonomy_invariance(H_orig, A):
            return False

    # 2. Test Proposition 1B: Nonlinear Recalibration Finite-Sample Transport Estimation
    sample_size = 500
    radius = 0.01

    z0 = np.array([0.5, -0.3], dtype=np.float64)
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    T_true = np.array([[c, -s], [s, c]], dtype=np.float64)

    deltas = np.random.normal(0, radius, (sample_size, 2))
    orbit_src = z0 + deltas
    orbit_tgt = z0 + np.dot(deltas, T_true.T)

    # Apply nonlinear recalibration f(z)
    orbit_src_f = nonlinear_recalibration_map(orbit_src)
    orbit_tgt_f = nonlinear_recalibration_map(orbit_tgt)

    estimator = ConnectionEstimator()
    T_est_f = estimator.estimate_linear_transport("gen", "src", "tgt", orbit_src_f, orbit_tgt_f).matrix_2d

    # Expected Jacobian pushforward: Df(z0) T_true Df(z0)^(-1)
    Df = nonlinear_recalibration_jacobian(z0)
    T_expected_f = np.dot(Df, np.dot(T_true, np.linalg.inv(Df)))

    # Error between estimated nonlinear transport and analytical Jacobian pushforward
    rmse_est = float(np.sqrt(np.mean((T_est_f - T_expected_f) ** 2)))

    # Convergence criterion: RMSE < 0.05
    return bool(rmse_est < 0.05)


if __name__ == "__main__":
    success = run_e003_calibration_invariance_experiment()
    print(f"E003 Theorem 1A & Proposition 1B Experiment Passed: {success}")
