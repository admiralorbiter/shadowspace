"""Unit tests for Phase E0.7.1 holonomy estimation, TLS conditioning, and 3-group Monte Carlo sweeps."""

import numpy as np
import pytest

from research.holonomy.experiments.e000_flat_world import run_e000_flat_world_experiment
from research.holonomy.experiments.e001_planted_curvature import (
    run_e001_monte_carlo_sweeps,
    run_e001_planted_curvature_experiment,
)
from research.holonomy.geometry.connection import ConnectionEstimator, EstimatorIdentifiabilityError, ParallelTransportMap
from research.holonomy.geometry.holonomy import evaluate_holonomy
from research.holonomy.geometry.parallel_transport import PathTransport


def test_e000_flat_world():
    assert run_e000_flat_world_experiment()


def test_e001_planted_curvature_4corner_recovery():
    assert run_e001_planted_curvature_experiment(np.pi / 4)


def test_e001_monte_carlo_sweeps():
    results = run_e001_monte_carlo_sweeps(num_seeds=5)
    assert len(results) > 0
    res_250 = [r for r in results if r.sample_size == 250][0]
    # Verify TLS matrix bias reduction over OLS
    assert res_250.tls_matrix_bias_norm < res_250.ols_matrix_bias_norm
    # Verify TLS true matrix holonomy RMSE reduction over OLS
    assert res_250.tls_matrix_holonomy_rmse < res_250.ols_matrix_holonomy_rmse


def test_tls_estimator_numerical_fallback():
    estimator = ConnectionEstimator()
    # Singular source coordinates (1D line) where V22 is singular
    src = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
    tgt = np.array([[2.0, 1.0], [4.0, 2.0], [6.0, 3.0]], dtype=np.float64)

    with pytest.raises(EstimatorIdentifiabilityError):
        estimator.estimate_total_least_squares_transport("singular", "s", "t", src, tgt, strict_identifiability=True)

    t_map = estimator.estimate_total_least_squares_transport("singular", "s", "t", src, tgt, strict_identifiability=False)
    assert t_map.matrix_2d.shape == (2, 2)
    assert not np.isnan(t_map.matrix_2d).any()



def test_3x3_homogeneous_affine_composition():
    A1 = np.array([[0.0, -1.0], [1.0, 0.0]])
    b1 = np.array([1.0, 2.0])
    t1 = ParallelTransportMap("t1", "x0", "x1", A1, b1)

    path_transport = PathTransport([t1])
    assert path_transport.compute_homogeneous_matrix().shape == (3, 3)
    assert np.allclose(path_transport.compute_composite_matrix(), A1)
    assert np.allclose(path_transport.compute_translation_defect(), b1)
