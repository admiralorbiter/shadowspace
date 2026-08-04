"""Unit tests for Phase E0.5 holonomy estimation, 3x3 homogeneous composition, and sweeps."""

import numpy as np
import pytest

from research.holonomy.experiments.e000_flat_world import run_e000_flat_world_experiment
from research.holonomy.experiments.e001_planted_curvature import (
    run_e001_estimator_sweeps,
    run_e001_planted_curvature_experiment,
)
from research.holonomy.geometry.connection import ParallelTransportMap
from research.holonomy.geometry.holonomy import evaluate_holonomy
from research.holonomy.geometry.parallel_transport import PathTransport


def test_e000_flat_world():
    assert run_e000_flat_world_experiment()


def test_e001_planted_curvature_4corner_recovery():
    assert run_e001_planted_curvature_experiment(np.pi / 4)


def test_e001_estimator_sweeps():
    results = run_e001_estimator_sweeps()
    assert len(results) > 0
    # RMSE should decrease with sample size N
    rmse_20 = [r.matrix_rmse for r in results if r.sample_size == 20][0]
    rmse_500 = [r.matrix_rmse for r in results if r.sample_size == 500][0]
    assert rmse_500 <= rmse_20


def test_3x3_homogeneous_affine_composition():
    A1 = np.array([[0.0, -1.0], [1.0, 0.0]])
    b1 = np.array([1.0, 2.0])
    t1 = ParallelTransportMap("t1", "x0", "x1", A1, b1)

    path_transport = PathTransport([t1])
    assert path_transport.compute_homogeneous_matrix().shape == (3, 3)
    assert np.allclose(path_transport.compute_composite_matrix(), A1)
    assert np.allclose(path_transport.compute_translation_defect(), b1)
