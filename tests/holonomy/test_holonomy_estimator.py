"""Unit tests for holonomy estimation and polar invariants."""

import numpy as np
import pytest

from research.holonomy.experiments.e000_flat_world import run_e000_flat_world_experiment
from research.holonomy.experiments.e001_planted_curvature import run_e001_planted_curvature_experiment
from research.holonomy.geometry.connection import ConnectionEstimator, ParallelTransportMap
from research.holonomy.geometry.holonomy import evaluate_holonomy
from research.holonomy.geometry.parallel_transport import PathTransport


def test_e000_flat_world():
    assert run_e000_flat_world_experiment()


def test_e001_planted_curvature_recovery():
    assert run_e001_planted_curvature_experiment(np.pi / 4)


def test_polar_decomposition_invariants():
    angle = np.pi / 3
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    scale = np.diag([2.0, 0.5])
    H = np.dot(R, scale)

    T_map = ParallelTransportMap("gen", "x0", "x0", H, np.zeros(2))
    res = evaluate_holonomy("TestLoop", PathTransport([T_map]))

    assert np.isclose(res.rotation_angle, angle, atol=1e-5)
    assert np.isclose(res.volume_distortion, 0.0, atol=1e-5)  # det(R*scale) = 2.0 * 0.5 = 1 => log(1) = 0
