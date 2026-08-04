"""Unit tests for Theorem 1 (Calibration-Holonomy Invariance)."""

import numpy as np
import pytest

from research.holonomy.experiments.e003_calibration_invariance import run_e003_calibration_invariance_experiment
from research.holonomy.geometry.gauge_invariants import verify_calibration_holonomy_invariance


def test_theorem1_calibration_invariance():
    assert run_e003_calibration_invariance_experiment()


def test_conjugation_invariance_explicit():
    # Arbitrary non-flat holonomy matrix
    H = np.array([[1.2, -0.4], [0.3, 0.9]])

    # Arbitrary invertible recalibration Jacobian Df
    Df = np.array([[2.0, 1.0], [0.5, 3.0]])

    assert verify_calibration_holonomy_invariance(H, Df) is True
