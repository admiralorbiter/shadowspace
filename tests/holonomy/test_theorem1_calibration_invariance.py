"""Unit tests for Theorem 1A & Proposition 1B (Calibration-Holonomy Invariance)."""

import numpy as np
import pytest

from research.holonomy.experiments.e003_calibration_invariance import run_e003_calibration_invariance_experiment
from research.holonomy.geometry.gauge_invariants import SimilarityInvariants, verify_calibration_holonomy_invariance


def test_theorem1_calibration_invariance():
    assert run_e003_calibration_invariance_experiment()


def test_similarity_invariants_taxonomy():
    H = np.array([[1.0, 0.0], [0.0, 1.0]])
    inv = SimilarityInvariants.compute(H)

    assert inv.is_identity_flat is True
    assert inv.rank_defect == 0
    assert inv.ker_dimension == 2
