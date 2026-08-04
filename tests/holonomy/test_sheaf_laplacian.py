"""Unit tests for Cellular Sheaf Laplacian and GlueOOD score."""

import numpy as np
import pytest

from research.holonomy.sheaf.coboundary import CoboundaryOperator, OverlapEdge
from research.holonomy.sheaf.laplacian import SheafLaplacian
from research.holonomy.sheaf.ood_gluing import compute_glue_ood_score
from research.holonomy.sheaf.restriction import LocalCalibrator


def test_sheaf_laplacian_kernel_dimension():
    patches = ["U1", "U2", "U3"]
    overlaps = [
        OverlapEdge("U1", "U2", ("item1",)),
        OverlapEdge("U2", "U3", ("item2",)),
    ]

    cob = CoboundaryOperator(patches, overlaps)
    lap = SheafLaplacian(cob, param_dim=6)
    spec = lap.compute_spectrum()

    # For a connected line graph of 3 patches, dim ker(L_F) = param_dim = 6
    assert spec.zero_eigenvalues_count == 6


def test_glue_ood_score_computation():
    # Coherent predictions (same vector)
    v1 = np.array([0.5, -0.2])
    v2 = np.array([0.5, -0.2])
    score_low = compute_glue_ood_score([v1, v2])
    assert np.isclose(score_low, 0.0, atol=1e-6)

    # Incompatible predictions
    v3 = np.array([-1.0, 2.0])
    score_high = compute_glue_ood_score([v1, v3])
    assert score_high > 1.0
