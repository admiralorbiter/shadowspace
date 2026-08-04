"""Unit tests for Data-Dependent Cellular Sheaf Laplacian and GlueOOD solver."""

import numpy as np
import pytest

from research.holonomy.geometry.connection import ParallelTransportMap
from research.holonomy.sheaf.coboundary import CoboundaryOperator, OverlapEdge
from research.holonomy.sheaf.laplacian import SheafLaplacian
from research.holonomy.sheaf.ood_gluing import compute_glue_ood_score


def test_data_dependent_sheaf_laplacian_cohomology():
    patches = ["U0", "U1", "U2"]
    coords = np.array([[0.5, -0.3], [0.1, 0.4], [-0.2, 0.8]], dtype=np.float64)
    overlaps = [
        OverlapEdge("U0", "U1", coords),
        OverlapEdge("U1", "U2", coords),
    ]

    cob = CoboundaryOperator(patches, overlaps)
    lap = SheafLaplacian(cob, param_dim=6)
    spec = lap.compute_spectrum()

    assert spec.dim_H0 == 6


def test_glue_ood_least_squares_solver():
    s1 = np.array([0.5, -0.2])
    s2 = np.array([0.5, -0.2])

    R_id = ParallelTransportMap("id", "u", "v", np.eye(2), np.zeros(2))
    score_low = compute_glue_ood_score([s1, s2], [R_id, R_id])
    assert np.isclose(score_low, 0.0, atol=1e-5)

    s3 = np.array([-1.0, 2.0])
    score_high = compute_glue_ood_score([s1, s3], [R_id, R_id])
    assert score_high > 1.0
