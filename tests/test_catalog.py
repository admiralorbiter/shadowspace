"""Tests for Sprint 6 — Projection catalog, basis orthonormality, and planted bridge collapse detection."""

from __future__ import annotations

import numpy as np

from shadowspace.data.calibration import calibration_fixture
from shadowspace.diagnostics.knn import compute_point_diagnostics
from shadowspace.math.metrics import pairwise_euclidean
from shadowspace.projection.basis import project, validate_orthonormal_basis
from shadowspace.projection.catalog import (
    build_projection_catalog,
    create_collapsed_bridge_view,
)


def test_catalog_basis_orthonormality() -> None:
    matrix, object_ids = calibration_fixture()
    feature_names = ["p0", "p1", "p2"]

    catalog = build_projection_catalog(matrix, object_ids, feature_names)

    assert "pca_corners" in catalog
    assert "collapsed_bridge" in catalog
    assert "entropy_axis" in catalog

    for _view_id, view in catalog.items():
        # Validate F^T F = I
        validated = validate_orthonormal_basis(view.basis)
        assert validated.shape[1] == 2
        np.testing.assert_allclose(validated.T @ validated, np.eye(2), atol=1e-10)


def test_collapsed_bridge_misleading_diagnostics() -> None:
    matrix, object_ids = calibration_fixture()

    # Collapsed bridge view (deliberately misleading)
    v_bridge = create_collapsed_bridge_view(matrix)
    coords_bridge = project(matrix, v_bridge.basis)

    src_dists = pairwise_euclidean(matrix)
    dists_bridge = pairwise_euclidean(coords_bridge)

    # Inspect midpoint_01 (index 3)
    diag_bridge = compute_point_diagnostics(src_dists, dists_bridge, k=3, object_ids=object_ids, target_id="midpoint_01")

    # Verify that the misleading collapsed_bridge view produces diagnostics with torn or false neighbors
    assert len(diag_bridge.torn) > 0 or len(diag_bridge.false_neighbors) > 0
