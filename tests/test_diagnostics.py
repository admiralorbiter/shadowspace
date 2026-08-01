"""Tests for Sprint 4 — Local integrity diagnostics engine."""

from __future__ import annotations

import numpy as np

from shadowspace.data.calibration import calibration_fixture
from shadowspace.diagnostics.knn import (
    classify_point_neighbors,
    compute_knn,
    compute_point_diagnostics,
)
from shadowspace.diagnostics.trustworthiness import (
    compute_kruskal_stress,
    compute_view_continuity,
    compute_view_trustworthiness,
)
from shadowspace.math.metrics import pairwise_euclidean
from shadowspace.projection.basis import project
from shadowspace.projection.pca import fit_representation_pca


def test_compute_knn_basic() -> None:
    object_ids = ["a", "b", "c", "d"]
    # 4x4 distance matrix
    dist_mat = np.array(
        [
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 0.0, 2.0, 6.0],
            [4.0, 2.0, 0.0, 3.0],
            [5.0, 6.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )

    knn = compute_knn(dist_mat, k=2, object_ids=object_ids)

    assert knn["a"] == ["b", "c"]
    assert knn["b"] == ["a", "c"]
    assert knn["c"] == ["b", "d"]
    assert knn["d"] == ["c", "a"]


def test_knn_deterministic_tie_breaking() -> None:
    object_ids = ["z_point", "a_point", "m_point"]
    # Equal distance from z_point to a_point and m_point
    dist_mat = np.array(
        [
            [0.0, 2.0, 2.0],
            [2.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )

    knn = compute_knn(dist_mat, k=1, object_ids=object_ids)
    # Tie broken by string sort order ('a_point' < 'm_point')
    assert knn["z_point"] == ["a_point"]


def test_classify_point_neighbors_partition() -> None:
    source_knn = ["b", "c", "d"]
    proj_knn = ["c", "d", "e"]

    diag = classify_point_neighbors(source_knn, proj_knn, target_id="a", k=3)

    assert diag.target_id == "a"
    assert diag.k == 3
    assert diag.preserved == ["c", "d"]
    assert diag.torn == ["b"]
    assert diag.false_neighbors == ["e"]

    # Check precision & recall formulas
    # precision = |preserved| / |proj_knn| = 2 / 3
    assert np.isclose(diag.precision, 2 / 3)
    # recall = |preserved| / |source_knn| = 2 / 3
    assert np.isclose(diag.recall, 2 / 3)
    # jaccard = |preserved| / |union| = 2 / 4 = 0.5
    assert np.isclose(diag.jaccard_overlap, 0.5)


def test_compute_point_diagnostics_calibration() -> None:
    matrix, object_ids = calibration_fixture()
    feature_names = ["p0", "p1", "p2"]

    # Compute high-D Euclidean distance matrix
    src_dists = pairwise_euclidean(matrix)

    # Fit 2D PCA and project
    basis, _ = fit_representation_pca(matrix, "probability", object_ids, feature_names)
    coords_2d = project(matrix, basis)
    proj_dists = pairwise_euclidean(coords_2d)

    # Check corner_0 diagnostics
    diag = compute_point_diagnostics(
        src_dists, proj_dists, k=3, object_ids=object_ids, target_id="corner_0"
    )

    assert diag.target_id == "corner_0"
    assert len(diag.preserved) + len(diag.torn) == 3
    assert len(diag.preserved) + len(diag.false_neighbors) == 3
    # Check disjointness
    assert set(diag.preserved).isdisjoint(set(diag.torn))
    assert set(diag.preserved).isdisjoint(set(diag.false_neighbors))


def test_trustworthiness_and_continuity_bounds() -> None:
    matrix, object_ids = calibration_fixture()
    src_dists = pairwise_euclidean(matrix)

    # Ideal projection: proj_dists identical to src_dists (or 2D linear subset)
    basis, _ = fit_representation_pca(matrix, "probability", object_ids, ["p0", "p1", "p2"])
    coords_2d = project(matrix, basis)
    proj_dists = pairwise_euclidean(coords_2d)

    t_val = compute_view_trustworthiness(src_dists, proj_dists, k=3)
    c_val = compute_view_continuity(src_dists, proj_dists, k=3)
    s_val = compute_kruskal_stress(src_dists, proj_dists)

    assert 0.0 <= t_val <= 1.0
    assert 0.0 <= c_val <= 1.0
    assert s_val >= 0.0
