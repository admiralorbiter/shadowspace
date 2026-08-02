"""Unit tests for ChaosNLI distance matrices, k-NN neighbor graphs, and graph metrics."""

from __future__ import annotations

import numpy as np

from shadowspace.chaosnli.distances import (
    build_distance_matrix,
    compute_aitchison_matrix,
    compute_euclidean_matrix,
    compute_hellinger_matrix,
    compute_jensen_shannon_matrix,
    compute_total_variation_matrix,
)
from shadowspace.chaosnli.graph_metrics import compute_lcmc, compute_local_overlap, compute_qnx
from shadowspace.chaosnli.neighbors import extract_knn_graph


def _sample_distributions() -> np.ndarray:
    return np.array([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.33, 0.33, 0.34],
    ])


def test_distance_matrices_properties() -> None:
    p = _sample_distributions()
    n = len(p)

    for metric in ["hellinger", "jensen_shannon", "total_variation", "euclidean", "aitchison"]:
        d = build_distance_matrix(p, metric=metric)
        assert d.shape == (n, n)
        # Symmetry
        np.testing.assert_allclose(d, d.T, atol=1e-5)
        # Diagonal is 0
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-5)
        # Non-negative
        assert np.all(d >= -1e-6)


def test_hellinger_bounded() -> None:
    p = _sample_distributions()
    d_hel = compute_hellinger_matrix(p)
    assert np.all(d_hel >= 0.0) and np.all(d_hel <= 1.0)


def test_knn_extraction() -> None:
    p = _sample_distributions()
    d = compute_hellinger_matrix(p)
    ids = [f"item_{i}" for i in range(len(p))]

    knn_idx, df = extract_knn_graph(d, ids, k=2, space_id="human", metric_id="hellinger")

    assert knn_idx.shape == (len(p), 2)
    assert len(df) == len(p) * 2
    # Ensure self loop is omitted
    for i in range(len(p)):
        assert i not in knn_idx[i]


def test_graph_metrics() -> None:
    ref = np.array([
        [1, 2],
        [0, 2],
        [0, 1],
    ])
    comp_same = np.array([
        [1, 2],
        [0, 2],
        [0, 1],
    ])
    comp_diff = np.array([
        [2, 0],
        [1, 0],
        [1, 2],
    ])

    qnx_same = compute_qnx(ref, comp_same)
    assert qnx_same == 1.0

    lcmc_same = compute_lcmc(qnx_same, n_items=3, k=2)
    # LCMC = 1.0 - 2 / (3 - 1) = 0.0
    assert abs(lcmc_same - 0.0) < 1e-5

    qnx_diff = compute_qnx(ref, comp_diff)
    assert 0.0 <= qnx_diff <= 1.0
