"""Tests for MetricRegistry compatibility validation and neighbor queries."""

from __future__ import annotations

import numpy as np
import pytest

from shadowspace.math.registry import MetricRegistry


def test_metric_registry_defaults() -> None:
    registry = MetricRegistry()
    assert registry.get_spec("fisher_rao").id == "fisher_rao"
    assert registry.get_spec("aitchison").id == "aitchison"
    assert registry.get_spec("euclidean").id == "euclidean"
    assert registry.get_spec("hellinger").id == "hellinger"
    assert registry.get_spec("jensen_shannon").id == "jensen_shannon"


def test_metric_registry_compatibility_validation() -> None:
    registry = MetricRegistry()

    # Valid combinations
    registry.validate_compatibility("fisher_rao", "probability")
    registry.validate_compatibility("aitchison", "probability")
    registry.validate_compatibility("euclidean", "sqrt_probability")

    # Incompatible combinations
    with pytest.raises(ValueError, match="incompatible"):
        registry.validate_compatibility("fisher_rao", "clr_probability")

    with pytest.raises(ValueError, match="incompatible"):
        registry.validate_compatibility("hellinger", "clr_probability")

    # aitchison applies CLR internally - passing pre-CLR data is not valid
    with pytest.raises(ValueError, match="incompatible"):
        registry.validate_compatibility("aitchison", "clr_probability")

    # Unknown metric
    with pytest.raises(KeyError, match="unknown_metric"):
        registry.validate_compatibility("unknown_metric", "probability")


def test_compute_pairwise_distances() -> None:
    registry = MetricRegistry()
    mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float64)

    dist_matrix = registry.compute_pairwise_distances(mat, "fisher_rao", "probability")
    assert dist_matrix.shape == (3, 3)
    np.testing.assert_allclose(dist_matrix[0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(dist_matrix[0, 1], np.pi, atol=1e-12)


def test_find_k_nearest_neighbors() -> None:
    registry = MetricRegistry()
    mat = np.array(
        [
            [1.0, 0.0, 0.0],  # 0: corner 0
            [0.9, 0.1, 0.0],  # 1: near corner 0
            [0.0, 1.0, 0.0],  # 2: corner 1
        ],
        dtype=np.float64,
    )
    ids = ["c0", "near_c0", "c1"]

    # Target: c0 (idx 0), k=2
    knn = registry.find_k_nearest_neighbors(
        matrix=mat,
        target_idx=0,
        k=2,
        metric_id="fisher_rao",
        representation_id="probability",
        object_ids=ids,
    )

    assert len(knn) == 2
    # Nearest to c0 (excluding self) should be near_c0
    assert knn[0][0] == "near_c0"
    assert knn[1][0] == "c1"
    assert knn[0][1] < knn[1][1]  # distance order
