"""Tests and Hypothesis metric axiom properties for distance metrics."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from shadowspace.math.metrics import (
    aitchison_distance,
    euclidean_distance,
    fisher_rao_distance,
    hellinger_distance,
    jensen_shannon_distance,
    pairwise_aitchison,
    pairwise_euclidean,
    pairwise_fisher_rao,
    pairwise_hellinger,
    pairwise_jensen_shannon,
)
from shadowspace.math.registry import MetricRegistry


def _simplex_3() -> NDArray[np.float64]:
    """3-Class test fixture with corners, midpoints, and center."""
    c0 = np.array([1.0, 0.0, 0.0])
    c1 = np.array([0.0, 1.0, 0.0])
    c2 = np.array([0.0, 0.0, 1.0])
    m01 = np.array([0.5, 0.5, 0.0])
    center = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return np.vstack([c0, c1, c2, m01, center])


def test_fisher_rao_orthogonal_corners() -> None:
    c0 = np.array([1.0, 0.0, 0.0])
    c1 = np.array([0.0, 1.0, 0.0])
    d_fr = fisher_rao_distance(c0, c1)
    np.testing.assert_allclose(d_fr, np.pi, atol=1e-12)


def test_hellinger_orthogonal_corners() -> None:
    c0 = np.array([1.0, 0.0, 0.0])
    c1 = np.array([0.0, 1.0, 0.0])
    d_h = hellinger_distance(c0, c1)
    np.testing.assert_allclose(d_h, 1.0, atol=1e-12)


def test_jensen_shannon_orthogonal_corners() -> None:
    c0 = np.array([1.0, 0.0, 0.0])
    c1 = np.array([0.0, 1.0, 0.0])
    d_js = jensen_shannon_distance(c0, c1)
    np.testing.assert_allclose(d_js, 1.0, atol=1e-12)


def test_euclidean_orthogonal_corners() -> None:
    c0 = np.array([1.0, 0.0, 0.0])
    c1 = np.array([0.0, 1.0, 0.0])
    d_e = euclidean_distance(c0, c1)
    np.testing.assert_allclose(d_e, np.sqrt(2.0), atol=1e-12)


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

    # aitchison always applies CLR internally: clr_probability input is not valid
    with pytest.raises(ValueError, match="incompatible"):
        registry.validate_compatibility("aitchison", "clr_probability")

    # Unknown metric
    with pytest.raises(KeyError, match="unknown_metric"):
        registry.validate_compatibility("unknown_metric", "probability")


def test_pairwise_functions_match_single_pair() -> None:
    mat = _simplex_3()

    p_euc = pairwise_euclidean(mat)
    p_hel = pairwise_hellinger(mat)
    p_fr = pairwise_fisher_rao(mat)
    p_ait = pairwise_aitchison(mat)
    p_js = pairwise_jensen_shannon(mat)

    n = len(mat)
    for i in range(n):
        for j in range(n):
            np.testing.assert_allclose(p_euc[i, j], euclidean_distance(mat[i], mat[j]), atol=1e-10)
            np.testing.assert_allclose(p_hel[i, j], hellinger_distance(mat[i], mat[j]), atol=1e-10)
            np.testing.assert_allclose(p_fr[i, j], fisher_rao_distance(mat[i], mat[j]), atol=1e-10)
            np.testing.assert_allclose(p_ait[i, j], aitchison_distance(mat[i], mat[j]), atol=1e-10)
            np.testing.assert_allclose(
                p_js[i, j], jensen_shannon_distance(mat[i], mat[j]), atol=1e-10
            )


def test_pairwise_euclidean_diagonal_is_exact_zero() -> None:
    mat = _simplex_3()
    d = pairwise_euclidean(mat)
    np.testing.assert_array_equal(np.diag(d), 0.0)


@given(
    w0=st.floats(min_value=0.01, max_value=1.0),
    w1=st.floats(min_value=0.01, max_value=1.0),
    w2=st.floats(min_value=0.01, max_value=1.0),
    v0=st.floats(min_value=0.01, max_value=1.0),
    v1=st.floats(min_value=0.01, max_value=1.0),
    v2=st.floats(min_value=0.01, max_value=1.0),
)
def test_hypothesis_metric_axioms(
    w0: float, w1: float, w2: float, v0: float, v1: float, v2: float
) -> None:
    p = np.array([w0, w1, w2], dtype=np.float64)
    p /= p.sum()
    q = np.array([v0, v1, v2], dtype=np.float64)
    q /= q.sum()

    # 1. Identity of indiscernibles
    np.testing.assert_allclose(euclidean_distance(p, p), 0.0, atol=1e-12)
    np.testing.assert_allclose(hellinger_distance(p, p), 0.0, atol=1e-12)
    np.testing.assert_allclose(fisher_rao_distance(p, p), 0.0, atol=1e-12)
    np.testing.assert_allclose(aitchison_distance(p, p), 0.0, atol=1e-12)
    np.testing.assert_allclose(jensen_shannon_distance(p, p), 0.0, atol=1e-12)

    # 2. Symmetry
    np.testing.assert_allclose(euclidean_distance(p, q), euclidean_distance(q, p), atol=1e-12)
    np.testing.assert_allclose(hellinger_distance(p, q), hellinger_distance(q, p), atol=1e-12)
    np.testing.assert_allclose(fisher_rao_distance(p, q), fisher_rao_distance(q, p), atol=1e-12)
    np.testing.assert_allclose(aitchison_distance(p, q), aitchison_distance(q, p), atol=1e-12)
    np.testing.assert_allclose(
        jensen_shannon_distance(p, q), jensen_shannon_distance(q, p), atol=1e-12
    )

    # 3. Non-negativity
    assert euclidean_distance(p, q) >= 0.0
    assert hellinger_distance(p, q) >= 0.0
    assert fisher_rao_distance(p, q) >= 0.0
    assert aitchison_distance(p, q) >= 0.0
    assert jensen_shannon_distance(p, q) >= 0.0
