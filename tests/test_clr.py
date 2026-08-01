"""Tests for shadowspace.math — CLR transform and related utilities."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from shadowspace.math.clr import clr_transform


def _simplex(n_classes: int = 3) -> NDArray[np.float64]:
    """Create a simple test matrix — unit simplex corners plus center."""
    corners = np.eye(n_classes, dtype=np.float64)
    center = np.full((1, n_classes), 1.0 / n_classes)
    return np.vstack([corners, center])


def test_clr_output_shape() -> None:
    mat = _simplex(3)
    result = clr_transform(mat)
    assert result.shape == mat.shape


def test_clr_rows_sum_to_zero() -> None:
    """The geometric mean subtraction makes every CLR row sum to 0."""
    mat = np.array([[0.7, 0.2, 0.1], [0.3, 0.3, 0.4]], dtype=np.float64)
    result = clr_transform(mat)
    np.testing.assert_allclose(result.sum(axis=1), 0.0, atol=1e-10)


def test_clr_zero_replacement_preserves_simplex_sum() -> None:
    """After zero replacement, the adjusted vector should still be a probability simplex."""
    # (1, 0, 0) has two exact zeros
    mat = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    result = clr_transform(mat)
    assert result.shape == (1, 3)
    # CLR row sums to 0
    np.testing.assert_allclose(result.sum(axis=1), 0.0, atol=1e-10)
    # First entry is largest (dominant class = class 0)
    assert result[0, 0] > result[0, 1]
    assert result[0, 0] > result[0, 2]


def test_clr_uniform_has_zero_clr() -> None:
    """A uniform distribution produces all-zero CLR (geometric mean = arithmetic mean)."""
    mat = np.full((1, 4), 0.25, dtype=np.float64)
    result = clr_transform(mat)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_clr_requires_2d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        clr_transform(np.array([0.5, 0.5]))


def test_clr_4class_calibration_fixture() -> None:
    """Round-trip: 4-class Dirichlet samples survive CLR without error."""
    rng = np.random.default_rng(42)
    mat = rng.dirichlet([5.0, 5.0, 5.0, 5.0], size=100).astype(np.float64)
    result = clr_transform(mat)
    assert result.shape == (100, 4)
    np.testing.assert_allclose(result.sum(axis=1), 0.0, atol=1e-10)
    assert np.all(np.isfinite(result))
