"""Tests for probability space transforms (sqrt, logit, CLR)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from shadowspace.math.transforms import logit_transform, sqrt_transform


def test_sqrt_transform_basic() -> None:
    probs = np.array([[0.5, 0.5], [1.0, 0.0], [0.25, 0.75]], dtype=np.float64)
    res = sqrt_transform(probs)

    assert res.shape == (3, 2)
    # Unit norm in L2
    row_norms = np.linalg.norm(res, axis=1)
    np.testing.assert_allclose(row_norms, 1.0, atol=1e-12)


def test_sqrt_transform_rejects_negative_or_1d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        sqrt_transform(np.array([0.5, 0.5]))

    # Genuinely negative values (beyond floating-point tolerance) must be rejected
    with pytest.raises(ValueError, match="non-negative"):
        sqrt_transform(np.array([[-0.01, 1.01]]))

    # Tiny floating-point artifacts are tolerated (not raised)
    sqrt_transform(np.array([[-1e-10, 1.0 + 1e-10]]))


def test_logit_transform_basic() -> None:
    probs = np.array([[0.5, 0.5], [1.0, 0.0]], dtype=np.float64)
    res = logit_transform(probs, eps=1e-6)

    assert res.shape == (2, 2)
    # For p=0.5, logit(0.5) = log(1) = 0
    np.testing.assert_allclose(res[0], 0.0, atol=1e-12)
    assert np.all(np.isfinite(res))


def test_logit_transform_rejects_1d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        logit_transform(np.array([0.5, 0.5]))


def test_logit_transform_roundtrip() -> None:
    """sigmoid(logit(p)) ≈ p for all interior probabilities."""
    probs = np.array([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5]], dtype=np.float64)
    logits = logit_transform(probs, eps=1e-9)
    recovered = 1.0 / (1.0 + np.exp(-logits))
    np.testing.assert_allclose(recovered, probs, atol=1e-6)


@given(
    p0=st.floats(min_value=0.01, max_value=0.99),
    p1=st.floats(min_value=0.01, max_value=0.99),
)
def test_hypothesis_sqrt_transform_properties(p0: float, p1: float) -> None:
    total = p0 + p1
    vec = np.array([[p0 / total, p1 / total]], dtype=np.float64)
    res = sqrt_transform(vec)

    assert np.all(res >= 0.0)
    assert np.all(np.isfinite(res))
    np.testing.assert_allclose(np.linalg.norm(res, axis=1), 1.0, atol=1e-12)
