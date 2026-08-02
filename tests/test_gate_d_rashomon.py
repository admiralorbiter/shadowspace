"""Unit tests for Gate D Rashomon set formalization and structural stability."""

from __future__ import annotations

import numpy as np
import pytest

from shadowspace.math.stability import generate_rashomon_set, sample_uniform_haar_grassmannian
from shadowspace.math.subspace_angles import compute_canonical_angles, compute_grassmannian_distance
from shadowspace.projection.basis import canonicalize_basis, validate_orthonormal_basis


def test_subspace_angles_identity_and_clamping() -> None:
    """Verify principal angles and Grassmannian distance between identical bases are exactly zero."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 5))
    Q, _ = np.linalg.qr(X)
    b1 = Q[:, :2]

    a1, a2 = compute_canonical_angles(b1, b1)
    assert pytest.approx(a1, abs=1e-5) == 0.0
    assert pytest.approx(a2, abs=1e-5) == 0.0
    assert pytest.approx(compute_grassmannian_distance(a1, a2), abs=1e-5) == 0.0


def test_canonicalize_basis_sign_invariance() -> None:
    """Verify canonicalize_basis produces identical bases regardless of column sign flips."""
    rng = np.random.default_rng(123)
    X = rng.standard_normal((8, 4))
    Q, _ = np.linalg.qr(X)
    b_orig = Q[:, :2]

    # Flip signs of columns
    b_flipped = b_orig.copy()
    b_flipped[:, 0] *= -1.0
    b_flipped[:, 1] *= -1.0

    c_orig = canonicalize_basis(b_orig)
    c_flipped = canonicalize_basis(b_flipped)

    np.testing.assert_allclose(c_orig, c_flipped)


def test_generate_rashomon_set_diversity() -> None:
    """Verify generate_rashomon_set produces distinct, high-quality candidate projection bases."""
    rng = np.random.default_rng(42)
    X = rng.dirichlet(alpha=[1.0, 1.0, 1.0, 1.0], size=30)
    labels = np.array([0] * 15 + [1] * 15)

    candidates = generate_rashomon_set(
        X=X,
        Y_labels=labels,
        n_candidates=5,
        quality_threshold=0.30,
        seed=42,
    )

    assert len(candidates) >= 2
    for cand in candidates:
        assert "id" in cand
        assert "display_name" in cand
        assert "trustworthiness" in cand
        assert cand["trustworthiness"] >= 0.30
        assert "basis" in cand
        validate_orthonormal_basis(np.array(cand["basis"]))
