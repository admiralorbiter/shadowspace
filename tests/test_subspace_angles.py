"""tests.test_subspace_angles — Unit tests for canonical principal angles and Grassmannian distance."""

import numpy as np
import pytest
from shadowspace.math.subspace_angles import compute_canonical_angles, compute_grassmannian_distance


def test_canonical_angles_identity():
    """Identical bases should have 0 principal angles and 0 distance."""
    b1 = np.eye(4, 2)
    b2 = np.eye(4, 2)
    t1, t2 = compute_canonical_angles(b1, b2)
    dist = compute_grassmannian_distance(t1, t2)
    
    assert abs(t1) < 1e-5
    assert abs(t2) < 1e-5
    assert abs(dist) < 1e-5


def test_canonical_angles_orthogonal():
    """Orthogonal 2D subspaces in 4D space should have 90 degree principal angles."""
    b1 = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=float)
    b2 = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=float)
    t1, t2 = compute_canonical_angles(b1, b2)
    dist = compute_grassmannian_distance(t1, t2)
    
    assert pytest.approx(t1, abs=1e-4) == 90.0
    assert pytest.approx(t2, abs=1e-4) == 90.0
    assert pytest.approx(dist, abs=1e-4) == np.sqrt(90.0**2 + 90.0**2)


def test_canonical_angles_invalid_shape():
    """Mismatched shapes should raise ValueError."""
    b1 = np.eye(3, 2)
    b2 = np.eye(4, 2)
    with pytest.raises(ValueError, match="Bases must be shape"):
        compute_canonical_angles(b1, b2)
