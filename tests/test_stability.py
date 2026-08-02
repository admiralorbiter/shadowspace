"""Tests for point stability metrics and Rashomon set generation."""

import numpy as np
import pytest

from shadowspace.math.stability import (
    compute_point_stability,
    generate_rashomon_set,
    sample_uniform_haar_grassmannian,
)


def test_sample_uniform_haar_grassmannian():
    """Verify Haar sampling yields orthonormal bases of correct shape."""
    p = 10
    k = 2
    basis = sample_uniform_haar_grassmannian(p=p, k=k, seed=123)
    assert basis.shape == (p, k)
    # Check orthonormality: B^T B = I_2
    gram = basis.T @ basis
    np.testing.assert_allclose(gram, np.eye(k), atol=1e-10)


def test_sample_uniform_haar_grassmannian_invalid_p():
    """Verify ValueError is raised if p < k."""
    with pytest.raises(ValueError, match="must be >= projection dimension"):
        sample_uniform_haar_grassmannian(p=1, k=2)


def test_compute_point_stability_basic():
    """Test point stability calculation returns expected structure and range [0, 1]."""
    rng = np.random.default_rng(42)
    N = 30
    p = 5
    X = rng.standard_normal((N, p))
    
    # Fake catalog coords
    catalog_coords = {
        "view_1": X[:, :2],
        "view_2": X[:, 2:4],
        "view_3": X[:, [0, 3]],
    }
    
    # Pre-build fake src k-NN
    src_knn = np.array([np.argsort(np.sum((X - X[i]) ** 2, axis=1))[1:6] for i in range(N)])
    
    res = compute_point_stability(X, catalog_coords, src_knn, k=5)
    
    assert "mean_stability" in res
    assert "persistence_index" in res
    assert "volatile_index" in res
    assert "stability_scores" in res
    assert len(res["stability_scores"]) == N
    
    for s in res["stability_scores"]:
        assert 0.0 <= s <= 1.0


def test_generate_rashomon_set_basic():
    """Test Rashomon set generation returns valid ranked candidates."""
    rng = np.random.default_rng(100)
    N = 40
    p = 6
    X = rng.standard_normal((N, p))
    Y_labels = np.array([0] * 20 + [1] * 20)
    
    candidates = generate_rashomon_set(
        X, Y_labels=Y_labels, n_candidates=4, quality_threshold=0.30, seed=42
    )
    
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    assert len(candidates) <= 4
    
    for cand in candidates:
        assert "id" in cand
        assert "display_name" in cand
        assert "trustworthiness" in cand
        assert "grassmannian_dist_deg" in cand
        assert "basis" in cand
        assert cand["trustworthiness"] >= 0.30
        basis_arr = np.array(cand["basis"])
        assert basis_arr.shape == (p, 2)
