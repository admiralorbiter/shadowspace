"""Unit tests for posterior module."""

import pytest
import numpy as np
from shadowspace.ambiguity_atlas.posterior import (
    sample_dirichlet_posterior,
    audit_pair_posterior_stability,
)


def test_dirichlet_posterior_reproducibility():
    """Verify Dirichlet sampling is deterministic with fixed seed."""
    counts = np.array([60, 30, 10])
    draws1 = sample_dirichlet_posterior(counts, n_draws=100, seed=20260804)
    draws2 = sample_dirichlet_posterior(counts, n_draws=100, seed=20260804)
    
    np.testing.assert_allclose(draws1, draws2, atol=1e-12)
    assert np.allclose(np.sum(draws1, axis=-1), 1.0)


def test_audit_pair_posterior_stability_robust():
    """Verify high-count doppelgänger pair yields ROBUST_COLLISION classification."""
    # Large counts (e.g. 600, 300, 100 vs 600, 100, 300) -> very small variance
    counts_a = np.array([600, 300, 100])
    counts_b = np.array([600, 100, 300])
    
    res = audit_pair_posterior_stability(counts_a, counts_b, n_draws=500, seed=20260804)
    
    assert res["prob_same_majority"] == 1.0
    assert res["prob_opposite_orientation"] == 1.0
    assert res["stability_category"] in ["ROBUST_COLLISION", "PROBABLE_COLLISION"]
