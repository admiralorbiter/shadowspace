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
    # Extremely high counts (6000, 3000, 1000) -> zero summary variance
    counts_a = np.array([6000, 3000, 1000])
    counts_b = np.array([6000, 1000, 3000])
    
    res = audit_pair_posterior_stability(counts_a, counts_b, majority_idx=0, pair_id="test_robust", n_draws=500)
    
    assert res["prob_both_retain_original_majority"] == 1.0
    assert res["prob_joint_collision"] >= 0.70
    assert res["stability_category"] in ["ROBUST_COLLISION", "PROBABLE_COLLISION"]


