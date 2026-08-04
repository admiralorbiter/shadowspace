"""Unit tests asserting mathematical rank equivalence between Hellinger and Fisher-Rao distances."""

import numpy as np
from scipy.stats import spearmanr
from shadowspace.chaosnli.distances import compute_hellinger_matrix
from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights


def distance_fisher_rao_matrix(P: np.ndarray) -> np.ndarray:
    sqrt_P = np.sqrt(np.clip(P, 1e-12, 1.0))
    bc = np.dot(sqrt_P, sqrt_P.T)
    bc = np.clip(bc, 0.0, 1.0)
    return 2.0 * np.arccos(bc)


def test_hellinger_fisher_rao_exact_monotonicity():
    """Assert H^2(p,q) = 1 - BC(p,q) and d_FR(p,q) = 2 arccos BC(p,q) identity."""
    rng = np.random.default_rng(20260803)
    P = rng.dirichlet([0.5, 0.5, 0.5], size=100)

    D_hellinger = compute_hellinger_matrix(P)
    D_fisher_rao = distance_fisher_rao_matrix(P)

    # Monotonicity test: Spearman rank correlation must be 1.0
    iu = np.triu_indices(len(P), k=1)
    corr, pval = spearmanr(D_hellinger[iu], D_fisher_rao[iu])
    assert np.isclose(corr, 1.0, atol=1e-10), f"Expected Spearman correlation 1.0, got {corr}"


def test_hellinger_fisher_rao_exact_soft_overlap():
    """Assert fractional soft top-k neighborhood overlap Q_NX_soft(k) is exactly 1.0."""
    rng = np.random.default_rng(20260803)
    P = rng.dirichlet([0.5, 0.5, 0.5], size=100)

    D_hellinger = compute_hellinger_matrix(P)
    D_fisher_rao = distance_fisher_rao_matrix(P)

    for k in [5, 10, 20]:
        W_h = compute_soft_neighborhood_weights(D_hellinger, k=k)
        W_fr = compute_soft_neighborhood_weights(D_fisher_rao, k=k)
        min_w = np.minimum(W_h, W_fr)
        overlap = np.mean(np.sum(min_w, axis=1) / float(k))
        assert np.isclose(overlap, 1.0, atol=1e-10), f"Expected soft overlap 1.0 for k={k}, got {overlap}"
