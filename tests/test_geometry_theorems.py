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


def clr_transform_from_logits(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Exact CLR transform from logits: clr(q(T))_c = (z_c - mean(z)) / T."""
    z = logits / temp
    mean_z = np.mean(z, axis=-1, keepdims=True)
    return z - mean_z


def clr_transform_from_probs(p: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Dirichlet-smoothed CLR transform for probability distributions."""
    counts = p * 100.0 + alpha
    p_smooth = counts / np.sum(counts, axis=-1, keepdims=True)
    log_p = np.log(p_smooth)
    mean_log_p = np.mean(log_p, axis=-1, keepdims=True)
    return log_p - mean_log_p


def test_clr_calibration_ray_identity():
    """Assert theorem: clr(softmax(z/T)) = (1/T) clr(softmax(z))."""
    rng = np.random.default_rng(20260803)
    logits = rng.normal(size=(50, 3))
    
    clr_1 = clr_transform_from_logits(logits, 1.0)
    
    for T in [0.1, 0.5, 2.0, 5.0, 10.0]:
        clr_T = clr_transform_from_logits(logits, T)
        expected_clr_T = (1.0 / T) * clr_1
        assert np.allclose(clr_T, expected_clr_T, atol=1e-12), f"CLR calibration ray identity failed at T={T}"


def test_ambiguity_angle_temperature_invariance():
    """Assert theorem: ambiguity angle theta(p, q(T)) is 100% invariant under temperature scaling T > 0."""
    rng = np.random.default_rng(20260803)
    p = rng.dirichlet([0.5, 0.5, 0.5], size=50)
    logits = rng.normal(size=(50, 3))
    
    clr_p = clr_transform_from_probs(p, alpha=0.5)
    clr_q1 = clr_transform_from_logits(logits, temp=1.0)
    
    norm_p = np.linalg.norm(clr_p, axis=1)
    norm_q1 = np.linalg.norm(clr_q1, axis=1)
    dot_1 = np.sum(clr_p * clr_q1, axis=1)
    cos_1 = np.clip(dot_1 / (norm_p * norm_q1 + 1e-12), -1.0, 1.0)
    theta_1 = np.arccos(cos_1)
    
    for T in [0.05, 0.2, 1.5, 4.0, 20.0]:
        clr_qT = clr_transform_from_logits(logits, temp=T)
        norm_qT = np.linalg.norm(clr_qT, axis=1)
        dot_T = np.sum(clr_p * clr_qT, axis=1)
        cos_T = np.clip(dot_T / (norm_p * norm_qT + 1e-12), -1.0, 1.0)
        theta_T = np.arccos(cos_T)
        
        assert np.allclose(theta_T, theta_1, atol=1e-12), f"Ambiguity angle changed at T={T}"


def probs_to_ilr(p: np.ndarray, alpha: float = 1e-12) -> np.ndarray:
    """Isometric Log-Ratio (ILR) transform for 3-class probability vectors to 2D orthonormal space."""
    p_safe = np.clip(p, alpha, 1.0)
    p_safe = p_safe / np.sum(p_safe, axis=-1, keepdims=True)
    log_p = np.log(p_safe)
    x1 = (log_p[..., 0] - log_p[..., 1]) / np.sqrt(2.0)
    x2 = (log_p[..., 0] + log_p[..., 1] - 2.0 * log_p[..., 2]) / np.sqrt(6.0)
    return np.stack([x1, x2], axis=-1)


def ilr_to_probs(x: np.ndarray) -> np.ndarray:
    """Inverse ILR transform from 2D orthonormal space to 3-class probability simplex."""
    x1 = x[..., 0]
    x2 = x[..., 1]
    clr1 = x1 / np.sqrt(2.0) + x2 / np.sqrt(6.0)
    clr2 = -x1 / np.sqrt(2.0) + x2 / np.sqrt(6.0)
    clr3 = -2.0 * x2 / np.sqrt(6.0)
    clr = np.stack([clr1, clr2, clr3], axis=-1)
    clr_max = np.max(clr, axis=-1, keepdims=True)
    exp_clr = np.exp(clr - clr_max)
    return exp_clr / np.sum(exp_clr, axis=-1, keepdims=True)


def test_ilr_roundtrip_identity():
    """Assert theorem: ilr_to_probs(probs_to_ilr(P)) == P."""
    rng = np.random.default_rng(20260803)
    P = rng.dirichlet([0.5, 0.5, 0.5], size=100)
    
    X_ilr = probs_to_ilr(P)
    P_rec = ilr_to_probs(X_ilr)
    
    assert np.allclose(P_rec, P, atol=1e-10), "ILR roundtrip identity failed!"



