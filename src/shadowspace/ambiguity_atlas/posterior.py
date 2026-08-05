"""Dirichlet posterior sampling and pair stability classification."""

import numpy as np
import polars as pl
from typing import Dict, Any, List, Tuple
from .geometry import hellinger_distance
from .summaries import compute_minority_orientation_batch, compute_shannon_entropy


def sample_dirichlet_posterior(
    counts: np.ndarray,
    n_draws: int = 2000,
    alpha: float = 0.5,
    seed: int = 20260804
) -> np.ndarray:
    """Draw n_draws probability vectors from Dirichlet(counts + alpha)."""
    rng = np.random.default_rng(seed)
    params = counts.astype(np.float64) + alpha
    return rng.dirichlet(params, size=n_draws)


def audit_pair_posterior_stability(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    n_draws: int = 2000,
    alpha: float = 0.5,
    seed: int = 20260804
) -> Dict[str, Any]:
    """Audit posterior stability for a single item pair using vectorized batch operations."""
    draws_a = sample_dirichlet_posterior(counts_a, n_draws=n_draws, alpha=alpha, seed=seed)
    draws_b = sample_dirichlet_posterior(counts_b, n_draws=n_draws, alpha=alpha, seed=seed + 1)
    
    maj_a = np.argmax(draws_a, axis=-1)
    maj_b = np.argmax(draws_b, axis=-1)
    
    conf_a = np.max(draws_a, axis=-1)
    conf_b = np.max(draws_b, axis=-1)
    
    ent_a = compute_shannon_entropy(draws_a)
    ent_b = compute_shannon_entropy(draws_b)
    
    # Vectorized batch minority orientation
    deltas_a = compute_minority_orientation_batch(draws_a, maj_a)
    deltas_b = compute_minority_orientation_batch(draws_b, maj_b)
        
    d_h_draws = hellinger_distance(draws_a, draws_b)
    
    same_maj_mask = (maj_a == maj_b)
    prob_same_majority = float(np.mean(same_maj_mask))
    
    opposite_ori_mask = (deltas_a * deltas_b < 0) & (np.abs(deltas_a) > 1e-4) & (np.abs(deltas_b) > 1e-4)
    prob_opposite_ori = float(np.mean(opposite_ori_mask))
    
    conf_diffs = np.abs(conf_a - conf_b)
    ent_diffs = np.abs(ent_a - ent_b)
    
    tight_mask = (conf_diffs <= 0.01) & (ent_diffs <= 0.02)
    prob_tight_summary = float(np.mean(tight_mask))
    
    dh_median = float(np.median(d_h_draws))
    dh_q025 = float(np.percentile(d_h_draws, 2.5))
    dh_q975 = float(np.percentile(d_h_draws, 97.5))
    
    if prob_same_majority >= 0.90 and prob_opposite_ori >= 0.90 and prob_tight_summary >= 0.70:
        category = "ROBUST_COLLISION"
    elif prob_same_majority >= 0.80 and prob_opposite_ori >= 0.75:
        category = "PROBABLE_COLLISION"
    elif prob_same_majority >= 0.60 and prob_opposite_ori >= 0.50:
        category = "UNCERTAIN_COLLISION"
    else:
        category = "POINT_ESTIMATE_ONLY"
        
    return {
        "prob_same_majority": prob_same_majority,
        "prob_opposite_orientation": prob_opposite_ori,
        "prob_tight_summary": prob_tight_summary,
        "dh_median": dh_median,
        "dh_q025": dh_q025,
        "dh_q975": dh_q975,
        "stability_category": category,
    }
