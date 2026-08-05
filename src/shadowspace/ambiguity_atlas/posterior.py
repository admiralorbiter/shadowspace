"""Dirichlet posterior sampling and pair stability classification with joint estimands."""

import hashlib
import numpy as np
import polars as pl
from typing import Dict, Any, List, Tuple
from .geometry import hellinger_distance
from .summaries import compute_minority_orientation_batch, compute_shannon_entropy


def stable_seed(*parts: object) -> int:
    """Generate a deterministic, row-order-invariant seed from object parts using SHA-256."""
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


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
    majority_idx: int,
    pair_id: str = "",
    n_draws: int = 2000,
    alpha: float = 0.5,
    confidence_tol: float = 0.01,
    entropy_tol_bits: float = 0.02,
    orientation_eps: float = 1e-4,
) -> Dict[str, Any]:
    """Audit posterior stability for a single item pair using fixed original majority coordinate system.
    
    Evaluates:
    - prob_both_retain_original_majority: P(argmax(A) == M0 and argmax(B) == M0)
    - prob_joint_collision: P(both_retain AND opposite_orientation AND tight_summary)
    """
    seed_a = stable_seed("posterior_a", pair_id) if pair_id else 20260804
    seed_b = stable_seed("posterior_b", pair_id) if pair_id else 20260805
    
    draws_a = sample_dirichlet_posterior(counts_a, n_draws=n_draws, alpha=alpha, seed=seed_a)
    draws_b = sample_dirichlet_posterior(counts_b, n_draws=n_draws, alpha=alpha, seed=seed_b)
    
    # 1. Retention of original majority class M0
    retains_a = (np.argmax(draws_a, axis=-1) == majority_idx)
    retains_b = (np.argmax(draws_b, axis=-1) == majority_idx)
    both_retain = retains_a & retains_b
    prob_both_retain = float(np.mean(both_retain))
    
    # 2. Minority orientation computed in fixed M0 coordinate system on every draw
    fixed_majorities = np.full(n_draws, majority_idx, dtype=np.int32)
    deltas_a = compute_minority_orientation_batch(draws_a, fixed_majorities)
    deltas_b = compute_minority_orientation_batch(draws_b, fixed_majorities)
    
    opposite_orientation = (
        (deltas_a * deltas_b < 0) &
        (np.abs(deltas_a) > orientation_eps) &
        (np.abs(deltas_b) > orientation_eps)
    )
    prob_opposite_ori = float(np.mean(opposite_orientation))
    
    # 3. Summary diffs
    conf_a = draws_a[:, majority_idx]
    conf_b = draws_b[:, majority_idx]
    ent_a = compute_shannon_entropy(draws_a)
    ent_b = compute_shannon_entropy(draws_b)
    
    tight_summary = (np.abs(conf_a - conf_b) <= confidence_tol) & (np.abs(ent_a - ent_b) <= entropy_tol_bits)
    prob_tight_summary = float(np.mean(tight_summary))
    
    # 4. Joint collision event
    joint_collision = both_retain & opposite_orientation & tight_summary
    prob_joint_collision = float(np.mean(joint_collision))
    
    # Conditional probabilities given both retain M0
    n_both = int(np.sum(both_retain))
    if n_both > 0:
        prob_opposite_given_retain = float(np.mean(opposite_orientation[both_retain]))
        prob_tight_given_retain = float(np.mean(tight_summary[both_retain]))
    else:
        prob_opposite_given_retain = 0.0
        prob_tight_given_retain = 0.0
        
    # Full-space Hellinger distance credible interval
    d_h_draws = hellinger_distance(draws_a, draws_b)
    dh_median = float(np.median(d_h_draws))
    dh_q025 = float(np.percentile(d_h_draws, 2.5))
    dh_q975 = float(np.percentile(d_h_draws, 97.5))
    
    # Classification based on joint estimand
    if prob_joint_collision >= 0.70:
        category = "ROBUST_COLLISION"
    elif prob_joint_collision >= 0.40:
        category = "PROBABLE_COLLISION"
    elif prob_joint_collision >= 0.15 or (prob_both_retain >= 0.50 and prob_opposite_given_retain >= 0.50):
        category = "UNCERTAIN_COLLISION"
    else:
        category = "POINT_ESTIMATE_ONLY"
        
    return {
        "prob_both_retain_original_majority": prob_both_retain,
        "prob_opposite_orientation": prob_opposite_ori,
        "prob_tight_summary": prob_tight_summary,
        "prob_joint_collision": prob_joint_collision,
        "prob_opposite_given_retain": prob_opposite_given_retain,
        "prob_tight_given_retain": prob_tight_given_retain,
        "dh_median": dh_median,
        "dh_q025": dh_q025,
        "dh_q975": dh_q975,
        "stability_category": category,
    }
