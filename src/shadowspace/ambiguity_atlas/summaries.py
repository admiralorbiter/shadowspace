"""Summary mapping operations and minority orientation extraction."""

import numpy as np
from typing import Tuple, Dict, Any, Union
from .geometry import binary_entropy

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_TO_IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}


def compute_shannon_entropy(p: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy in bits for probability vectors p (shape (..., 3)).
    
    H(p) = - sum(p_i * log2(p_i))
    """
    p_arr = np.asarray(p, dtype=np.float64)
    p_clamped = np.clip(p_arr, 1e-15, 1.0)
    log2_p = np.log2(p_clamped)
    terms = np.where(p_arr <= 0.0, 0.0, -p_arr * log2_p)
    return np.sum(terms, axis=-1)


def extract_summary(p: np.ndarray) -> Dict[str, Any]:
    """Extract standard dashboard summary (majority_idx, max_p, entropy_bits)."""
    p_arr = np.asarray(p, dtype=np.float64)
    maj_idx = int(np.argmax(p_arr))
    max_p = float(p_arr[maj_idx])
    entropy = float(compute_shannon_entropy(p_arr))
    
    return {
        "majority_idx": maj_idx,
        "majority_label": LABEL_MAP[maj_idx],
        "confidence": max_p,
        "entropy_bits": entropy,
    }


def compute_minority_orientation(p: np.ndarray, majority_idx: int) -> float:
    """Compute minority orientation delta in [-1, 1] for probability vector p."""
    p_arr = np.asarray(p, dtype=np.float64)
    minority_indices = [i for i in range(3) if i != majority_idx]
    idx_a, idx_b = minority_indices[0], minority_indices[1]
    
    p_a = p_arr[idx_a]
    p_b = p_arr[idx_b]
    denom = p_a + p_b
    if denom <= 1e-12:
        return 0.0
    return float((p_a - p_b) / denom)


def compute_minority_orientation_batch(draws: np.ndarray, maj_indices: np.ndarray) -> np.ndarray:
    """Compute minority orientation delta for (N, 3) array of probability draws."""
    n_draws = draws.shape[0]
    deltas = np.zeros(n_draws, dtype=np.float64)
    
    for maj_idx in range(3):
        mask = (maj_indices == maj_idx)
        if not np.any(mask):
            continue
            
        min_indices = [i for i in range(3) if i != maj_idx]
        p_a = draws[mask, min_indices[0]]
        p_b = draws[mask, min_indices[1]]
        
        denom = p_a + p_b
        valid_denom = (denom > 1e-12)
        
        d = np.zeros_like(p_a)
        d[valid_denom] = (p_a[valid_denom] - p_b[valid_denom]) / denom[valid_denom]
        deltas[mask] = d
        
    return deltas
