"""Exact distance matrix computation module for probability distributions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def compute_hellinger_matrix(p: np.ndarray) -> np.ndarray:
    """Compute exact NxN Hellinger distance matrix for 3-class probability distributions.

    d_H(p, q) = sqrt(1 - sum(sqrt(p_j * q_j)))
    """
    sqrt_p = np.sqrt(np.clip(p, 0.0, 1.0))
    bc = np.dot(sqrt_p, sqrt_p.T)  # Bhattacharyya coefficient matrix
    bc = np.clip(bc, 0.0, 1.0)
    dist = np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)


def compute_jensen_shannon_matrix(p: np.ndarray) -> np.ndarray:
    """Compute exact NxN Jensen-Shannon distance matrix with base-2 logs.

    d_JS(p, q) = sqrt( 0.5 * KL(p || m) + 0.5 * KL(q || m) )
    """
    n = len(p)
    dist = np.zeros((n, n), dtype=np.float32)
    p_clipped = np.clip(p, 1e-12, 1.0)

    # Compute row-wise KL divergence to mixture m_ij = 0.5*(p_i + p_j)
    for i in range(n):
        pi = p_clipped[i]
        # Mixture with all other rows: (N, 3)
        m = 0.5 * (pi + p_clipped)
        # KL(pi || m) = sum(pi * log2(pi / m))
        kl_i = np.sum(pi * np.log2(pi / m), axis=1)
        # KL(pj || m) = sum(pj * log2(pj / m))
        kl_j = np.sum(p_clipped * np.log2(p_clipped / m), axis=1)
        jsd = np.clip(0.5 * kl_i + 0.5 * kl_j, 0.0, 1.0)
        dist[i] = np.sqrt(jsd).astype(np.float32)

    np.fill_diagonal(dist, 0.0)
    return dist


def compute_total_variation_matrix(p: np.ndarray) -> np.ndarray:
    """Compute exact NxN Total Variation distance matrix.

    d_TV(p, q) = 0.5 * sum(|p_j - q_j|)
    """
    n = len(p)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        dist[i] = (0.5 * np.sum(np.abs(p[i] - p), axis=1)).astype(np.float32)
    np.fill_diagonal(dist, 0.0)
    return dist


def compute_euclidean_matrix(p: np.ndarray) -> np.ndarray:
    """Compute exact NxN Euclidean probability distance matrix.

    d_E(p, q) = sqrt(sum((p_j - q_j)^2))
    """
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]  # (N, N, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)


def compute_aitchison_matrix(p: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    """Compute exact NxN Aitchison distance matrix with zero replacement policy delta.

    clr(p)_j = log(p_tilde_j) - 1/3 * sum(log(p_tilde_k))
    d_A(p, q) = ||clr(p) - clr(q)||_2
    """
    p_replaced = np.maximum(p, delta)
    p_norm = p_replaced / p_replaced.sum(axis=1, keepdims=True)

    log_p = np.log(p_norm)
    clr = log_p - log_p.mean(axis=1, keepdims=True)

    diff = clr[:, np.newaxis, :] - clr[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)


def build_distance_matrix(p: np.ndarray, metric: str = "hellinger", delta: float = 1e-6) -> np.ndarray:
    """Compute NxN distance matrix for a given metric key."""
    m = metric.lower()
    if m == "hellinger":
        return compute_hellinger_matrix(p)
    elif m in ("jensen_shannon", "jsd"):
        return compute_jensen_shannon_matrix(p)
    elif m in ("total_variation", "tvd"):
        return compute_total_variation_matrix(p)
    elif m in ("euclidean", "euc"):
        return compute_euclidean_matrix(p)
    elif m in ("aitchison", "clr"):
        return compute_aitchison_matrix(p, delta=delta)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
