"""Fractional tie-aware neighborhood extraction and soft Q_NX evaluation module."""

from __future__ import annotations

import numpy as np


def compute_soft_neighborhood_weights(
    dist_matrix: np.ndarray,
    k: int = 10,
    atol: float = 1e-7,
) -> np.ndarray:
    """Compute NxN fractional tie-aware neighborhood weight matrix W.

    For each node i:
      w_ij = 1.0 for j strictly closer than k-th distance
      w_ij = r_i / |B_i| for j tied at k-th distance
      w_ij = 0.0 otherwise

    Guarantees sum_j w_ij = k for every node i.
    """
    n = len(dist_matrix)
    weights = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        row = dist_matrix[i].copy()
        row[i] = np.inf  # Exclude self

        sorted_d = np.sort(row)
        k_dist = sorted_d[k - 1]

        # A_i: strictly closer than k_dist
        mask_closer = row < (k_dist - atol)
        # B_i: tied at k_dist
        mask_tied = np.isclose(row, k_dist, atol=atol)

        n_closer = int(mask_closer.sum())
        n_tied = int(mask_tied.sum())

        r_i = k - n_closer

        weights[i, mask_closer] = 1.0
        if n_tied > 0:
            weights[i, mask_tied] = r_i / float(n_tied)

    return weights


def compute_soft_qnx(
    weights_ref: np.ndarray,
    weights_comp: np.ndarray,
    k: int = 10,
) -> tuple[float, np.ndarray]:
    """Compute fractional tie-aware neighborhood overlap O_i_soft(k) and global Q_NX_soft(k).

    O_i_soft(k) = 1/k * sum_j min(w_ij_ref, w_ij_comp)
    Q_NX_soft(k) = mean_i(O_i_soft(k))
    """
    min_weights = np.minimum(weights_ref, weights_comp)
    local_overlap = min_weights.sum(axis=1) / float(k)
    global_qnx = float(local_overlap.mean())
    return global_qnx, local_overlap
