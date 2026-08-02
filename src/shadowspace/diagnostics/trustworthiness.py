"""shadowspace.diagnostics.trustworthiness — Global trustworthiness, continuity, and stress metrics.

Sprint 4: Local integrity diagnostics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "compute_kruskal_stress",
    "compute_view_continuity",
    "compute_view_trustworthiness",
]


def _k_nn_normalization(n_samples: int, k: int) -> float:
    """Compute Venna-Kaski normalization factor handling k <= N/2 and k > N/2 cases.

    Ref: Venna & Kaski (2006), Neighborhood Preservation in Nonlinear Projection Methods.
    """
    if k <= n_samples / 2.0:
        denom = n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0)
    else:
        denom = n_samples * (n_samples - k) * (n_samples - k - 1.0)

    if denom <= 0.0:
        return 0.0
    return 2.0 / denom


def compute_view_trustworthiness(
    source_dists: NDArray[np.float64],
    proj_dists: NDArray[np.float64],
    k: int,
) -> float:
    """Compute Venna-Kaski Trustworthiness metric T(k) for a 2D projection.

    Trustworthiness measures how well 2D projected neighborhoods avoid false neighbors.
    Range: [0.0, 1.0], where 1.0 indicates perfect trustworthiness (no false neighbors).

    Args:
        source_dists: Shape (N, N) high-D distance matrix.
        proj_dists: Shape (N, N) 2D distance matrix.
        k: Neighborhood size k.

    Returns:
        Trustworthiness score in [0.0, 1.0].
    """
    n_samples = source_dists.shape[0]
    if n_samples <= k + 1 or k < 1:
        return 1.0

    normalization = _k_nn_normalization(n_samples, k)
    if normalization == 0.0:
        return 1.0

    # Rank of point j from point i in high-D space (0-indexed rank matrix)
    src_ranks = np.argsort(np.argsort(source_dists, axis=1), axis=1)

    penalty_sum = 0.0

    for i in range(n_samples):
        # 2D k-NN of point i (excluding self)
        proj_knn_indices = [j for j in np.argsort(proj_dists[i]) if j != i][:k]

        # High-D k-NN set of point i (excluding self)
        src_knn_set = set([j for j in np.argsort(source_dists[i]) if j != i][:k])

        for j in proj_knn_indices:
            if j not in src_knn_set:
                r_ij = src_ranks[i, j]  # 1-indexed rank among non-self points (self is rank 0)
                penalty_sum += r_ij - k

    trustworthiness = 1.0 - (normalization * penalty_sum)
    return float(np.clip(trustworthiness, 0.0, 1.0))


def compute_view_continuity(
    source_dists: NDArray[np.float64],
    proj_dists: NDArray[np.float64],
    k: int,
) -> float:
    """Compute Venna-Kaski Continuity metric C(k) for a 2D projection.

    Continuity measures how well high-D neighborhoods avoid being torn apart in 2D.
    Range: [0.0, 1.0], where 1.0 indicates perfect continuity (no torn neighbors).

    Args:
        source_dists: Shape (N, N) high-D distance matrix.
        proj_dists: Shape (N, N) 2D distance matrix.
        k: Neighborhood size k.

    Returns:
        Continuity score in [0.0, 1.0].
    """
    n_samples = source_dists.shape[0]
    if n_samples <= k + 1 or k < 1:
        return 1.0

    normalization = _k_nn_normalization(n_samples, k)
    if normalization == 0.0:
        return 1.0

    # Rank of point j from point i in 2D space (0-indexed rank matrix)
    proj_ranks = np.argsort(np.argsort(proj_dists, axis=1), axis=1)

    penalty_sum = 0.0

    for i in range(n_samples):
        # High-D k-NN of point i (excluding self)
        src_knn_indices = [j for j in np.argsort(source_dists[i]) if j != i][:k]

        # 2D k-NN set of point i (excluding self)
        proj_knn_set = set([j for j in np.argsort(proj_dists[i]) if j != i][:k])

        for j in src_knn_indices:
            if j not in proj_knn_set:
                rhat_ij = proj_ranks[i, j]  # 1-indexed rank among non-self points (self is rank 0)
                penalty_sum += rhat_ij - k

    continuity = 1.0 - (normalization * penalty_sum)
    return float(np.clip(continuity, 0.0, 1.0))


def compute_kruskal_stress(
    source_dists: NDArray[np.float64],
    proj_dists: NDArray[np.float64],
) -> float:
    """Compute Kruskal's Normalized Stress-1 between high-D and 2D distances with optimal scaling.

    Stress-1 = sqrt( sum_{i < j} (d_ij - alpha * dhat_ij)^2 / sum_{i < j} d_ij^2 )
    where alpha = sum(d_ij * dhat_ij) / sum(dhat_ij^2) optimal uniform scale factor.

    Args:
        source_dists: Shape (N, N) high-D distance matrix.
        proj_dists: Shape (N, N) 2D distance matrix.

    Returns:
        Kruskal stress value (0.0 = perfect isometric preservation).
    """
    iu = np.triu_indices(source_dists.shape[0], k=1)
    high_d = source_dists[iu]
    proj_d = proj_dists[iu]

    denom = np.sum(high_d**2)
    if denom == 0.0:
        return 0.0

    proj_denom = np.sum(proj_d**2)
    if proj_denom == 0.0:
        return 1.0

    alpha = np.sum(high_d * proj_d) / proj_denom
    num = np.sum((high_d - alpha * proj_d) ** 2)
    return float(np.sqrt(num / denom))
