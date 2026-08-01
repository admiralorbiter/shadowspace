"""shadowspace.math.metrics — Pure distance and dissimilarity metrics for probability representations.

Implemented metrics:
- Euclidean distance
- Hellinger distance (scaled to [0, 1])
- Fisher-Rao distance (factor-of-two convention, ADR-013)
- Aitchison distance (Euclidean distance on CLR-transformed compositions)
- Jensen-Shannon distance (square root of base-2 JS divergence, in [0, 1])
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from shadowspace.conventions import FISHER_RAO_SCALE
from shadowspace.math.clr import clr_transform

__all__ = [
    "aitchison_distance",
    "euclidean_distance",
    "fisher_rao_distance",
    "hellinger_distance",
    "jensen_shannon_distance",
    "pairwise_aitchison",
    "pairwise_euclidean",
    "pairwise_fisher_rao",
    "pairwise_hellinger",
    "pairwise_jensen_shannon",
]


# ---------------------------------------------------------------------------
# 1. Euclidean Distance
# ---------------------------------------------------------------------------


def euclidean_distance(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Compute Euclidean distance between two vectors."""
    diff = x - y
    return float(np.sqrt(np.dot(diff, diff)))


def pairwise_euclidean(
    x: NDArray[np.float64], y: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """Compute pairwise Euclidean distance matrix between rows of x (and y).

    Args:
        x: Shape (N, K).
        y: Shape (M, K) or None (defaults to x, producing an N x N self-distance matrix).

    Returns:
        Shape (N, M) distance matrix. Diagonal is exactly 0.0 in the self-distance case.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got shape {x.shape}")
    is_self = y is None
    if y is None:
        y = x
    elif y.ndim != 2:
        raise ValueError(f"y must be 2-D, got shape {y.shape}")

    # Vectorized computation via (x - y)^2 = x^2 + y^2 - 2xy
    x_sq = np.sum(x**2, axis=1, keepdims=True)  # (N, 1)
    y_sq = np.sum(y**2, axis=1, keepdims=True).T  # (1, M)
    sq_dist = x_sq + y_sq - 2.0 * (x @ y.T)
    sq_dist = np.clip(sq_dist, 0.0, None)
    result = np.asarray(np.sqrt(sq_dist), dtype=np.float64)
    if is_self:
        np.fill_diagonal(result, 0.0)
    return result


# ---------------------------------------------------------------------------
# 2. Hellinger Distance
# ---------------------------------------------------------------------------


def hellinger_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Compute Hellinger distance between two probability vectors.

    d_H(p, q) = (1 / sqrt(2)) * ||sqrt(p) - sqrt(q)||_2

    Range: [0.0, 1.0].
    """
    if p is q or np.array_equal(p, q):
        return 0.0
    sp = np.sqrt(np.clip(p, 0.0, None))
    sq = np.sqrt(np.clip(q, 0.0, None))
    diff = sp - sq
    return float((1.0 / np.sqrt(2.0)) * np.sqrt(float(np.dot(diff, diff))))


def pairwise_hellinger(
    p: NDArray[np.float64], q: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """Compute pairwise Hellinger distance matrix.

    Args:
        p: Shape (N, K) probability matrix.
        q: Shape (M, K) probability matrix or None.

    Returns:
        Shape (N, M) distance matrix with values in [0, 1].
    """
    is_self = q is None
    if q is None:
        q = p
    sp = np.sqrt(np.clip(p, 0.0, None))
    sq = np.sqrt(np.clip(q, 0.0, None))
    dist = (1.0 / np.sqrt(2.0)) * pairwise_euclidean(sp, sq)
    dist_clipped = np.clip(dist, 0.0, 1.0)
    if is_self:
        np.fill_diagonal(dist_clipped, 0.0)
    return np.asarray(dist_clipped, dtype=np.float64)


# ---------------------------------------------------------------------------
# 3. Fisher-Rao Distance
# ---------------------------------------------------------------------------


def fisher_rao_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Compute Fisher-Rao distance between two probability vectors.

    Follows ADR-013: d_FR(p, q) = 2.0 * arccos(sum(sqrt(p * q)))

    Range: [0.0, pi] radians (FISHER_RAO_SCALE = 2.0).
    """
    if p is q or np.array_equal(p, q):
        return 0.0
    sp = np.sqrt(np.clip(p, 0.0, None))
    sq = np.sqrt(np.clip(q, 0.0, None))
    bc = float(np.dot(sp, sq))
    if bc >= 1.0:
        return 0.0
    bc_clamped = float(np.clip(bc, -1.0, 1.0))
    return float(FISHER_RAO_SCALE * np.arccos(bc_clamped))


def pairwise_fisher_rao(
    p: NDArray[np.float64], q: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """Compute pairwise Fisher-Rao distance matrix under ADR-013 convention.

    Args:
        p: Shape (N, K) probability matrix.
        q: Shape (M, K) probability matrix or None.

    Returns:
        Shape (N, M) distance matrix in radians.
    """
    is_self = q is None
    if q is None:
        q = p
    sp = np.sqrt(np.clip(p, 0.0, None))
    sq = np.sqrt(np.clip(q, 0.0, None))
    bc = sp @ sq.T
    bc_clamped = np.clip(bc, -1.0, 1.0)
    dist = np.asarray(FISHER_RAO_SCALE * np.arccos(bc_clamped), dtype=np.float64)
    if is_self:
        np.fill_diagonal(dist, 0.0)
    return dist


# ---------------------------------------------------------------------------
# 4. Aitchison Distance
# ---------------------------------------------------------------------------


def aitchison_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Compute Aitchison distance between two compositions (Euclidean distance on CLR)."""
    p_2d = p.reshape(1, -1)
    q_2d = q.reshape(1, -1)
    clr_p = clr_transform(p_2d).ravel()
    clr_q = clr_transform(q_2d).ravel()
    return euclidean_distance(clr_p, clr_q)


def pairwise_aitchison(
    p: NDArray[np.float64], q: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """Compute pairwise Aitchison distance matrix.

    Aitchison distance is Euclidean distance on the CLR embedding. This function
    always applies ``clr_transform`` to its inputs. Both ``p`` and ``q`` must be
    raw probability matrices (non-negative rows that sum to a positive value).
    If you already hold CLR-transformed coordinates and want Euclidean distance,
    call ``pairwise_euclidean`` directly.

    Args:
        p: Shape (N, K) non-negative probability matrix.
        q: Shape (M, K) non-negative probability matrix or None.

    Returns:
        Shape (N, M) distance matrix.
    """
    clr_p = clr_transform(p)
    if q is None:
        return pairwise_euclidean(clr_p)
    clr_q = clr_transform(q)
    return pairwise_euclidean(clr_p, clr_q)


# ---------------------------------------------------------------------------
# 5. Jensen-Shannon Distance
# ---------------------------------------------------------------------------


def _kl_divergence_base2(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Compute Kullback-Leibler divergence D_KL(p || q) in bits."""
    mask = p > 0.0
    if not np.any(mask):
        return 0.0
    p_m = p[mask]
    q_m = q[mask]
    return float(np.sum(p_m * np.log2(p_m / q_m)))


def jensen_shannon_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Compute Jensen-Shannon distance (sqrt of JS divergence in bits).

    d_JS(p, q) = sqrt(0.5 * D_KL(p || m) + 0.5 * D_KL(q || m)), m = 0.5*(p+q)

    Range: [0.0, 1.0].
    """
    if p is q or np.array_equal(p, q):
        return 0.0
    m = 0.5 * (p + q)
    jsd = 0.5 * _kl_divergence_base2(p, m) + 0.5 * _kl_divergence_base2(q, m)
    return float(np.sqrt(np.clip(jsd, 0.0, 1.0)))


def pairwise_jensen_shannon(
    p: NDArray[np.float64], q: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """Compute pairwise Jensen-Shannon distance matrix.

    Args:
        p: Shape (N, K) probability matrix.
        q: Shape (M, K) probability matrix or None.

    Returns:
        Shape (N, M) distance matrix in [0, 1].
    """
    is_self = q is None
    if q is None:
        q = p

    n_rows = p.shape[0]
    m_rows = q.shape[0]
    dist = np.zeros((n_rows, m_rows), dtype=np.float64)

    for i in range(n_rows):
        p_row = p[i]
        for j in range(m_rows):
            if is_self and j == i:
                dist[i, j] = 0.0
            elif is_self and j < i:
                dist[i, j] = dist[j, i]
            else:
                dist[i, j] = jensen_shannon_distance(p_row, q[j])

    return dist
