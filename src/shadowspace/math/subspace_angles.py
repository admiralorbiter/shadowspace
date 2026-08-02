"""shadowspace.math.subspace_angles — Canonical principal angles and Grassmannian distance between 2D projection bases.

Sprint 13: Geometric analysis tools.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_canonical_angles(
    b1: NDArray[np.float64],
    b2: NDArray[np.float64],
) -> tuple[float, float]:
    """Compute canonical principal angles (in degrees) between two d x 2 orthonormal bases.

    Args:
        b1: Shape (d, 2) orthonormal matrix representing subspace 1.
        b2: Shape (d, 2) orthonormal matrix representing subspace 2.

    Returns:
        Tuple of (theta_1_deg, theta_2_deg) principal angles sorted ascending in [0, 90] degrees.

    Raises:
        ValueError: If matrix shapes do not match or are not 2D column bases.
    """
    if b1.ndim != 2 or b2.ndim != 2 or b1.shape != b2.shape or b1.shape[1] != 2:
        raise ValueError(
            f"Bases must be shape (d, 2) matching matrices, got {b1.shape} and {b2.shape}"
        )

    # Compute M = B1^T * B2 (shape 2x2)
    m_mat = b1.T @ b2

    # Singular values sigma_i = cos(theta_i)
    _, s_vals, _ = np.linalg.svd(m_mat)

    # Clamp singular values to [0.0, 1.0] to prevent arccos domain errors
    s_clamped = np.clip(s_vals, 0.0, 1.0)

    # Arccos gives principal angles in radians
    angles_rad = np.arccos(s_clamped)
    angles_deg = np.degrees(angles_rad)

    sorted_angles = np.sort(angles_deg)
    return float(sorted_angles[0]), float(sorted_angles[1])


def compute_grassmannian_distance(theta_1_deg: float, theta_2_deg: float) -> float:
    """Compute Grassmannian geodesic distance d_G = sqrt(theta_1^2 + theta_2^2) in degrees.

    Args:
        theta_1_deg: First principal angle in degrees.
        theta_2_deg: Second principal angle in degrees.

    Returns:
        Geodesic distance in degrees.
    """
    return float(np.hypot(theta_1_deg, theta_2_deg))
