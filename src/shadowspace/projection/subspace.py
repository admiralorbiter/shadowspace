"""shadowspace.projection.subspace — Grassmannian subspace distance and principal angle calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from shadowspace.projection.basis import validate_orthonormal_basis

__all__ = ["grassmannian_distance", "principal_angles"]


def principal_angles(
    basis1: NDArray[np.float64], basis2: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute the principal canonical angles between two subspace bases basis1 and basis2.

    Args:
        basis1: Shape (K, d) orthonormal basis matrix.
        basis2: Shape (K, d) orthonormal basis matrix.

    Returns:
        Array of d principal angles in radians, sorted in non-decreasing order.
    """
    val_basis1 = validate_orthonormal_basis(basis1)
    val_basis2 = validate_orthonormal_basis(basis2)

    if val_basis1.shape != val_basis2.shape:
        raise ValueError(
            f"Shape mismatch between basis1 {val_basis1.shape} and basis2 {val_basis2.shape}."
        )

    # Singular values of basis1^T basis2 are cos(theta_i)
    overlap_mat = val_basis1.T @ val_basis2
    singular_values = np.linalg.svd(overlap_mat, compute_uv=False)

    # Clamp to [0, 1] for arccos numerical safety
    cos_theta = np.clip(singular_values, 0.0, 1.0)
    angles = np.arccos(cos_theta)

    # Sort angles ascending
    return np.asarray(np.sort(angles), dtype=np.float64)


def grassmannian_distance(basis1: NDArray[np.float64], basis2: NDArray[np.float64]) -> float:
    """Compute Grassmannian distance between two subspace bases basis1 and basis2.

    d_Gr(basis1, basis2) = sqrt(sum(theta_i^2)) where theta_i are principal angles.

    Invariant to in-plane basis rotation (spans same subspace => distance is 0.0).

    Args:
        basis1: Shape (K, d) orthonormal basis matrix.
        basis2: Shape (K, d) orthonormal basis matrix.

    Returns:
        Grassmannian distance in radians.
    """
    angles = principal_angles(basis1, basis2)
    return float(np.sqrt(np.sum(angles**2)))
