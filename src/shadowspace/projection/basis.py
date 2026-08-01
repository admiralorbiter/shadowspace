"""shadowspace.projection.basis — Orthonormal basis validation and linear projection utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["canonicalize_basis", "project", "validate_orthonormal_basis"]


def validate_orthonormal_basis(
    basis: NDArray[np.float64], rtol: float = 1e-5, atol: float = 1e-5
) -> NDArray[np.float64]:
    """Validate that matrix basis is a valid orthonormal projection basis (basis^T basis approx I).

    Args:
        basis: Shape (K, d), float64 basis matrix (default d=2 for 2-D tours).
        rtol: Relative tolerance for orthonormality check.
        atol: Absolute tolerance for orthonormality check.

    Returns:
        The validated float64 basis array.

    Raises:
        ValueError: If basis is not 2-D, K < d, or basis^T basis is not close to identity.
    """
    if basis.ndim != 2:
        raise ValueError(f"Basis matrix must be 2-D, got shape {basis.shape}")

    k_features, d_dims = basis.shape
    if k_features < d_dims:
        raise ValueError(
            f"Feature dimension K ({k_features}) must be >= projection dimension d ({d_dims})."
        )

    # Compute inner products of columns
    identity_approx = basis.T @ basis
    identity_expected = np.eye(d_dims, dtype=np.float64)

    if not np.allclose(identity_approx, identity_expected, rtol=rtol, atol=atol):
        raise ValueError(
            f"Basis columns are not orthonormal: basis^T basis differs from Identity. "
            f"Max absolute diff: {np.max(np.abs(identity_approx - identity_expected)):.6e}"
        )

    return np.asarray(basis, dtype=np.float64)


def project(matrix: NDArray[np.float64], basis: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project coordinate matrix using orthonormal basis (projected = matrix @ basis).

    Args:
        matrix: Shape (N, K) data matrix.
        basis: Shape (K, d) validated orthonormal projection basis.

    Returns:
        Shape (N, d) projected coordinate matrix.

    Raises:
        ValueError: If matrix or basis shape mismatched.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Data matrix must be 2-D, got shape {matrix.shape}")

    validated_b = validate_orthonormal_basis(basis)
    if matrix.shape[1] != validated_b.shape[0]:
        raise ValueError(
            f"Dimension mismatch between data (features={matrix.shape[1]}) "
            f"and basis (rows={validated_b.shape[0]})."
        )

    return np.asarray(matrix @ validated_b, dtype=np.float64)


def canonicalize_basis(basis: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sign-stabilize the columns of basis so the max-magnitude entry in each column is positive.

    Args:
        basis: Shape (K, d) basis matrix.

    Returns:
        Shape (K, d) sign-canonicalized basis matrix.
    """
    if basis.ndim != 2:
        raise ValueError(f"Basis matrix must be 2-D, got shape {basis.shape}")

    basis_canon = basis.copy()
    for col in range(basis_canon.shape[1]):
        max_idx = np.argmax(np.abs(basis_canon[:, col]))
        if basis_canon[max_idx, col] < 0.0:
            basis_canon[:, col] *= -1.0

    return basis_canon
