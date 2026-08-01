"""shadowspace.projection.subspace — Grassmannian subspace distance and principal angle calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from shadowspace.projection.basis import validate_orthonormal_basis

__all__ = [
    "grassmannian_distance",
    "principal_angles",
    "find_discriminative_basis",
    "find_integrity_optimal_basis",
]


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


def find_discriminative_basis(
    matrix: NDArray[np.float64],
    labels: NDArray[np.int_] | list[str | int],
    n_components: int = 2,
) -> NDArray[np.float64]:
    """Compute an optimal 2D projection basis that maximizes class separation (Fisher LDA).

    Computes between-class scatter S_B and within-class scatter S_W, then solves the
    generalized eigenvalue problem S_B v = lambda S_W v to extract the top discriminant axes.

    Returns:
        Shape (n_features, 2) orthonormal basis matrix.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    labels_arr = np.asarray(labels)
    _n_samples, n_features = matrix.shape

    if n_features < 2:
        raise ValueError("Discriminative basis optimization requires at least 2 features.")

    unique_labels = np.unique(labels_arr)
    if len(unique_labels) < 2:
        # Fallback to PCA if only 1 class label is present
        centered = matrix - matrix.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        return validate_orthonormal_basis(vh[:2, :].T)

    mean_overall = matrix.mean(axis=0)
    sw = np.zeros((n_features, n_features), dtype=np.float64)
    sb = np.zeros((n_features, n_features), dtype=np.float64)

    for c in unique_labels:
        mask = labels_arr == c
        x_c = matrix[mask]
        n_c = x_c.shape[0]
        mean_c = x_c.mean(axis=0)

        # Within-class scatter
        diff_w = x_c - mean_c
        sw += diff_w.T @ diff_w

        # Between-class scatter
        diff_b = (mean_c - mean_overall).reshape(-1, 1)
        sb += n_c * (diff_b @ diff_b.T)

    # Regularize S_W to handle potential collinearity / zero variance
    sw += 1e-4 * np.eye(n_features)

    # Solve generalized eigenvalue problem: inv(S_W) @ S_B
    try:
        mat_target = np.linalg.inv(sw) @ sb
        eigvals, eigvecs = np.linalg.eig(mat_target)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        top_indices = np.argsort(eigvals)[::-1][:n_components]
        basis_raw = eigvecs[:, top_indices]
    except np.linalg.LinAlgError:
        # Fallback to PCA SVD
        centered = matrix - matrix.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=True)
        basis_raw = vh[:2, :].T

    # Guarantee orthonormality
    q, _ = np.linalg.qr(basis_raw)
    return validate_orthonormal_basis(q[:, :2])


def find_integrity_optimal_basis(
    matrix: NDArray[np.float64],
    target_indices: list[int],
    n_components: int = 2,
) -> NDArray[np.float64]:
    """Compute an optimal 2D projection basis that minimizes variance loss for a selected subset.

    Extracts local covariance of the target subset and aligns top projection axes with it.

    Returns:
        Shape (n_features, 2) orthonormal basis matrix.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    _n_samples, n_features = matrix.shape

    if n_features < 2:
        raise ValueError("Integrity optimal basis requires at least 2 features.")

    if not target_indices or len(target_indices) == 0:
        # Default to overall PCA
        centered = matrix - matrix.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        return validate_orthonormal_basis(vh[:2, :].T)

    subset = matrix[target_indices]
    centered_subset = subset - subset.mean(axis=0)

    if centered_subset.shape[0] < 2:
        # Fallback: variance relative to overall dataset mean
        centered_subset = subset - matrix.mean(axis=0)

    _, _, vh = np.linalg.svd(centered_subset, full_matrices=False)
    raw_basis = vh[:2, :].T if vh.shape[0] >= 2 else np.eye(n_features, 2)

    q, _ = np.linalg.qr(raw_basis)
    return validate_orthonormal_basis(q[:, :2])

