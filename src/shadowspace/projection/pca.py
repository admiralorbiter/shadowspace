"""shadowspace.projection.pca — Per-representation PCA fitting and ViewSpec provenance tracking."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from shadowspace.models.schemas import ViewSpec
from shadowspace.projection.basis import canonicalize_basis, validate_orthonormal_basis

__all__ = [
    "compute_feature_schema_hash",
    "compute_object_id_hash",
    "fit_representation_pca",
    "validate_view_compatibility",
]


def compute_object_id_hash(object_ids: list[str]) -> str:
    """Compute SHA-256 hash of object IDs list.

    A null-byte separator is inserted between items to prevent hash collisions
    from adjacent-concatenation ambiguity (e.g. ['ab','c'] != ['a','bc']).
    """
    hasher = hashlib.sha256()
    _SEP = b"\x00"
    for oid in object_ids:
        hasher.update(oid.encode("utf-8"))
        hasher.update(_SEP)
    return hasher.hexdigest()


def compute_feature_schema_hash(feature_names: list[str]) -> str:
    """Compute SHA-256 hash of feature names schema list.

    A null-byte separator is inserted between items to prevent hash collisions
    from adjacent-concatenation ambiguity (e.g. ['ab','c'] != ['a','bc']).
    """
    hasher = hashlib.sha256()
    _SEP = b"\x00"
    for fn in feature_names:
        hasher.update(fn.encode("utf-8"))
        hasher.update(_SEP)
    return hasher.hexdigest()


def fit_representation_pca(
    matrix: NDArray[np.float64],
    representation_id: str,
    object_ids: list[str],
    feature_names: list[str],
    view_id: str = "pca_default",
) -> tuple[NDArray[np.float64], ViewSpec]:
    """Fit 2-D PCA separately on matrix for representation_id and build ViewSpec.

    Args:
        matrix: Shape (N, K) feature matrix.
        representation_id: Representation ID string (e.g., 'probability', 'clr_probability').
        object_ids: List of string object IDs.
        feature_names: List of feature column names.
        view_id: Unique view spec ID.

    Returns:
        Tuple of (basis F (K, 2), ViewSpec with complete provenance metadata).

    Raises:
        ValueError: If matrix is not 2-D, K < 2, or row count != len(object_ids).
    """
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
    n_rows, k_features = matrix.shape
    if n_rows != len(object_ids):
        raise ValueError(f"Row count {n_rows} does not match object_ids count {len(object_ids)}.")
    if k_features != len(feature_names):
        raise ValueError(
            f"Column count {k_features} does not match feature_names count {len(feature_names)}."
        )
    if k_features < 2:
        raise ValueError(f"Feature count K ({k_features}) must be >= 2 for 2-D PCA.")

    # Center matrix
    mean_vec = np.mean(matrix, axis=0)
    centered = matrix - mean_vec

    # SVD centered matrix: centered = u s vt
    _, s_values, vt_matrix = np.linalg.svd(centered, full_matrices=False)

    # Eigenvalues / variance explained
    eigenvalues = (s_values**2) / max(n_rows - 1, 1)

    # Top 2 components from V
    raw_basis = vt_matrix[:2, :].T  # Shape (K, 2)

    # Canonicalize sign and validate orthonormality
    basis_canon = canonicalize_basis(raw_basis)
    validated_basis = validate_orthonormal_basis(basis_canon)

    obj_hash = compute_object_id_hash(object_ids)
    feat_hash = compute_feature_schema_hash(feature_names)

    provenance = {
        "basis": validated_basis.tolist(),
        "object_id_fit_hash": obj_hash,
        "feature_schema_hash": feat_hash,
        "centering_policy": "mean_centered",
        "scaling_policy": "unscaled",
        "component_indices": [0, 1],
        "eigenvalues": eigenvalues[:2].tolist(),
        "implementation_version": "shadowspace-0.1.0",
    }

    spec = ViewSpec(
        id=view_id,
        representation_id=representation_id,
        kind="linear_projection",
        basis_ref=f"basis_{view_id}",
        provenance=provenance,
    )

    return validated_basis, spec


def validate_view_compatibility(
    view: ViewSpec,
    matrix: NDArray[np.float64],
    representation_id: str,
    object_ids: list[str] | None = None,
    feature_names: list[str] | None = None,
) -> None:
    """Validate that view spec basis is compatible with target matrix and representation.

    Raises:
        ValueError: If view representation_id does not match representation_id,
                    or feature/object provenance hashes mismatch.
    """
    if view.representation_id != representation_id:
        raise ValueError(
            f"View basis was fitted on representation {view.representation_id!r}, "
            f"cannot be applied to representation {representation_id!r}."
        )

    basis_list = view.provenance.get("basis")
    if basis_list is not None:
        basis_mat = np.array(basis_list, dtype=np.float64)
        if matrix.shape[1] != basis_mat.shape[0]:
            raise ValueError(
                f"Feature count mismatch: matrix has {matrix.shape[1]} columns, "
                f"view basis expects {basis_mat.shape[0]} features."
            )

    if object_ids is not None:
        obj_hash = compute_object_id_hash(object_ids)
        view_obj_hash = view.provenance.get("object_id_fit_hash")
        if view_obj_hash and view_obj_hash != obj_hash:
            raise ValueError(
                f"Object ID fit hash mismatch: view expected {view_obj_hash[:8]}..., "
                f"got {obj_hash[:8]}..."
            )

    if feature_names is not None:
        feat_hash = compute_feature_schema_hash(feature_names)
        view_feat_hash = view.provenance.get("feature_schema_hash")
        if view_feat_hash and view_feat_hash != feat_hash:
            raise ValueError(
                f"Feature schema hash mismatch: view expected {view_feat_hash[:8]}..., "
                f"got {feat_hash[:8]}..."
            )
