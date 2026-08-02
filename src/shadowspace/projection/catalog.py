"""shadowspace.projection.catalog — Projection catalog with revealing and deliberately misleading views.

Sprint 6: Four-class validation and projection catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from shadowspace.projection.basis import canonicalize_basis, validate_orthonormal_basis
from shadowspace.projection.pca import fit_representation_pca

__all__ = [
    "CatalogView",
    "build_projection_catalog",
    "create_collapsed_bridge_view",
    "create_corner_view",
    "create_entropy_view",
]


@dataclass(frozen=True)
class CatalogView:
    """Entry in the Projection Catalog."""

    view_id: str
    display_name: str
    basis: NDArray[np.float64]
    semantically_valid: bool = True
    is_misleading: bool = False
    description: str = ""
    warning_note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def create_corner_view(
    matrix: NDArray[np.float64],
    object_ids: list[str],
    feature_names: list[str],
    representation_id: str = "probability",
) -> CatalogView:
    """Create standard 2D PCA view revealing primary corner clusters."""
    basis, _view_spec = fit_representation_pca(
        matrix=matrix,
        representation_id=representation_id,
        object_ids=object_ids,
        feature_names=feature_names,
        view_id="pca_corners",
    )
    return CatalogView(
        view_id="pca_corners",
        display_name="PCA Corner View",
        basis=basis,
        semantically_valid=True,
        is_misleading=False,
        description="Standard 2D PCA projection revealing corner clusters and primary variance axes.",
    )


def create_collapsed_bridge_view(
    matrix: NDArray[np.float64],
    bridge_indices: list[int] | None = None,
) -> CatalogView:
    """Create a projection basis that deliberately collapses feature axes.

    Produces an orthonormal 2D basis that projects features 0 and 1 onto the same axis,
    forcing distinct populations to collapse onto each other in 2D. This view
    intentionally triggers high torn/false neighbor diagnostic indicators.
    """
    n_features = matrix.shape[1]

    raw_v: NDArray[np.float64] = np.zeros((n_features, 2), dtype=np.float64)

    if n_features == 3:
        # Collapse class 0 and class 1 onto identical x-coordinate
        raw_v[0, 0] = 1.0 / np.sqrt(2.0)
        raw_v[1, 0] = 1.0 / np.sqrt(2.0)
        raw_v[2, 1] = 1.0
    elif n_features == 4:
        # For 4-class simplex: collapse class 0 and 1 onto same axis
        raw_v[0, 0] = 0.70710678
        raw_v[1, 0] = 0.70710678
        raw_v[2, 1] = 0.70710678
        raw_v[3, 1] = 0.70710678
    else:
        # General fallback: QR decomposition
        rng = np.random.default_rng(42)
        raw_v = np.asarray(rng.normal(size=(n_features, 2)), dtype=np.float64)

    q_mat, _ = np.linalg.qr(raw_v)
    basis_raw = q_mat[:, :2]

    basis_canon = canonicalize_basis(basis_raw)
    validated_basis = validate_orthonormal_basis(basis_canon)

    return CatalogView(
        view_id="collapsed_bridge",
        display_name="Collapsed Bridge View (Misleading)",
        basis=validated_basis,
        semantically_valid=True,
        is_misleading=True,
        description="Deliberately constructed projection that collapses distinct feature axes into identical 2D locations.",
        warning_note="WARNING: This view artificially collapses distinct feature clusters onto each other, creating false proximity in 2D.",
    )


def create_entropy_view(matrix: NDArray[np.float64]) -> CatalogView:
    """Create a 2D projection view emphasizing distance from uniform center."""
    n_features = matrix.shape[1]
    center_vec = np.full(n_features, 1.0 / n_features, dtype=np.float64)

    centered = matrix - center_vec
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    if vt.shape[0] >= 3:
        raw_basis = vt[1:3, :].T
    else:
        raw_basis = vt[:2, :].T

    basis_canon = canonicalize_basis(raw_basis)
    validated_basis = validate_orthonormal_basis(basis_canon)

    return CatalogView(
        view_id="entropy_axis",
        display_name="Entropy Emphasis View",
        basis=validated_basis,
        semantically_valid=True,
        is_misleading=False,
        description="2D projection emphasizing interior variance and distance from uniform center.",
    )


def create_minor_pca_view(matrix: NDArray[np.float64]) -> CatalogView:
    """Create 2D PCA view using PC3 & PC4 for higher dimensional datasets."""
    n_features = matrix.shape[1]
    if n_features < 4:
        raise ValueError("PCA minor view requires at least 4 features.")

    mean_vec = np.mean(matrix, axis=0)
    centered = matrix - mean_vec
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    raw_basis = vt[2:4, :].T
    basis_canon = canonicalize_basis(raw_basis)
    validated_basis = validate_orthonormal_basis(basis_canon)

    return CatalogView(
        view_id="pca_minor",
        display_name="PCA Minor View (PC3–PC4)",
        basis=validated_basis,
        semantically_valid=True,
        is_misleading=False,
        description="Orthographic 2D projection onto 3rd and 4th principal components.",
    )


def create_fisher_lda_view(matrix: NDArray[np.float64]) -> CatalogView:
    """Create 2D Linear Discriminant view optimizing class separation."""
    n_features = matrix.shape[1]
    pseudo_labels = np.argmax(matrix, axis=1) if n_features >= 3 else np.zeros(matrix.shape[0], dtype=int)
    classes = np.unique(pseudo_labels)

    if len(classes) >= 2:
        overall_mean = np.mean(matrix, axis=0)
        s_b = np.zeros((n_features, n_features), dtype=np.float64)
        s_w = np.zeros((n_features, n_features), dtype=np.float64)

        for c in classes:
            c_mask = pseudo_labels == c
            n_c = np.sum(c_mask)
            if n_c == 0:
                continue
            c_mean = np.mean(matrix[c_mask], axis=0)
            mean_diff = (c_mean - overall_mean).reshape(-1, 1)
            s_b += n_c * (mean_diff @ mean_diff.T)

            class_diffs = matrix[c_mask] - c_mean
            s_w += class_diffs.T @ class_diffs

        s_w += 1e-4 * np.eye(n_features)
        try:
            eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(s_w) @ s_b)
            sorted_idx = np.argsort(np.real(eigvals))[::-1]
            top_vecs = np.real(eigvecs[:, sorted_idx[:2]])
            q_mat, _ = np.linalg.qr(top_vecs)
            raw_basis = q_mat[:, :2]
        except Exception:
            raw_basis = np.eye(n_features)[:, :2]
    else:
        raw_basis = np.eye(n_features)[:, :2]

    basis_canon = canonicalize_basis(raw_basis)
    validated_basis = validate_orthonormal_basis(basis_canon)

    return CatalogView(
        view_id="fisher_lda",
        display_name="Fisher LDA View (Max Separation)",
        basis=validated_basis,
        semantically_valid=True,
        is_misleading=False,
        description="Linear Discriminant projection optimized for maximum class separability.",
    )


def build_projection_catalog(
    matrix: NDArray[np.float64],
    object_ids: list[str],
    feature_names: list[str],
    representation_id: str = "probability",
) -> dict[str, CatalogView]:
    """Build full Projection Catalog tailored for a dataset matrix.

    Returns:
        Dict mapping view_id -> CatalogView.
    """
    n_features = matrix.shape[1]

    v_corner = create_corner_view(matrix, object_ids, feature_names, representation_id)
    catalog = {v_corner.view_id: v_corner}

    if n_features >= 4:
        v_minor = create_minor_pca_view(matrix)
        catalog[v_minor.view_id] = v_minor

    if n_features >= 3:
        v_lda = create_fisher_lda_view(matrix)
        catalog[v_lda.view_id] = v_lda

    v_bridge = create_collapsed_bridge_view(matrix)
    catalog[v_bridge.view_id] = v_bridge

    v_entropy = create_entropy_view(matrix)
    catalog[v_entropy.view_id] = v_entropy

    return catalog
