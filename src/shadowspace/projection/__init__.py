"""shadowspace.projection — Basis validation, Grassmannian distance, PCA tours, and path semantics."""

from shadowspace.projection.basis import canonicalize_basis, project, validate_orthonormal_basis
from shadowspace.projection.paths import (
    create_linear_projection_path,
    create_representation_morph_path,
    create_sequential_embedding_path,
)
from shadowspace.projection.pca import (
    compute_feature_schema_hash,
    compute_object_id_hash,
    fit_representation_pca,
    validate_view_compatibility,
)
from shadowspace.projection.subspace import grassmannian_distance, principal_angles

__all__ = [
    "canonicalize_basis",
    "compute_feature_schema_hash",
    "compute_object_id_hash",
    "create_linear_projection_path",
    "create_representation_morph_path",
    "create_sequential_embedding_path",
    "fit_representation_pca",
    "grassmannian_distance",
    "principal_angles",
    "project",
    "validate_orthonormal_basis",
    "validate_view_compatibility",
]
