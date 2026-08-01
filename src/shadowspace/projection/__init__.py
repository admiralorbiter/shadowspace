"""shadowspace.projection — Basis validation, Grassmannian distance, PCA tours, catalog, and path semantics."""

from shadowspace.projection.basis import canonicalize_basis, project, validate_orthonormal_basis
from shadowspace.projection.catalog import (
    CatalogView,
    build_projection_catalog,
    create_collapsed_bridge_view,
    create_corner_view,
    create_entropy_view,
)
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
from shadowspace.projection.subspace import (
    find_discriminative_basis,
    find_integrity_optimal_basis,
    grassmannian_distance,
    principal_angles,
)

__all__ = [
    "CatalogView",
    "build_projection_catalog",
    "canonicalize_basis",
    "compute_feature_schema_hash",
    "compute_object_id_hash",
    "create_collapsed_bridge_view",
    "create_corner_view",
    "create_entropy_view",
    "create_linear_projection_path",
    "create_representation_morph_path",
    "create_sequential_embedding_path",
    "find_discriminative_basis",
    "find_integrity_optimal_basis",
    "fit_representation_pca",
    "grassmannian_distance",
    "principal_angles",
    "project",
    "validate_orthonormal_basis",
    "validate_view_compatibility",
]
