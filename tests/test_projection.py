"""Tests for Sprint 3 projection core, basis validation, Grassmannian distance, PCA tours, and path semantics."""

from __future__ import annotations

import numpy as np
import pytest

from shadowspace.data.calibration import calibration_fixture
from shadowspace.projection.basis import canonicalize_basis, project, validate_orthonormal_basis
from shadowspace.projection.paths import (
    create_linear_projection_path,
    create_representation_morph_path,
    create_sequential_embedding_path,
)
from shadowspace.projection.pca import fit_representation_pca, validate_view_compatibility
from shadowspace.projection.subspace import grassmannian_distance


def test_validate_orthonormal_basis() -> None:
    # Valid 3x2 orthonormal basis
    basis_valid = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64
    )
    basis_res = validate_orthonormal_basis(basis_valid)
    np.testing.assert_array_equal(basis_res, basis_valid)

    # Non-orthonormal basis (columns not orthogonal or unit norm)
    basis_invalid = np.array([[1.0, 0.5], [0.5, 1.0], [0.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="not orthonormal"):
        validate_orthonormal_basis(basis_invalid)

    # Shape mismatch (K < d)
    with pytest.raises(ValueError, match="must be >= projection dimension"):
        validate_orthonormal_basis(np.array([[1.0, 0.0]], dtype=np.float64))


def test_project() -> None:
    x_mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)

    projected = project(x_mat, basis)
    assert projected.shape == (2, 2)
    np.testing.assert_allclose(projected, [[1.0, 2.0], [4.0, 5.0]])


def test_canonicalize_basis() -> None:
    basis = np.array([[-0.8, 0.1], [0.2, -0.9]], dtype=np.float64)
    basis_canon = canonicalize_basis(basis)

    # Column 0 max mag entry is -0.8 at idx 0 => flipped to +0.8
    assert basis_canon[0, 0] > 0.0
    # Column 1 max mag entry is -0.9 at idx 1 => flipped to +0.9
    assert basis_canon[1, 1] > 0.0


def test_grassmannian_distance_invariance() -> None:
    basis1 = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)

    # In-plane 90 degree rotation matrix R
    angle = np.pi / 2.0
    rotation_mat = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64
    )
    basis2 = basis1 @ rotation_mat

    # Same subspace => Grassmannian distance is zero
    d_gr_same = grassmannian_distance(basis1, basis2)
    np.testing.assert_allclose(d_gr_same, 0.0, atol=1e-12)

    # Orthogonal 2D subspaces in 4D
    basis_sub1 = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float64)
    basis_sub2 = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=np.float64)

    d_gr_orth = grassmannian_distance(basis_sub1, basis_sub2)
    expected_d_gr = np.sqrt(2.0 * (np.pi / 2.0) ** 2)
    np.testing.assert_allclose(d_gr_orth, expected_d_gr, atol=1e-12)


def test_pca_fit_and_view_provenance() -> None:
    matrix, object_ids = calibration_fixture()
    feature_names = ["p0", "p1", "p2"]

    basis, view = fit_representation_pca(
        matrix=matrix,
        representation_id="probability",
        object_ids=object_ids,
        feature_names=feature_names,
        view_id="pca_prob_v1",
    )

    assert basis.shape == (3, 2)
    assert view.id == "pca_prob_v1"
    assert view.representation_id == "probability"
    assert len(view.provenance["object_id_fit_hash"]) == 64
    assert len(view.provenance["feature_schema_hash"]) == 64
    assert view.provenance["component_indices"] == [0, 1]
    assert len(view.provenance["eigenvalues"]) == 2

    # Validation against matching matrix passes
    validate_view_compatibility(view, matrix, "probability", object_ids, feature_names)

    # Rejection against different representation raises ValueError
    with pytest.raises(ValueError, match="cannot be applied to representation"):
        validate_view_compatibility(view, matrix, "sqrt_probability")


def test_path_semantics() -> None:
    matrix, object_ids = calibration_fixture()
    feature_names = ["p0", "p1", "p2"]

    _, view_prob = fit_representation_pca(matrix, "probability", object_ids, feature_names, "v_prob")
    _, view_sqrt = fit_representation_pca(matrix, "sqrt_probability", object_ids, feature_names, "v_sqrt")

    # Linear projection path
    linear_path = create_linear_projection_path("path_linear", [view_prob])
    assert linear_path.kind == "linear_projection"
    assert linear_path.intermediate_frames_semantically_valid is True

    # Multi-representation in linear projection path raises ValueError
    with pytest.raises(ValueError, match="must share the same representation_id"):
        create_linear_projection_path("path_invalid", [view_prob, view_sqrt])

    # Representation morph path
    morph_path = create_representation_morph_path("path_morph", view_prob, view_sqrt)
    assert morph_path.kind == "representation_morph"
    assert morph_path.intermediate_frames_semantically_valid is False
    assert "warning" in morph_path.metadata

    # Sequential embedding path
    seq_path = create_sequential_embedding_path("path_seq", [view_prob, view_sqrt])
    assert seq_path.kind == "sequential_embedding"
    assert seq_path.intermediate_frames_semantically_valid is False
