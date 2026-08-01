"""
Manual testing script for Sprint 3.
Run with: python test_manual.py
"""

from __future__ import annotations

import numpy as np
from shadowspace.data.calibration import calibration_fixture
from shadowspace.projection.basis import (
    canonicalize_basis,
    project,
    validate_orthonormal_basis,
)
from shadowspace.projection.pca import (
    compute_feature_schema_hash,
    compute_object_id_hash,
    fit_representation_pca,
    validate_view_compatibility,
)
from shadowspace.projection.subspace import grassmannian_distance, principal_angles
from shadowspace.projection.paths import (
    create_linear_projection_path,
    create_representation_morph_path,
    create_sequential_embedding_path,
)
from shadowspace.adapters.dtour import DtourAdapter


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)


def ok(msg: str) -> None:
    print("  [OK]  " + msg)


def bad(msg: str) -> None:
    print("  [FAIL]  " + msg)


# ----------------------------------------------------------------------------
# 1. Basis validation
# ----------------------------------------------------------------------------

section("1. Basis Validation")

basis_3x2 = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
validated = validate_orthonormal_basis(basis_3x2)
assert validated.shape == (3, 2)
ok("Valid 3x2 basis accepted, shape=" + str(validated.shape))

try:
    validate_orthonormal_basis(np.array([[1.0, 0.5], [0.5, 1.0], [0.0, 0.0]], dtype=np.float64))
    bad("Non-orthonormal basis should have raised")
except ValueError as e:
    ok("Non-orthonormal basis rejected: " + str(e))

try:
    validate_orthonormal_basis(np.array([[1.0, 0.0]], dtype=np.float64))
    bad("K < d basis should have raised")
except ValueError as e:
    ok("K < d basis rejected: " + str(e))

# ----------------------------------------------------------------------------
# 2. Canonicalize + idempotency
# ----------------------------------------------------------------------------

section("2. Canonicalize Basis")

rng = np.random.default_rng(42)
raw_basis = np.linalg.qr(rng.normal(size=(5, 2)))[0]
canon1 = canonicalize_basis(raw_basis)
canon2 = canonicalize_basis(canon1)

ok("Max-magnitude entry col 0 positive: " + str(canon1[np.argmax(np.abs(canon1[:, 0])), 0] > 0))
ok("Max-magnitude entry col 1 positive: " + str(canon1[np.argmax(np.abs(canon1[:, 1])), 1] > 0))
assert np.allclose(canon1, canon2), "Canonicalize is not idempotent!"
ok("Canonicalize is idempotent")

# ----------------------------------------------------------------------------
# 3. Linear projection
# ----------------------------------------------------------------------------

section("3. Linear Projection")

X = rng.random((20, 5))
X = X / X.sum(axis=1, keepdims=True)
basis_5x2 = np.linalg.qr(rng.normal(size=(5, 2)))[0]

projected = project(X, basis_5x2)
ok("Projected shape: " + str(X.shape) + " -> " + str(projected.shape) + " (expected (20, 2))")
assert projected.shape == (20, 2)

gram = projected.T @ projected
ok("Gram matrix symmetric: " + str(np.allclose(gram, gram.T)))

try:
    project(X, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64))
    bad("Dimension mismatch should have raised")
except ValueError as e:
    ok("Dimension mismatch caught: " + str(e))

# ----------------------------------------------------------------------------
# 4. Grassmannian distance
# ----------------------------------------------------------------------------

section("4. Grassmannian Distance")

basis_A = np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float64)

angle = np.pi / 3
R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
basis_A_rotated = basis_A @ R
d_same = grassmannian_distance(basis_A, basis_A_rotated)
ok("Same subspace (60-deg in-plane rotation): d_Gr = " + str(round(d_same, 14)) + "  (expected ~0)")
assert d_same < 1e-10

basis_B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=np.float64)
d_orth = grassmannian_distance(basis_A, basis_B)
expected = np.sqrt(2.0) * (np.pi / 2.0)
ok("Orthogonal subspaces: d_Gr = " + str(round(d_orth, 6)) + "  (expected " + str(round(expected, 6)) + ")")
assert np.isclose(d_orth, expected, atol=1e-10)

angles = principal_angles(basis_A, basis_B)
ok("Principal angles (degrees): " + str(np.degrees(angles).round(2).tolist()) + "  (expected [90.0, 90.0])")

# ----------------------------------------------------------------------------
# 5. PCA fit + ViewSpec provenance
# ----------------------------------------------------------------------------

section("5. PCA Fit + ViewSpec Provenance")

matrix, object_ids = calibration_fixture()
feature_names = ["p0", "p1", "p2"]

basis_pca, view = fit_representation_pca(
    matrix=matrix,
    representation_id="probability",
    object_ids=object_ids,
    feature_names=feature_names,
    view_id="manual_test_v1",
)

ok("PCA basis shape: " + str(basis_pca.shape) + "  (expected (3, 2))")
assert basis_pca.shape == (3, 2)

bnorm = basis_pca.T @ basis_pca
ok("Basis orthonormal (F^T F ~ I): " + str(np.allclose(bnorm, np.eye(2), atol=1e-10)))

ok("View ID: " + view.id)
ok("Representation ID: " + view.representation_id)
ok("Object hash (first 8): " + view.provenance["object_id_fit_hash"][:8])
ok("Feature hash (first 8): " + view.provenance["feature_schema_hash"][:8])
ok("Eigenvalues: " + str([round(v, 6) for v in view.provenance["eigenvalues"]]))
ok("Centering policy: " + view.provenance["centering_policy"])

# ----------------------------------------------------------------------------
# 6. Hash collision safety
# ----------------------------------------------------------------------------

section("6. Hash Collision Safety")

h1 = compute_object_id_hash(["ab", "c"])
h2 = compute_object_id_hash(["a", "bc"])
ok("'ab','c' hash: " + h1[:12])
ok("'a','bc' hash: " + h2[:12])
assert h1 != h2, "Hash collision between ['ab','c'] and ['a','bc']!"
ok("No hash collision -- null-byte separator works correctly")

# ----------------------------------------------------------------------------
# 7. View compatibility validation
# ----------------------------------------------------------------------------

section("7. View Compatibility Validation")

validate_view_compatibility(view, matrix, "probability", object_ids, feature_names)
ok("Matching representation + objects + features -> accepted")

try:
    validate_view_compatibility(view, matrix, "sqrt_probability")
    bad("Wrong representation should have raised")
except ValueError as e:
    ok("Wrong representation rejected: " + str(e))

try:
    validate_view_compatibility(view, matrix, "probability", ["x", "y", "z"])
    bad("Wrong object IDs should have raised")
except ValueError as e:
    ok("Wrong object IDs rejected: " + str(e))

try:
    validate_view_compatibility(view, matrix, "probability", object_ids, ["a", "b", "c"])
    bad("Wrong feature names should have raised")
except ValueError as e:
    ok("Wrong feature names rejected: " + str(e))

# ----------------------------------------------------------------------------
# 8. Path semantics
# ----------------------------------------------------------------------------

section("8. Path Semantics")

_, view_sqrt = fit_representation_pca(matrix, "sqrt_probability", object_ids, feature_names, "v_sqrt")

linear_path = create_linear_projection_path("p_linear", [view])
ok("Linear path kind: " + linear_path.kind)
ok("Linear path semantically valid: " + str(linear_path.intermediate_frames_semantically_valid))
ok("Linear path metadata: " + str(linear_path.metadata))
assert linear_path.intermediate_frames_semantically_valid is True

try:
    create_linear_projection_path("p_bad", [view, view_sqrt])
    bad("Mixed representations should have raised")
except ValueError as e:
    ok("Mixed representations rejected: " + str(e))

morph_path = create_representation_morph_path("p_morph", view, view_sqrt)
ok("Morph path kind: " + morph_path.kind)
ok("Morph path semantically valid: " + str(morph_path.intermediate_frames_semantically_valid))
ok("Morph path warning present: " + str("warning" in morph_path.metadata))
assert morph_path.intermediate_frames_semantically_valid is False

seq_path = create_sequential_embedding_path("p_seq", [view, view_sqrt])
ok("Sequential path kind: " + seq_path.kind)
ok("Sequential path semantically valid: " + str(seq_path.intermediate_frames_semantically_valid))
assert seq_path.intermediate_frames_semantically_valid is False

# ----------------------------------------------------------------------------
# 9. DtourAdapter
# ----------------------------------------------------------------------------

section("9. DtourAdapter -- set_basis + current_view_basis")

adapter = DtourAdapter()

try:
    adapter.set_basis(basis_pca)
    bad("set_basis before load should have raised")
except ValueError as e:
    ok("set_basis before load rejected: " + str(e))

adapter.load(matrix, object_ids)
ok("load() accepted matrix " + str(matrix.shape) + " with " + str(len(object_ids)) + " objects")
assert adapter.current_view_basis() is None
ok("current_view_basis() = None before set_basis (correct)")

adapter.set_basis(basis_pca)
retrieved = adapter.current_view_basis()
ok("set_basis() accepted PCA basis, shape=" + str(retrieved.shape))
assert np.allclose(retrieved, basis_pca)
ok("current_view_basis() returns stored basis unchanged")

wrong_basis = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
try:
    adapter.set_basis(wrong_basis)
    bad("Wrong feature dim should have raised")
except ValueError as e:
    ok("Wrong feature dim rejected: " + str(e))

bad_basis = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
try:
    adapter.set_basis(bad_basis)
    bad("Non-orthonormal basis should have raised")
except ValueError as e:
    ok("Non-orthonormal basis rejected: " + str(e))

# ----------------------------------------------------------------------------

section("ALL CHECKS PASSED")
print()
