"""tests.test_subspace — Unit tests for subspace optimization and /api/optimize-view endpoint."""

import numpy as np
import pytest

from shadowspace.projection.basis import validate_orthonormal_basis
from shadowspace.projection.subspace import (
    find_discriminative_basis,
    find_integrity_optimal_basis,
    grassmannian_distance,
    principal_angles,
)
from shadowspace.server.routes import workbench_bp
from flask import Flask


def test_principal_angles_and_distance() -> None:
    b1 = np.eye(4, 2)
    b2 = np.eye(4, 2)
    angles = principal_angles(b1, b2)
    dist = grassmannian_distance(b1, b2)
    assert np.allclose(angles, [0.0, 0.0])
    assert np.isclose(dist, 0.0)


def test_find_discriminative_basis_orthonormal() -> None:
    rng = np.random.default_rng(42)
    x1 = rng.normal(loc=-2.0, size=(20, 5))
    x2 = rng.normal(loc=2.0, size=(20, 5))
    matrix = np.vstack([x1, x2])
    labels = [0] * 20 + [1] * 20

    basis = find_discriminative_basis(matrix, labels)
    assert basis.shape == (5, 2)
    # Validate orthonormality
    validated = validate_orthonormal_basis(basis)
    assert validated.shape == (5, 2)


def test_find_integrity_optimal_basis_orthonormal() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(30, 6))
    subset = [0, 1, 2, 3, 4]

    basis = find_integrity_optimal_basis(matrix, subset)
    assert basis.shape == (6, 2)
    validated = validate_orthonormal_basis(basis)
    assert validated.shape == (6, 2)


def test_api_optimize_view_route() -> None:
    app = Flask(__name__)
    app.register_blueprint(workbench_bp)
    client = app.test_client()

    # Test class separation optimization
    res = client.get("/api/optimize-view?dataset=calibration_3class&criterion=class_separation&n_frames=20")
    assert res.status_code == 200
    data = res.get_json()
    assert data["dataset"] == "calibration_3class"
    assert data["criterion"] == "class_separation"
    assert len(data["frames"]) == 20
    assert len(data["bases"]) == 20
    assert data["geodesic_algorithm"] == "GLERP"

    # Test neighborhood integrity optimization
    res2 = client.get("/api/optimize-view?dataset=calibration_3class&criterion=neighborhood_integrity&target_id=corner_0&n_frames=20")
    assert res2.status_code == 200
    data2 = res2.get_json()
    assert data2["criterion"] == "neighborhood_integrity"
    assert len(data2["frames"]) == 20
