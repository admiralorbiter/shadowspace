"""Tests for Sprint 10 — Grand Tour Paths and /api/tour-path endpoint."""

from __future__ import annotations

import json

import numpy as np
import pytest
from flask.testing import FlaskClient

from shadowspace.projection.paths import (
    generate_grand_tour_path,
    grassmann_geodesic,
    interpolate_orthonormal_bases,
)
from shadowspace.server import create_app


@pytest.fixture()
def client() -> FlaskClient:
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_grassmann_geodesic_endpoints_and_orthonormality() -> None:
    """Test GLERP geodesic interpolation properties."""
    rng = np.random.default_rng(42)
    raw_a = rng.normal(size=(6, 2))
    raw_b = rng.normal(size=(6, 2))

    q_a, _ = np.linalg.qr(raw_a)
    q_b, _ = np.linalg.qr(raw_b)
    basis_a = q_a[:, :2]
    basis_b = q_b[:, :2]

    # Endpoints
    b_0 = grassmann_geodesic(basis_a, basis_b, 0.0)
    b_1 = grassmann_geodesic(basis_a, basis_b, 1.0)
    np.testing.assert_allclose(np.dot(b_0.T, b_0), np.eye(2), atol=1e-6)
    np.testing.assert_allclose(np.dot(b_1.T, b_1), np.eye(2), atol=1e-6)

    # Orthonormality along full trajectory
    for tau in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        b_tau = grassmann_geodesic(basis_a, basis_b, tau)
        assert b_tau.shape == (6, 2)
        gram = np.dot(b_tau.T, b_tau)
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-6)


def test_interpolate_orthonormal_bases() -> None:
    """Test that interpolated bases preserve orthonormality V^T V = I_2."""
    rng = np.random.default_rng(42)
    raw_a = rng.normal(size=(5, 2))
    raw_b = rng.normal(size=(5, 2))

    basis_a, _ = np.linalg.qr(raw_a)
    basis_b, _ = np.linalg.qr(raw_b)

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        b_interp = interpolate_orthonormal_bases(basis_a[:, :2], basis_b[:, :2], alpha)
        assert b_interp.shape == (5, 2)
        gram = np.dot(b_interp.T, b_interp)
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-6)


def test_generate_grand_tour_path_shape() -> None:
    """Test Grand Tour frame count and dimension bounds."""
    rng = np.random.default_rng(42)
    matrix = rng.uniform(0.1, 1.0, size=(20, 4))
    frames, bases = generate_grand_tour_path(matrix, n_frames=60, seed=42)

    assert len(frames) > 0
    assert len(bases) == len(frames)
    assert len(frames[0]) == 20  # 20 samples
    assert len(frames[0][0]) == 2  # 2D coordinates

    # Verify orthonormality of every tour frame basis
    for b in bases:
        b_arr = np.array(b)
        assert b_arr.shape == (4, 2)
        gram = np.dot(b_arr.T, b_arr)
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-6)


def test_api_tour_path_endpoint(client: FlaskClient) -> None:
    """Test GET /api/tour-path endpoint."""
    res = client.get("/api/tour-path?dataset=calibration_3class&n_frames=60")
    assert res.status_code == 200
    data = json.loads(res.data)

    assert data["dataset"] == "calibration_3class"
    assert data["semantically_valid"] is True
    assert data["kind"] == "linear_projection"
    assert data["geodesic_algorithm"] == "GLERP"
    assert "frames" in data
    assert "bases" in data
    assert len(data["frames"]) > 0
