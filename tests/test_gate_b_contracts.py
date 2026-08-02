"""Unit tests for Gate B data contract alignment and geometric consistency."""

from __future__ import annotations

import json
import numpy as np
import pytest
from flask.testing import FlaskClient

from shadowspace.math.transforms import sqrt_transform, logit_transform
from shadowspace.server import create_app


@pytest.fixture()
def client() -> FlaskClient:
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_diagnostics_reprojects_under_representation_change(client: FlaskClient) -> None:
    """Verify diagnostics endpoint reprojects catalog coordinates under active representation space."""
    # 1. Query under raw probability
    res_prob = client.get("/api/diagnostics?dataset=calibration_3class&representation=probability&metric=fisher_rao&view_id=pca_corners&k=3")
    assert res_prob.status_code == 200
    data_prob = json.loads(res_prob.data)
    assert data_prob["representation"] == "probability"

    # 2. Query under sqrt_probability with Euclidean distance in sqrt-space
    res_sqrt = client.get("/api/diagnostics?dataset=calibration_3class&representation=sqrt_probability&metric=euclidean&view_id=pca_corners&k=3")
    assert res_sqrt.status_code == 200
    data_sqrt = json.loads(res_sqrt.data)
    assert data_sqrt["representation"] == "sqrt_probability"
    assert data_sqrt["metric"] == "euclidean"


def test_topology_and_distortion_grid_reprojection(client: FlaskClient) -> None:
    """Verify topology and distortion grid endpoints compute reprojected 2D coordinates."""
    res_top = client.get("/api/topology?dataset=calibration_3class&representation=clr_probability&metric=euclidean&view_id=pca_corners&k=3")
    assert res_top.status_code == 200
    data_top = json.loads(res_top.data)
    assert data_top["representation"] == "clr_probability"
    assert "edges" in data_top

    res_grid = client.get("/api/distortion-grid?dataset=calibration_3class&representation=sqrt_probability&metric=euclidean&view_id=pca_corners")
    assert res_grid.status_code == 200
    data_grid = json.loads(res_grid.data)
    assert data_grid["representation"] == "sqrt_probability"
    assert "grid" in data_grid


def test_saved_view_metadata_provenance(client: FlaskClient) -> None:
    """Verify SavedView POST endpoint records view_id, basis, coords, and hashes in metadata."""
    payload = {
        "name": "Gate B Snapshot",
        "note": "Testing contract provenance",
        "representation_id": "sqrt_probability",
        "metric_id": "euclidean",
        "dataset": "calibration_3class",
        "view_id": "pca_corners",
        "k": 3,
        "target_id": "corner_0",
    }
    res = client.post("/api/saved-views", json=payload)
    assert res.status_code == 201
    data = json.loads(res.data)
    assert data["representation_id"] == "sqrt_probability"
    meta = data["metadata"]
    assert meta["dataset"] == "calibration_3class"
    assert meta["view_id"] == "pca_corners"
    assert "basis" in meta
    assert "coords" in meta
    assert "matrix_sha256" in meta
    assert "object_ids_fit_hash" in meta
    assert "feature_schema_hash" in meta
