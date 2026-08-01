"""Tests for Sprint 5 — Investigation record and Saved-View Atlas."""

from __future__ import annotations

import json

import pytest
from flask.testing import FlaskClient

from shadowspace.models.investigation import InvestigationRecord, SavedView
from shadowspace.server import create_app


@pytest.fixture()
def client() -> FlaskClient:
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_saved_view_model_defaults() -> None:
    view = SavedView(
        id="view_1",
        name="Test View",
        representation_id="probability",
        metric_id="euclidean",
        k=3,
        target_id="corner_0",
    )
    assert view.id == "view_1"
    assert view.name == "Test View"
    assert view.semantically_valid is True
    assert view.path_kind == "linear_projection"
    assert view.timestamp is not None


def test_investigation_record_model() -> None:
    view = SavedView(
        id="v1",
        name="View 1",
        representation_id="probability",
        metric_id="euclidean",
        k=3,
        target_id="corner_0",
    )
    record = InvestigationRecord(
        bundle_id="b1",
        saved_views=[view],
        artifact_hashes={"key": "hash123"},
    )
    assert record.schema_version == "0.1.0"
    assert len(record.saved_views) == 1
    assert record.artifact_hashes["key"] == "hash123"


def test_api_saved_views_crud(client: FlaskClient) -> None:
    # 1. GET initially empty or list
    res = client.get("/api/saved-views")
    assert res.status_code == 200
    initial_views = json.loads(res.data)
    assert isinstance(initial_views, list)

    # 2. POST create new view
    payload = {
        "name": "Corner 0 Baseline",
        "note": "Initial investigation point",
        "representation_id": "probability",
        "metric_id": "euclidean",
        "k": 3,
        "target_id": "corner_0",
    }
    res_post = client.post("/api/saved-views", json=payload)
    assert res_post.status_code == 201
    saved_data = json.loads(res_post.data)
    assert saved_data["name"] == "Corner 0 Baseline"
    view_id = saved_data["id"]

    # 3. GET verify view listed
    res_get = client.get("/api/saved-views")
    views_after_post = json.loads(res_get.data)
    assert any(v["id"] == view_id for v in views_after_post)

    # 4. DELETE view
    res_del = client.delete(f"/api/saved-views?id={view_id}")
    assert res_del.status_code == 200

    # 5. GET verify view removed
    res_final = client.get("/api/saved-views")
    views_final = json.loads(res_final.data)
    assert not any(v["id"] == view_id for v in views_final)


def test_api_saved_views_validation(client: FlaskClient) -> None:
    # Invalid k
    res1 = client.post("/api/saved-views", json={"k": "invalid_k"})
    assert res1.status_code == 400

    # Unknown representation
    res2 = client.post("/api/saved-views", json={"representation_id": "unknown_rep"})
    assert res2.status_code == 400

    # Unknown target_id
    res3 = client.post("/api/saved-views", json={"target_id": "unknown_id"})
    assert res3.status_code == 400


def test_api_export_record(client: FlaskClient) -> None:
    res = client.get("/api/export-record")
    assert res.status_code == 200
    assert "attachment; filename=investigation_record.json" in res.headers["Content-Disposition"]

    data = json.loads(res.data)
    assert data["schema_version"] == "0.1.0"
    assert "saved_views" in data
    assert "artifact_hashes" in data
    assert "calibration_matrix_sha256" in data["artifact_hashes"]
    assert "object_ids_fit_hash" in data["artifact_hashes"]
    assert "feature_schema_hash" in data["artifact_hashes"]
    assert "created_at" in data
