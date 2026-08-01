"""Tests for Sprint 3b & 4 — Flask workbench routes and diagnostics API."""

from __future__ import annotations

import json

import pytest
from flask.testing import FlaskClient

from shadowspace.server import create_app


@pytest.fixture()
def client() -> FlaskClient:
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_workbench_returns_200(client: FlaskClient) -> None:
    res = client.get("/")
    assert res.status_code == 200


def test_workbench_html_contains_panel_ids(client: FlaskClient) -> None:
    res = client.get("/")
    html = res.data.decode("utf-8")
    assert "panel-integrity-overlay" in html
    assert "panel-semantic-badge"    in html
    assert "panel-source-inspector"  in html
    assert "panel-saved-view-atlas"  in html


def test_workbench_html_contains_canvas(client: FlaskClient) -> None:
    res = client.get("/")
    html = res.data.decode("utf-8")
    assert "scatter-canvas" in html


def test_api_health(client: FlaskClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["status"] == "ok"
    assert data["sprint"] == "6"


def test_api_fixture_returns_200(client: FlaskClient) -> None:
    res = client.get("/api/fixture")
    assert res.status_code == 200
    assert res.content_type == "application/json"


def test_api_fixture_structure(client: FlaskClient) -> None:
    res = client.get("/api/fixture")
    data = json.loads(res.data)

    assert "representations" in data
    assert "probability"      in data["representations"]
    assert "sqrt_probability" in data["representations"]
    assert "object_ids"       in data
    assert "colors"           in data
    assert "raw_matrix"       in data


def test_api_fixture_object_count(client: FlaskClient) -> None:
    res = client.get("/api/fixture")
    data = json.loads(res.data)

    assert len(data["object_ids"]) == 15
    assert len(data["colors"])     == 15
    assert len(data["raw_matrix"]) == 15


def test_api_fixture_coords_shape(client: FlaskClient) -> None:
    res = client.get("/api/fixture")
    data = json.loads(res.data)

    for rep_id in ("probability", "sqrt_probability"):
        coords = data["representations"][rep_id]["coords"]
        assert len(coords) == 15, f"{rep_id}: expected 15 points"
        assert all(len(c) == 2 for c in coords), f"{rep_id}: each coord must be 2D"


def test_api_fixture_eigenvalues_present(client: FlaskClient) -> None:
    res = client.get("/api/fixture")
    data = json.loads(res.data)

    for rep_id in ("probability", "sqrt_probability"):
        evs = data["representations"][rep_id]["eigenvalues"]
        assert len(evs) == 2
        assert all(v >= 0 for v in evs)


def test_api_diagnostics_default(client: FlaskClient) -> None:
    res = client.get("/api/diagnostics")
    assert res.status_code == 200
    data = json.loads(res.data)

    assert data["target_id"] == "corner_0"
    assert data["representation"] == "probability"
    assert data["metric"] == "euclidean"
    assert data["k"] == 3
    assert "preserved" in data
    assert "torn" in data
    assert "false_neighbors" in data
    assert "precision" in data
    assert "recall" in data
    assert "jaccard_overlap" in data
    assert "trustworthiness" in data
    assert "continuity" in data
    assert "stress" in data


def test_api_diagnostics_query_params(client: FlaskClient) -> None:
    res = client.get("/api/diagnostics?target_id=center&representation=probability&metric=euclidean&k=4")
    assert res.status_code == 200
    data = json.loads(res.data)

    assert data["target_id"] == "center"
    assert data["k"] == 4
    assert len(data["preserved"]) + len(data["torn"]) == 4


def test_api_diagnostics_invalid_params(client: FlaskClient) -> None:
    # Invalid target_id
    res = client.get("/api/diagnostics?target_id=unknown_id")
    assert res.status_code == 400

    # Invalid representation
    res = client.get("/api/diagnostics?representation=unknown_rep")
    assert res.status_code == 400

    # Incompatible metric
    res = client.get("/api/diagnostics?metric=hellinger&representation=sqrt_probability")
    assert res.status_code == 400
