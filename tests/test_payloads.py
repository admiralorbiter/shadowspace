"""Tests for Sprint 9 — Object Image Payloads and Investigation Report Export API."""

from __future__ import annotations

import json

import pytest
from flask.testing import FlaskClient

from shadowspace.server import create_app


@pytest.fixture()
def client() -> FlaskClient:
    from pathlib import Path
    from shadowspace.datasets.fetchers.sklearn_datasets import fetch_dataset

    bundle_path = Path("data/bundles/digits_10class")
    if not bundle_path.exists():
        fetch_dataset("digits_10class", "data/bundles", seed=42)

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_api_object_payload_digits(client: FlaskClient) -> None:
    """Test object payload endpoint for digits_10class dataset."""
    res = client.get("/api/object-payload?dataset=digits_10class&target_id=digits_10class_00000")
    assert res.status_code == 200
    assert res.content_type == "image/png"
    assert len(res.data) > 0


def test_api_object_payload_unknown_id(client: FlaskClient) -> None:
    """Test object payload endpoint for nonexistent target_id."""
    res = client.get("/api/object-payload?dataset=calibration_3class&target_id=nonexistent_id")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["has_image"] is False


def test_api_export_report(client: FlaskClient) -> None:
    """Test investigation report export endpoint."""
    payload = {
        "dataset": "digits_10class",
        "target_id": "digits_10class_00000",
        "representation": "probability",
        "metric": "fisher_rao",
        "view_id": "pca_corners",
        "k": 5,
        "saved_views": [
            {
                "id": "v1",
                "name": "Digit 0 Baseline",
                "representation_id": "probability",
                "metric_id": "fisher_rao",
                "k": 5,
                "target_id": "digits_10class_00000",
                "note": "Initial investigation view",
            }
        ],
    }
    res = client.post("/api/export-report", json=payload)
    assert res.status_code == 200
    assert "text/markdown" in res.content_type
    report_text = res.data.decode("utf-8")
    assert "# Shadowspace Investigation Record" in report_text
    assert "digits_10class" in report_text
    assert "Digit 0 Baseline" in report_text
