"""Unit tests for Gate C saved-investigation reproducibility and replay."""

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


def test_export_import_roundtrip(client: FlaskClient) -> None:
    """Test full export -> import record roundtrip with view restoration."""
    # 1. Create a saved view
    post_res = client.post("/api/saved-views", json={
        "name": "Gate C Test View",
        "representation_id": "probability",
        "metric_id": "euclidean",
        "k": 3,
        "target_id": "corner_0",
    })
    assert post_res.status_code == 201

    # 2. Export record
    export_res = client.get("/api/export-record?dataset=calibration_3class")
    assert export_res.status_code == 200
    record_data = json.loads(export_res.data)

    assert record_data["bundle_id"] == "calibration_3class"
    assert len(record_data["saved_views"]) >= 1

    # 3. Import record back
    import_res = client.post("/api/import-record", json=record_data)
    assert import_res.status_code == 200
    import_payload = json.loads(import_res.data)
    assert import_payload["status"] == "imported"
    assert import_payload["n_views_imported"] >= 1


def test_import_record_rejects_corrupted_hash(client: FlaskClient) -> None:
    """Test import rejects record payload if matrix SHA-256 hash does not match target dataset."""
    export_res = client.get("/api/export-record?dataset=calibration_3class")
    assert export_res.status_code == 200
    record_data = json.loads(export_res.data)

    # Corrupt the matrix SHA-256 hash
    record_data["artifact_hashes"]["calibration_matrix_sha256"] = "0" * 64
    record_data["artifact_hashes"]["calibration_3class_matrix_sha256"] = "0" * 64

    import_res = client.post("/api/import-record", json=record_data)
    assert import_res.status_code == 400
    err_data = json.loads(import_res.data)
    assert "hash mismatch" in err_data["error"]


def test_import_record_rejects_invalid_schema(client: FlaskClient) -> None:
    """Test import rejects invalid JSON or schema violations."""
    res1 = client.post("/api/import-record", json={"invalid_key": 123})
    assert res1.status_code == 400

    res2 = client.post("/api/import-record", data="not json", content_type="application/json")
    assert res2.status_code == 400
