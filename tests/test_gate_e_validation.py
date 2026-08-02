"""Unit tests for Gate E developer validation and roadmap realignment."""

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


def test_api_health_endpoint(client: FlaskClient) -> None:
    """Verify /api/health endpoint returns system status and milestone completion."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["status"] == "ok"
    assert data["sqlite_vec_enabled"] is True
    assert "hardening_milestones" in data
    assert data["hardening_milestones"]["gate_a"] == "complete"
    assert data["hardening_milestones"]["gate_b"] == "complete"
    assert data["hardening_milestones"]["gate_c"] == "complete"
    assert data["hardening_milestones"]["gate_d"] == "complete"
    assert data["hardening_milestones"]["gate_e"] == "complete"
