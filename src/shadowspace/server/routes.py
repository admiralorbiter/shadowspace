"""shadowspace.server.routes — Workbench page and data API routes.

Sprint 3b: Full workbench with PCA scatter, integrity panels, and saved-view atlas.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from flask import Blueprint, Response, render_template

from shadowspace.data.calibration import calibration_fixture
from shadowspace.math.transforms import clr_transform, logit_transform, sqrt_transform
from shadowspace.projection.basis import project
from shadowspace.projection.pca import fit_representation_pca

workbench_bp = Blueprint("workbench", __name__)

# ---------------------------------------------------------------------------
# Fixture pre-computation (done once at import time)
# ---------------------------------------------------------------------------

_FEATURE_NAMES = ["class_0", "class_1", "class_2"]
_POINT_COLORS = {
    "corner": "#6ee7b7",
    "midpoint": "#93c5fd",
    "center": "#fbbf24",
    "interior": "#c4b5fd",
}


def _point_color(object_id: str) -> str:
    for prefix, color in _POINT_COLORS.items():
        if object_id.startswith(prefix):
            return color
    return "#e2e8f0"


def _build_fixture_data() -> dict[str, Any]:
    """Pre-compute PCA projections for all representations."""
    matrix, object_ids = calibration_fixture()

    representations: dict[str, np.ndarray] = {
        "probability": matrix,
        "sqrt_probability": sqrt_transform(matrix),
    }

    # CLR and logit require zero-handling — use only the two safe reps for now
    result: dict[str, Any] = {"representations": {}, "object_ids": object_ids}

    colors = [_point_color(oid) for oid in object_ids]

    for rep_id, mat in representations.items():
        basis, view = fit_representation_pca(
            mat,
            rep_id,
            object_ids,
            _FEATURE_NAMES,
            f"pca_{rep_id}",
        )
        coords_2d = project(mat, basis)

        result["representations"][rep_id] = {
            "coords": coords_2d.tolist(),
            "eigenvalues": view.provenance["eigenvalues"],
            "basis": view.provenance["basis"],
            "centering_policy": view.provenance["centering_policy"],
        }

    result["colors"] = colors
    result["raw_matrix"] = matrix.tolist()
    return result


_FIXTURE_DATA = _build_fixture_data()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@workbench_bp.route("/")
def workbench() -> str:
    """Main workbench page."""
    return render_template("workbench.html")


@workbench_bp.route("/api/fixture")
def api_fixture() -> Response:
    """Return the pre-computed calibration fixture data as JSON."""
    return Response(
        json.dumps(_FIXTURE_DATA, separators=(",", ":")),
        mimetype="application/json",
    )


@workbench_bp.route("/api/health")
def api_health() -> Response:
    """Health check endpoint."""
    return Response(
        json.dumps({"status": "ok", "sprint": "3b"}),
        mimetype="application/json",
    )
