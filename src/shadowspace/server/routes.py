"""shadowspace.server.routes — Workbench page and data API routes.

Sprint 3b: Full workbench with PCA scatter, integrity panels, and saved-view atlas.
Sprint 4: Local integrity diagnostics API endpoint.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray
from flask import Blueprint, Response, render_template, request

from shadowspace.data.calibration import calibration_fixture
from shadowspace.diagnostics.knn import compute_point_diagnostics
from shadowspace.diagnostics.trustworthiness import (
    compute_kruskal_stress,
    compute_view_continuity,
    compute_view_trustworthiness,
)
from shadowspace.math.metrics import pairwise_euclidean
from shadowspace.math.registry import MetricRegistry
from shadowspace.math.transforms import sqrt_transform
from shadowspace.projection.basis import project
from shadowspace.projection.pca import fit_representation_pca

workbench_bp = Blueprint("workbench", __name__)
_METRIC_REGISTRY = MetricRegistry()

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

    representations: dict[str, NDArray[np.float64]] = {
        "probability": matrix,
        "sqrt_probability": sqrt_transform(matrix),
    }

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
        json.dumps({"status": "ok", "sprint": "4"}),
        mimetype="application/json",
    )


@workbench_bp.route("/api/diagnostics")
def api_diagnostics() -> Response:
    """Compute local integrity diagnostics for a given target point, k, and metric.

    Query parameters:
        target_id: Target object ID string (default 'corner_0').
        representation: Representation ID (default 'probability').
        metric: Metric ID (default 'euclidean').
        k: Neighborhood size (default 3).
    """
    target_id = request.args.get("target_id", "corner_0")
    rep_id = request.args.get("representation", "probability")
    metric_id = request.args.get("metric", "euclidean")

    try:
        k = int(request.args.get("k", 3))
    except ValueError:
        return Response(json.dumps({"error": "Invalid k parameter"}), status=400, mimetype="application/json")

    if rep_id not in _FIXTURE_DATA["representations"]:
        return Response(json.dumps({"error": f"Unknown representation {rep_id!r}"}), status=400, mimetype="application/json")

    object_ids = _FIXTURE_DATA["object_ids"]
    if target_id not in object_ids:
        return Response(json.dumps({"error": f"Unknown target_id {target_id!r}"}), status=400, mimetype="application/json")

    raw_matrix = np.array(_FIXTURE_DATA["raw_matrix"], dtype=np.float64)
    if rep_id == "sqrt_probability":
        rep_matrix = sqrt_transform(raw_matrix)
    else:
        rep_matrix = raw_matrix

    # High-D pairwise distance matrix under specified metric
    try:
        src_dists = _METRIC_REGISTRY.compute_pairwise_distances(rep_matrix, metric_id, rep_id)
    except (KeyError, ValueError) as err:
        return Response(json.dumps({"error": str(err)}), status=400, mimetype="application/json")

    # 2D projected coordinates and 2D Euclidean distances
    coords_2d = np.array(_FIXTURE_DATA["representations"][rep_id]["coords"], dtype=np.float64)
    proj_dists = pairwise_euclidean(coords_2d)

    # Compute point diagnostics
    diag = compute_point_diagnostics(src_dists, proj_dists, k, object_ids, target_id)

    # Compute view global metrics
    trustworthiness = compute_view_trustworthiness(src_dists, proj_dists, k)
    continuity = compute_view_continuity(src_dists, proj_dists, k)
    stress = compute_kruskal_stress(src_dists, proj_dists)

    payload = {
        "target_id": target_id,
        "representation": rep_id,
        "metric": metric_id,
        "k": k,
        "preserved": diag.preserved,
        "torn": diag.torn,
        "false_neighbors": diag.false_neighbors,
        "precision": round(diag.precision, 4),
        "recall": round(diag.recall, 4),
        "jaccard_overlap": round(diag.jaccard_overlap, 4),
        "trustworthiness": round(trustworthiness, 4),
        "continuity": round(continuity, 4),
        "stress": round(stress, 4),
    }

    return Response(json.dumps(payload), mimetype="application/json")
