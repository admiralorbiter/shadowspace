"""shadowspace.server.routes — Workbench page and data API routes.

Sprint 3b: Full workbench with PCA scatter, integrity panels, and saved-view atlas.
Sprint 4: Local integrity diagnostics API endpoint.
Sprint 5: Saved-View Atlas and Investigation Record export.
Sprint 6: Four-Class Validation, Dataset Switcher, and Projection Catalog.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
from typing import Any

import numpy as np
from flask import Blueprint, Response, render_template, request
from numpy.typing import NDArray

from shadowspace.bundle.reader import BundleReader
from shadowspace.data.calibration import calibration_fixture
from shadowspace.diagnostics.knn import compute_point_diagnostics
from shadowspace.diagnostics.trustworthiness import (
    compute_kruskal_stress,
    compute_view_continuity,
    compute_view_trustworthiness,
)
from shadowspace.generators.synthetic import generate_synthetic_bundle
from shadowspace.math.metrics import pairwise_euclidean
from shadowspace.math.registry import MetricRegistry
from shadowspace.math.transforms import sqrt_transform
from shadowspace.models.investigation import InvestigationRecord, SavedView
from shadowspace.projection.basis import project
from shadowspace.projection.catalog import build_projection_catalog
from shadowspace.projection.pca import (
    compute_feature_schema_hash,
    compute_object_id_hash,
    fit_representation_pca,
)

workbench_bp = Blueprint("workbench", __name__)
_METRIC_REGISTRY = MetricRegistry()
_SAVED_VIEWS: dict[str, SavedView] = {}

# ---------------------------------------------------------------------------
# Fixture and Synthetic Data Pre-Computation (Sprint 6)
# ---------------------------------------------------------------------------

_POINT_COLORS_3CLASS = {
    "corner": "#6ee7b7",
    "midpoint": "#93c5fd",
    "center": "#fbbf24",
    "interior": "#c4b5fd",
}


def _point_color_3class(object_id: str) -> str:
    for prefix, color in _POINT_COLORS_3CLASS.items():
        if object_id.startswith(prefix):
            return color
    return "#e2e8f0"


def _point_color_4class(object_id: str) -> str:
    if "corner" in object_id or "synth_0000" in object_id:
        return "#6ee7b7"
    if "bridge" in object_id:
        return "#f87171"
    if "center" in object_id:
        return "#fbbf24"
    if "outlier" in object_id:
        return "#f472b6"
    return "#93c5fd"


def _build_dataset_entry(
    matrix: NDArray[np.float64],
    object_ids: list[str],
    feature_names: list[str],
    display_name: str,
    color_fn: Any,
) -> dict[str, Any]:
    """Build dataset dictionary with pre-computed representations and catalog."""
    representations: dict[str, NDArray[np.float64]] = {
        "probability": matrix,
        "sqrt_probability": sqrt_transform(matrix),
    }

    rep_data: dict[str, Any] = {}

    for rep_id, mat in representations.items():
        basis, view = fit_representation_pca(
            mat,
            rep_id,
            object_ids,
            feature_names,
            f"pca_{rep_id}",
        )
        coords_2d = project(mat, basis)

        rep_data[rep_id] = {
            "coords": coords_2d.tolist(),
            "eigenvalues": view.provenance["eigenvalues"],
            "basis": view.provenance["basis"],
            "centering_policy": view.provenance["centering_policy"],
        }

    # Build Projection Catalog for probability representation
    catalog_views = build_projection_catalog(matrix, object_ids, feature_names)
    catalog_payload: dict[str, Any] = {}

    for vid, cat_view in catalog_views.items():
        proj_coords = project(matrix, cat_view.basis)
        catalog_payload[vid] = {
            "view_id": cat_view.view_id,
            "display_name": cat_view.display_name,
            "basis": cat_view.basis.tolist(),
            "coords": proj_coords.tolist(),
            "semantically_valid": cat_view.semantically_valid,
            "is_misleading": cat_view.is_misleading,
            "description": cat_view.description,
            "warning_note": cat_view.warning_note,
        }

    colors = [color_fn(oid) for oid in object_ids]

    return {
        "display_name": display_name,
        "feature_names": feature_names,
        "object_ids": object_ids,
        "colors": colors,
        "raw_matrix": matrix.tolist(),
        "representations": rep_data,
        "catalog": catalog_payload,
    }


def _initialize_datasets() -> dict[str, dict[str, Any]]:
    """Initialize 3-class calibration and 4-class synthetic datasets."""
    # 1. 3-Class Calibration Fixture
    m3, ids3 = calibration_fixture()
    ds3 = _build_dataset_entry(
        matrix=m3,
        object_ids=ids3,
        feature_names=["class_0", "class_1", "class_2"],
        display_name="3-Class Calibration Fixture (15 pts)",
        color_fn=_point_color_3class,
    )

    # 2. 4-Class Synthetic World
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_synthetic_bundle(output_dir=tmpdir, seed=42, n_samples=100)
        reader = BundleReader(tmpdir)
        m4, ids4 = reader.get_representation_matrix("probability")

    ds4 = _build_dataset_entry(
        matrix=m4,
        object_ids=ids4,
        feature_names=["class_0", "class_1", "class_2", "class_3"],
        display_name="4-Class Synthetic World (98 pts)",
        color_fn=_point_color_4class,
    )

    return {
        "calibration_3class": ds3,
        "synthetic_4class": ds4,
    }


_DATASETS = _initialize_datasets()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@workbench_bp.route("/")
def workbench() -> str:
    """Main workbench page."""
    return render_template("workbench.html")


@workbench_bp.route("/api/fixture")
def api_fixture() -> Response:
    """Return dataset fixture data (defaults to calibration_3class)."""
    dataset_key = request.args.get("dataset", "calibration_3class")
    if dataset_key not in _DATASETS:
        dataset_key = "calibration_3class"

    return Response(
        json.dumps(_DATASETS[dataset_key], separators=(",", ":")),
        mimetype="application/json",
    )


@workbench_bp.route("/api/health")
def api_health() -> Response:
    """Health check endpoint."""
    return Response(
        json.dumps({"status": "ok", "sprint": "6"}),
        mimetype="application/json",
    )


@workbench_bp.route("/api/diagnostics")
def api_diagnostics() -> Response:
    """Compute local integrity diagnostics for a target point, k, metric, and view catalog."""
    dataset_key = request.args.get("dataset", "calibration_3class")
    if dataset_key not in _DATASETS:
        dataset_key = "calibration_3class"

    ds_data = _DATASETS[dataset_key]
    object_ids = ds_data["object_ids"]

    target_id = request.args.get("target_id", object_ids[0])
    rep_id = request.args.get("representation", "probability")
    metric_id = request.args.get("metric", "euclidean")
    view_id = request.args.get("view_id", "pca_corners")

    try:
        k = int(request.args.get("k", 3))
    except ValueError:
        return Response(json.dumps({"error": "Invalid k parameter"}), status=400, mimetype="application/json")

    if rep_id not in ds_data["representations"]:
        return Response(json.dumps({"error": f"Unknown representation {rep_id!r}"}), status=400, mimetype="application/json")

    if target_id not in object_ids:
        return Response(json.dumps({"error": f"Unknown target_id {target_id!r}"}), status=400, mimetype="application/json")

    raw_matrix = np.array(ds_data["raw_matrix"], dtype=np.float64)
    if rep_id == "sqrt_probability":
        rep_matrix = sqrt_transform(raw_matrix)
    else:
        rep_matrix = raw_matrix

    try:
        src_dists = _METRIC_REGISTRY.compute_pairwise_distances(rep_matrix, metric_id, rep_id)
    except (KeyError, ValueError) as err:
        return Response(json.dumps({"error": str(err)}), status=400, mimetype="application/json")

    # Select 2D projected coordinates: catalog view basis if requested, else PCA basis
    if view_id in ds_data.get("catalog", {}):
        coords_2d = np.array(ds_data["catalog"][view_id]["coords"], dtype=np.float64)
    else:
        coords_2d = np.array(ds_data["representations"][rep_id]["coords"], dtype=np.float64)

    proj_dists = pairwise_euclidean(coords_2d)

    diag = compute_point_diagnostics(src_dists, proj_dists, k, object_ids, target_id)
    trustworthiness = compute_view_trustworthiness(src_dists, proj_dists, k)
    continuity = compute_view_continuity(src_dists, proj_dists, k)
    stress = compute_kruskal_stress(src_dists, proj_dists)

    payload = {
        "dataset": dataset_key,
        "target_id": target_id,
        "representation": rep_id,
        "metric": metric_id,
        "view_id": view_id,
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


# ---------------------------------------------------------------------------
# Saved Views & Investigation Record API (Sprint 5)
# ---------------------------------------------------------------------------


@workbench_bp.route("/api/saved-views", methods=["GET", "POST", "DELETE"])
def api_saved_views() -> Response:
    """GET list of saved views, POST new saved view, or DELETE view by ID."""
    if request.method == "GET":
        views_list = [v.model_dump(mode="json") for v in _SAVED_VIEWS.values()]
        return Response(json.dumps(views_list), mimetype="application/json")

    elif request.method == "POST":
        data = request.get_json(silent=True) or {}
        view_name = str(data.get("name", "Untitled View")).strip() or "Untitled View"
        note = str(data.get("note", "")).strip()
        rep_id = str(data.get("representation_id", "probability"))
        metric_id = str(data.get("metric_id", "euclidean"))
        dataset_key = str(data.get("dataset", "calibration_3class"))
        target_id = str(data.get("target_id", "corner_0"))

        try:
            k = int(data.get("k", 3))
        except (TypeError, ValueError):
            return Response(json.dumps({"error": "Invalid k parameter"}), status=400, mimetype="application/json")

        ds_data = _DATASETS.get(dataset_key, _DATASETS["calibration_3class"])
        if rep_id not in ds_data["representations"]:
            return Response(json.dumps({"error": f"Unknown representation {rep_id!r}"}), status=400, mimetype="application/json")
        if target_id not in ds_data["object_ids"]:
            return Response(json.dumps({"error": f"Unknown target_id {target_id!r}"}), status=400, mimetype="application/json")

        view_id = f"view_{uuid.uuid4().hex[:8]}"
        saved = SavedView(
            id=view_id,
            name=view_name,
            note=note,
            representation_id=rep_id,
            metric_id=metric_id,
            k=k,
            target_id=target_id,
            variance_explained=ds_data["representations"].get(rep_id, {}).get("eigenvalues", [0.5, 0.5]),
            metadata={"dataset": dataset_key},
        )
        _SAVED_VIEWS[view_id] = saved
        return Response(json.dumps(saved.model_dump(mode="json")), status=201, mimetype="application/json")

    elif request.method == "DELETE":
        view_id = request.args.get("id", "")
        if view_id in _SAVED_VIEWS:
            del _SAVED_VIEWS[view_id]
            return Response(json.dumps({"status": "deleted", "id": view_id}), mimetype="application/json")
        return Response(json.dumps({"error": f"View {view_id!r} not found"}), status=404, mimetype="application/json")

    return Response(json.dumps({"error": "Method not allowed"}), status=405, mimetype="application/json")


@workbench_bp.route("/api/export-record")
def api_export_record() -> Response:
    """Export complete InvestigationRecord JSON with SHA-256 provenance hashes."""
    matrix = np.array(_DATASETS["calibration_3class"]["raw_matrix"], dtype=np.float64)
    object_ids = _DATASETS["calibration_3class"]["object_ids"]
    feature_names = _DATASETS["calibration_3class"]["feature_names"]

    matrix_hash = hashlib.sha256(matrix.tobytes()).hexdigest()
    object_hash = compute_object_id_hash(object_ids)
    feature_hash = compute_feature_schema_hash(feature_names)

    record = InvestigationRecord(
        bundle_id="calibration_fixture_v1",
        saved_views=list(_SAVED_VIEWS.values()),
        artifact_hashes={
            "calibration_matrix_sha256": matrix_hash,
            "object_ids_fit_hash": object_hash,
            "feature_schema_hash": feature_hash,
        },
        summary_note="Investigation record exported from Shadowspace Sprint 6 workbench shell.",
    )
    headers = {"Content-Disposition": "attachment; filename=investigation_record.json"}
    return Response(
        json.dumps(record.model_dump(mode="json"), indent=2),
        mimetype="application/json",
        headers=headers,
    )
