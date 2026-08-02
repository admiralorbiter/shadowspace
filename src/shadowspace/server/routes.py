"""shadowspace.server.routes — Workbench page and data API routes.

Sprint 3b: Full workbench with PCA scatter, integrity panels, and saved-view atlas.
Sprint 4: Local integrity diagnostics API endpoint.
Sprint 5: Saved-View Atlas and Investigation Record export.
Sprint 6: Four-Class Validation, Dataset Switcher, and Projection Catalog.
Sprint 7: Fashion-MNIST 10-Class Prediction Belief Space & Metadata Inspector.
Sprint 8: Real Data Ingestion & Flask Workbench Expansion.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from flask import Blueprint, Response, render_template, request
from numpy.typing import NDArray

from shadowspace.bundle.reader import BundleReader
from shadowspace.data.calibration import calibration_fixture
from shadowspace.diagnostics.knn import compute_knn, compute_point_diagnostics
from shadowspace.diagnostics.trustworthiness import (
    compute_kruskal_stress,
    compute_view_continuity,
    compute_view_trustworthiness,
)
from shadowspace.generators.fashion_mnist import FASHION_CLASSES, generate_fashion_mnist_bundle
from shadowspace.generators.synthetic import generate_synthetic_bundle
from shadowspace.importers.csv_importer import import_csv_bundle, import_parquet_bundle
from shadowspace.math.clr import clr_transform
from shadowspace.math.metrics import pairwise_euclidean
from shadowspace.math.registry import MetricRegistry
from shadowspace.math.stability import compute_point_stability, generate_rashomon_set
from shadowspace.math.subspace_angles import compute_canonical_angles, compute_grassmannian_distance
from shadowspace.math.transforms import sqrt_transform
from shadowspace.models.investigation import InvestigationRecord, SavedView
from shadowspace.projection.basis import project, validate_orthonormal_basis
from shadowspace.projection.catalog import build_projection_catalog
from shadowspace.projection.paths import generate_grand_tour_path, grassmann_geodesic
from shadowspace.projection.pca import (
    compute_feature_schema_hash,
    compute_object_id_hash,
    fit_representation_pca,
)
from shadowspace.projection.subspace import (
    find_discriminative_basis,
    find_integrity_optimal_basis,
    grassmannian_distance,
)

workbench_bp = Blueprint("workbench", __name__)
_METRIC_REGISTRY = MetricRegistry()
_SAVED_VIEWS: dict[str, SavedView] = {}

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB limit

# ---------------------------------------------------------------------------
# Fixture and Dataset Generators
# ---------------------------------------------------------------------------

_POINT_COLORS_3CLASS = {
    "corner": "#6ee7b7",
    "midpoint": "#93c5fd",
    "center": "#fbbf24",
    "interior": "#c4b5fd",
}

_FASHION_COLOR_PALETTE = [
    "#6ee7b7",  # 0: T-shirt/top
    "#93c5fd",  # 1: Trouser
    "#c4b5fd",  # 2: Pullover
    "#f472b6",  # 3: Dress
    "#fbbf24",  # 4: Coat
    "#a7f3d0",  # 5: Sandal
    "#f87171",  # 6: Shirt
    "#60a5fa",  # 7: Sneaker
    "#e879f9",  # 8: Bag
    "#38bdf8",  # 9: Ankle boot
]


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


def _point_color_fashion(object_id: str) -> str:
    try:
        idx = int(object_id.split("_")[-1])
        c = (idx // 20) % 10
        return _FASHION_COLOR_PALETTE[c]
    except Exception:
        return "#6ee7b7"


def _point_color_generic(object_id: str) -> str:
    # Hash object_id to a deterministic vibrant HSL color
    h = int(hashlib.md5(object_id.encode("utf-8")).hexdigest(), 16) % 360
    return f"hsl({h}, 70%, 65%)"


def _compute_class_colors(
    matrix: NDArray[np.float64],
    feature_names: list[str],
    objects_meta: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Compute class-matched colors based on true_label or highest probability."""
    palette = [
        "#6ee7b7", "#93c5fd", "#c4b5fd", "#f472b6", "#fbbf24",
        "#a7f3d0", "#f87171", "#60a5fa", "#e879f9", "#38bdf8"
    ]
    colors = []
    for i, row in enumerate(matrix):
        cls_idx = int(np.argmax(row))
        if objects_meta and i < len(objects_meta):
            meta = objects_meta[i]
            label = meta.get("true_label") or meta.get("true_class_name")
            if label is not None:
                str_label = str(label)
                for fn_i, fn in enumerate(feature_names):
                    clean_fn = fn.replace("p_", "")
                    if fn == str_label or clean_fn == str_label or f"p_{str_label}" == fn:
                        cls_idx = fn_i
                        break
        colors.append(palette[cls_idx % len(palette)])
    return colors


def _build_dataset_entry(
    matrix: NDArray[np.float64],
    object_ids: list[str],
    feature_names: list[str],
    display_name: str,
    color_fn: Any,
    objects_meta: list[dict[str, Any]] | None = None,
    source_type: str = "synthetic",
) -> dict[str, Any]:
    """Build dataset dictionary with pre-computed representations and catalog."""
    representations: dict[str, NDArray[np.float64]] = {
        "probability": matrix,
        "sqrt_probability": sqrt_transform(matrix),
        "clr_probability": clr_transform(matrix),
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

    payload_bytes: dict[str, bytes] = {}
    clean_objects_meta: list[dict[str, Any]] = []

    if objects_meta:
        for meta in objects_meta:
            m_copy = dict(meta)
            img_b = m_copy.pop("payload_image_bytes", None)
            if img_b is not None:
                if isinstance(img_b, bytes):
                    payload_bytes[m_copy["object_id"]] = img_b
                elif isinstance(img_b, str):
                    import base64

                    try:
                        payload_bytes[m_copy["object_id"]] = base64.b64decode(img_b)
                    except Exception:
                        pass
            clean_objects_meta.append(m_copy)

    if color_fn in (None, _point_color_generic):
        colors = _compute_class_colors(matrix, feature_names, clean_objects_meta)
    else:
        colors = [color_fn(oid) for oid in object_ids]

    return {
        "display_name": display_name,
        "feature_names": feature_names,
        "object_ids": object_ids,
        "colors": colors,
        "raw_matrix": matrix.tolist(),
        "representations": rep_data,
        "catalog": catalog_payload,
        "objects_meta": clean_objects_meta,
        "payload_bytes": payload_bytes,
        "has_payloads": len(payload_bytes) > 0,
        "source_type": source_type,
        "n_objects": len(object_ids),
        "n_classes": len(feature_names),
    }


# Cache for datasets
_DATASETS: dict[str, dict[str, Any]] = {}


def _get_dataset(key: str) -> dict[str, Any] | None:
    """Lazy loader for built-in and imported datasets."""
    if key in _DATASETS:
        return _DATASETS[key]

    if key == "calibration_3class":
        m3, ids3 = calibration_fixture()
        _DATASETS["calibration_3class"] = _build_dataset_entry(
            matrix=m3,
            object_ids=ids3,
            feature_names=["class_0", "class_1", "class_2"],
            display_name="3-Class Calibration Fixture (15 pts)",
            color_fn=_point_color_3class,
            source_type="synthetic",
        )
        return _DATASETS["calibration_3class"]

    elif key == "synthetic_4class":
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_synthetic_bundle(output_dir=tmpdir, seed=42, n_samples=100)
            reader4 = BundleReader(tmpdir)
            m4, ids4 = reader4.get_representation_matrix("probability")

        _DATASETS["synthetic_4class"] = _build_dataset_entry(
            matrix=m4,
            object_ids=ids4,
            feature_names=["class_0", "class_1", "class_2", "class_3"],
            display_name="4-Class Synthetic World (98 pts)",
            color_fn=_point_color_4class,
            source_type="synthetic",
        )
        return _DATASETS["synthetic_4class"]

    elif key == "fashion_mnist_10class":
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_fashion_mnist_bundle(output_dir=tmpdir, seed=20260801, n_samples=200)
            reader_f = BundleReader(tmpdir)
            mf, idsf = reader_f.get_representation_matrix("probability")
            objects_meta = reader_f.get_objects().to_dicts()

        _DATASETS["fashion_mnist_10class"] = _build_dataset_entry(
            matrix=mf,
            object_ids=idsf,
            feature_names=FASHION_CLASSES,
            display_name="Fashion-MNIST Predictions (10-class, 200 pts)",
            color_fn=_point_color_fashion,
            objects_meta=objects_meta,
            source_type="generated",
        )
        return _DATASETS["fashion_mnist_10class"]

    # Check for fetched/pre-built bundles on disk in data/bundles/
    bundle_dir = Path("data/bundles") / key
    if bundle_dir.exists() and (bundle_dir / "manifest.json").exists():
        try:
            reader = BundleReader(bundle_dir)
            matrix, object_ids = reader.get_representation_matrix("probability")
            spec = reader.get_representation_spec("probability")
            feature_names = list(spec.feature_columns)
            display_name = reader.manifest.description or key
            objects_meta = reader.get_objects().to_dicts()
            _DATASETS[key] = _build_dataset_entry(
                matrix=matrix,
                object_ids=object_ids,
                feature_names=feature_names,
                display_name=display_name,
                color_fn=_point_color_generic,
                objects_meta=objects_meta,
                source_type="fetched",
            )
            return _DATASETS[key]
        except Exception:
            pass

    return None


# Pre-initialize only calibration_3class for immediate startup
_get_dataset("calibration_3class")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@workbench_bp.route("/")
def workbench() -> str:
    """Main workbench page."""
    return render_template("workbench.html")


@workbench_bp.route("/api/datasets")
def api_datasets() -> Response:
    """List available datasets and their metadata."""
    from shadowspace.datasets.bundle_discovery import scan_bundle_dir

    builtin_keys = ["calibration_3class", "synthetic_4class", "fashion_mnist_10class"]
    discovered_bundles = scan_bundle_dir("data/bundles")

    # Combine keys maintaining priority order
    all_keys = list(builtin_keys)
    for k in discovered_bundles:
        if k not in all_keys and k not in ("calibration-v1", "synthetic-v1"):
            all_keys.append(k)
    for k in _DATASETS:
        if k not in all_keys:
            all_keys.append(k)

    result = []
    for k in all_keys:
        ds = _get_dataset(k)
        if ds:
            result.append(
                {
                    "key": k,
                    "display_name": ds["display_name"],
                    "n_objects": ds["n_objects"],
                    "n_classes": ds["n_classes"],
                    "source_type": ds.get("source_type", "imported"),
                }
            )
    return Response(json.dumps(result), mimetype="application/json")


@workbench_bp.route("/api/dataset-status")
def api_dataset_status() -> Response:
    """Check dataset loading status."""
    key = request.args.get("dataset", "calibration_3class")
    loaded = key in _DATASETS
    return Response(
        json.dumps({"dataset": key, "status": "loaded" if loaded else "unloaded"}),
        mimetype="application/json",
    )


@workbench_bp.route("/api/fixture")
def api_fixture() -> Response:
    """Return dataset fixture data (defaults to calibration_3class)."""
    dataset_key = request.args.get("dataset", "calibration_3class")
    ds_data = _get_dataset(dataset_key)
    if ds_data is None:
        return Response(
            json.dumps({"error": f"Unknown dataset '{dataset_key}'"}),
            status=404,
            mimetype="application/json",
        )

    fixture_payload = {k: v for k, v in ds_data.items() if k != "payload_bytes"}
    return Response(
        json.dumps(fixture_payload, separators=(",", ":")),
        mimetype="application/json",
    )


@workbench_bp.route("/api/import-dataset", methods=["POST"])
def api_import_dataset() -> Response:
    """Import custom CSV or Parquet prediction dataset."""
    if "file" not in request.files:
        return Response(
            json.dumps({"error": "No file uploaded in form field 'file'"}),
            status=400,
            mimetype="application/json",
        )

    file_obj = request.files["file"]
    filename = file_obj.filename or "uploaded_data.csv"

    # Check file size
    file_obj.seek(0, 2)
    file_size = file_obj.tell()
    file_obj.seek(0)

    if file_size > MAX_UPLOAD_BYTES:
        return Response(
            json.dumps(
                {
                    "error": f"File size ({file_size / (1024 * 1024):.1f} MB) exceeds maximum limit of 10 MB."
                }
            ),
            status=413,
            mimetype="application/json",
        )

    # Form parameters
    id_col = request.form.get("id_column", "").strip() or None
    label_col = request.form.get("label_column", "").strip() or None
    raw_feat_cols = request.form.get("feature_columns", "").strip()
    feat_cols = (
        [c.strip() for c in raw_feat_cols.split(",") if c.strip()] if raw_feat_cols else None
    )
    normalize = request.form.get("normalize", "false").lower() in ("true", "1", "yes")
    dataset_name = (
        request.form.get("dataset_name", "").strip() or f"Imported_{uuid.uuid4().hex[:6]}"
    )

    is_parquet = filename.endswith(".parquet") or filename.endswith(".pq")
    is_csv = filename.endswith(".csv") or filename.endswith(".txt")

    if not (is_parquet or is_csv):
        return Response(
            json.dumps({"error": "Only .csv and .parquet files are supported."}),
            status=400,
            mimetype="application/json",
        )

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = f"{tmp_dir}/{filename}"
            file_obj.save(temp_path)

            bundle_out_dir = f"{tmp_dir}/bundle"
            if is_csv:
                import_csv_bundle(
                    csv_path=temp_path,
                    output_dir=bundle_out_dir,
                    id_column=id_col,
                    label_column=label_col,
                    feature_columns=feat_cols,
                    normalize=normalize,
                    dataset_name=dataset_name,
                )
            else:
                import_parquet_bundle(
                    parquet_path=temp_path,
                    output_dir=bundle_out_dir,
                    id_column=id_col,
                    label_column=label_col,
                    feature_columns=feat_cols,
                    normalize=normalize,
                    dataset_name=dataset_name,
                )

            reader = BundleReader(bundle_out_dir)
            matrix, object_ids = reader.get_representation_matrix("probability")
            rep_spec = reader.get_representation_spec("probability")
            feature_names = rep_spec.feature_columns
            objects_meta = reader.get_objects().to_dicts()

        dataset_key = f"imported_{uuid.uuid4().hex[:8]}"
        ds_entry = _build_dataset_entry(
            matrix=matrix,
            object_ids=object_ids,
            feature_names=feature_names,
            display_name=f"{dataset_name} ({len(object_ids)} pts)",
            color_fn=_point_color_generic,
            objects_meta=objects_meta,
            source_type="imported",
        )
        _DATASETS[dataset_key] = ds_entry

        return Response(
            json.dumps(
                {
                    "status": "success",
                    "dataset_key": dataset_key,
                    "display_name": ds_entry["display_name"],
                    "n_objects": len(object_ids),
                    "n_classes": len(feature_names),
                }
            ),
            status=201,
            mimetype="application/json",
        )

    except ImportValidationError as err:
        return Response(
            json.dumps({"error": f"Import Validation Failed: {err}"}),
            status=400,
            mimetype="application/json",
        )
    except Exception as err:
        return Response(
            json.dumps({"error": f"Failed to process dataset file: {err}"}),
            status=400,
            mimetype="application/json",
        )


@workbench_bp.route("/api/health")
def api_health() -> Response:
    """Health check endpoint."""
    return Response(
        json.dumps({"status": "ok", "sprint": "8"}),
        mimetype="application/json",
    )


@workbench_bp.route("/api/diagnostics")
def api_diagnostics() -> Response:
    """Compute local integrity diagnostics for a target point, k, metric, and view catalog."""
    dataset_key = request.args.get("dataset", "calibration_3class")
    ds_data = _get_dataset(dataset_key)
    if ds_data is None:
        ds_data = _get_dataset("calibration_3class")
        dataset_key = "calibration_3class"

    assert ds_data is not None
    object_ids = ds_data["object_ids"]

    target_id = request.args.get("target_id", object_ids[0])
    rep_id = request.args.get("representation", "probability")
    metric_id = request.args.get("metric", "euclidean")
    view_id = request.args.get("view_id", "pca_corners")

    try:
        k = int(request.args.get("k", 3))
    except ValueError:
        return Response(
            json.dumps({"error": "Invalid k parameter"}), status=400, mimetype="application/json"
        )

    if rep_id not in ds_data["representations"]:
        return Response(
            json.dumps({"error": f"Unknown representation {rep_id!r}"}),
            status=400,
            mimetype="application/json",
        )

    if target_id not in object_ids:
        return Response(
            json.dumps({"error": f"Unknown target_id {target_id!r}"}),
            status=400,
            mimetype="application/json",
        )

    raw_matrix = np.array(ds_data["raw_matrix"], dtype=np.float64)
    if rep_id == "sqrt_probability":
        rep_matrix = sqrt_transform(raw_matrix)
    elif rep_id == "clr_probability":
        rep_matrix = clr_transform(raw_matrix)
    else:
        rep_matrix = raw_matrix

    try:
        src_dists = _METRIC_REGISTRY.compute_pairwise_distances(rep_matrix, metric_id, rep_id)
    except (KeyError, ValueError) as err:
        return Response(json.dumps({"error": str(err)}), status=400, mimetype="application/json")

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


@workbench_bp.route("/api/topology")
def api_topology() -> Response:
    """Compute full dataset k-NN topology graph, returning all edges classified as preserved, torn, or false."""
    dataset_key = request.args.get("dataset", "calibration_3class")
    rep_id = request.args.get("representation", "probability")
    metric_id = request.args.get("metric", "euclidean")
    view_id = request.args.get("view_id", "pca_corners")

    try:
        k = int(request.args.get("k", 3))
    except ValueError:
        return Response(json.dumps({"error": "Invalid k"}), status=400, mimetype="application/json")

    ds_data = _get_dataset(dataset_key)
    if ds_data is None:
        return Response(json.dumps({"error": f"Unknown dataset '{dataset_key}'"}), status=404, mimetype="application/json")

    object_ids = ds_data["object_ids"]
    raw_matrix = np.array(ds_data["raw_matrix"], dtype=np.float64)

    if rep_id == "sqrt_probability":
        rep_matrix = sqrt_transform(raw_matrix)
    elif rep_id == "clr_probability":
        rep_matrix = clr_transform(raw_matrix)
    else:
        rep_matrix = raw_matrix

    try:
        src_dists = _METRIC_REGISTRY.compute_pairwise_distances(rep_matrix, metric_id, rep_id)
    except (KeyError, ValueError) as err:
        return Response(json.dumps({"error": str(err)}), status=400, mimetype="application/json")

    if view_id in ds_data.get("catalog", {}):
        coords_2d = np.array(ds_data["catalog"][view_id]["coords"], dtype=np.float64)
    else:
        coords_2d = np.array(ds_data["representations"].get(rep_id, {}).get("coords", []), dtype=np.float64)
        if coords_2d.size == 0:
            coords_2d = np.array(ds_data["representations"]["probability"]["coords"], dtype=np.float64)

    proj_dists = pairwise_euclidean(coords_2d)

    src_knn = compute_knn(src_dists, k, object_ids)
    proj_knn = compute_knn(proj_dists, k, object_ids)

    edges = []
    seen = set()

    for src_id in object_ids:
        src_set = set(src_knn[src_id])
        proj_set = set(proj_knn[src_id])

        for nbr_id in src_knn[src_id]:
            pair_key = tuple(sorted([src_id, nbr_id]))
            edge_type = "preserved" if nbr_id in proj_set else "torn"
            if (pair_key, edge_type) not in seen:
                seen.add((pair_key, edge_type))
                edges.append({"source": src_id, "target": nbr_id, "type": edge_type})

        for nbr_id in proj_knn[src_id]:
            if nbr_id not in src_set:
                pair_key = tuple(sorted([src_id, nbr_id]))
                if (pair_key, "false") not in seen:
                    seen.add((pair_key, "false"))
                    edges.append({"source": src_id, "target": nbr_id, "type": "false"})

    return Response(
        json.dumps({
            "dataset": dataset_key,
            "representation": rep_id,
            "metric": metric_id,
            "k": k,
            "edges": edges,
            "n_edges": len(edges),
        }),
        mimetype="application/json",
    )


@workbench_bp.route("/api/distortion-grid")
def api_distortion_grid() -> Response:
    """Compute spatial projection distortion grid over 2D viewport."""
    dataset_key = request.args.get("dataset", "calibration_3class")
    rep_id = request.args.get("representation", "probability")
    metric_id = request.args.get("metric", "euclidean")
    view_id = request.args.get("view_id", "pca_corners")

    try:
        res = max(8, min(64, int(request.args.get("resolution", 32))))
    except ValueError:
        res = 32

    ds_data = _get_dataset(dataset_key)
    if ds_data is None:
        return Response(json.dumps({"error": f"Unknown dataset '{dataset_key}'"}), status=404, mimetype="application/json")

    raw_matrix = np.array(ds_data["raw_matrix"], dtype=np.float64)
    if rep_id == "sqrt_probability":
        rep_matrix = sqrt_transform(raw_matrix)
    elif rep_id == "clr_probability":
        rep_matrix = clr_transform(raw_matrix)
    else:
        rep_matrix = raw_matrix

    try:
        src_dists = _METRIC_REGISTRY.compute_pairwise_distances(rep_matrix, metric_id, rep_id)
    except (KeyError, ValueError) as err:
        return Response(json.dumps({"error": str(err)}), status=400, mimetype="application/json")

    if view_id in ds_data.get("catalog", {}):
        coords_2d = np.array(ds_data["catalog"][view_id]["coords"], dtype=np.float64)
    else:
        coords_2d = np.array(ds_data["representations"].get(rep_id, {}).get("coords", []), dtype=np.float64)
        if coords_2d.size == 0:
            coords_2d = np.array(ds_data["representations"]["probability"]["coords"], dtype=np.float64)

    proj_dists = pairwise_euclidean(coords_2d)

    src_mean = float(np.mean(src_dists[src_dists > 0])) if np.any(src_dists > 0) else 1.0
    proj_mean = float(np.mean(proj_dists[proj_dists > 0])) if np.any(proj_dists > 0) else 1.0

    n_pts = len(coords_2d)
    point_distortions = np.ones(n_pts, dtype=np.float64)

    for i in range(n_pts):
        ratios = []
        for j in range(n_pts):
            if i == j:
                continue
            s_d = src_dists[i, j] / src_mean
            p_d = proj_dists[i, j] / proj_mean
            if s_d > 1e-6:
                ratios.append(p_d / s_d)
        if ratios:
            point_distortions[i] = float(np.mean(ratios))

    # Compute actual data bounds with 10% padding (works for any dataset scale)
    xs = coords_2d[:, 0]
    ys = coords_2d[:, 1]
    x_min_data, x_max_data = float(xs.min()), float(xs.max())
    y_min_data, y_max_data = float(ys.min()), float(ys.max())
    pad_x = max(0.05, (x_max_data - x_min_data) * 0.12)
    pad_y = max(0.05, (y_max_data - y_min_data) * 0.12)
    bx_min = x_min_data - pad_x
    bx_max = x_max_data + pad_x
    by_min = y_min_data - pad_y
    by_max = y_max_data + pad_y
    span_x = bx_max - bx_min
    span_y = by_max - by_min

    grid = [[None for _ in range(res)] for _ in range(res)]
    cell_w = span_x / res
    cell_h = span_y / res

    for r in range(res):
        cell_y_min = by_max - (r + 1) * cell_h
        cell_y_max = by_max - r * cell_h
        for c in range(res):
            cell_x_min = bx_min + c * cell_w
            cell_x_max = bx_min + (c + 1) * cell_w

            cell_pts = []
            for idx, (x, y) in enumerate(coords_2d):
                if cell_x_min <= x <= cell_x_max and cell_y_min <= y <= cell_y_max:
                    cell_pts.append(point_distortions[idx])

            if cell_pts:
                grid[r][c] = round(float(np.mean(cell_pts)), 3)

    return Response(
        json.dumps({
            "dataset": dataset_key,
            "resolution": res,
            "grid": grid,
            "bounds": {"xMin": round(bx_min, 4), "xMax": round(bx_max, 4), "yMin": round(by_min, 4), "yMax": round(by_max, 4)},
        }),
        mimetype="application/json",
    )


@workbench_bp.route("/api/subspace-angles")
def api_subspace_angles() -> Response:
    """Compute canonical principal angles between View A and View B projection bases."""
    dataset_key = request.args.get("dataset", "calibration_3class")
    rep_id = request.args.get("representation", "probability")
    view_a = request.args.get("view_a", "pca_corners")
    view_b = request.args.get("view_b", "fisher_lda")

    ds_data = _get_dataset(dataset_key)
    if ds_data is None:
        return Response(json.dumps({"error": f"Unknown dataset '{dataset_key}'"}), status=404, mimetype="application/json")

    raw_matrix = np.array(ds_data["raw_matrix"], dtype=np.float64)
    if rep_id == "sqrt_probability":
        rep_matrix = sqrt_transform(raw_matrix)
    elif rep_id == "clr_probability":
        rep_matrix = clr_transform(raw_matrix)
    else:
        rep_matrix = raw_matrix

    def get_basis(view_name: str) -> NDArray[np.float64]:
        if view_name == "fisher_lda":
            from shadowspace.projection.subspace import find_discriminative_basis
            objects_meta = ds_data.get("objects_meta", [])
            labels = []
            for i in range(len(raw_matrix)):
                meta = objects_meta[i] if (isinstance(objects_meta, list) and i < len(objects_meta)) else None
                if isinstance(meta, dict) and "generator_component" in meta and meta["generator_component"]:
                    labels.append(meta["generator_component"])
                elif isinstance(meta, dict) and "pred_class_name" in meta and meta["pred_class_name"]:
                    labels.append(meta["pred_class_name"])
                else:
                    labels.append(f"c_{i % 3}")
            return find_discriminative_basis(rep_matrix, labels)

        cat = ds_data.get("catalog", {})
        if view_name in cat:
            return np.array(cat[view_name]["basis"], dtype=np.float64)

        mat_centered = rep_matrix - rep_matrix.mean(axis=0)
        _, _, vh = np.linalg.svd(mat_centered, full_matrices=False)
        return validate_orthonormal_basis(vh[:2, :].T)

    b_a = get_basis(view_a)
    b_b = get_basis(view_b)

    t1, t2 = compute_canonical_angles(b_a, b_b)
    dist = compute_grassmannian_distance(t1, t2)

    interpretation = "tight" if dist < 15.0 else ("moderate" if dist < 45.0 else "divergent")

    return Response(
        json.dumps({
            "dataset": dataset_key,
            "representation": rep_id,
            "view_a": view_a,
            "view_b": view_b,
            "theta_1_deg": round(t1, 2),
            "theta_2_deg": round(t2, 2),
            "grassmannian_dist_deg": round(dist, 2),
            "interpretation": interpretation,
        }),
        mimetype="application/json",
    )


@workbench_bp.route("/api/object-payload")
def api_object_payload() -> Response:
    """Return raw object payload image or JSON metadata for target object."""
    dataset_key = request.args.get("dataset", "calibration_3class")
    target_id = request.args.get("target_id", "")

    ds_data = _get_dataset(dataset_key)
    if ds_data is None or not target_id:
        return Response(
            json.dumps({"error": "Dataset or target_id not found"}),
            status=404,
            mimetype="application/json",
        )

    payload_map = ds_data.get("payload_bytes", {})
    if target_id in payload_map:
        return Response(payload_map[target_id], mimetype="image/png")

    return Response(
        json.dumps({"object_id": target_id, "has_image": False}),
        status=200,
        mimetype="application/json",
    )


@workbench_bp.route("/api/export-report", methods=["POST"])
def api_export_report() -> Response:
    """Generate and return a downloadable Markdown investigation report."""
    data = request.get_json(silent=True) or {}
    dataset_key = data.get("dataset", "calibration_3class")
    target_id = data.get("target_id", "none")
    rep_id = data.get("representation", "probability")
    metric_id = data.get("metric", "euclidean")
    view_id = data.get("view_id", "pca_corners")
    k = data.get("k", 3)
    saved_views = data.get("saved_views", [])

    ds_data = _get_dataset(dataset_key)
    display_name = ds_data["display_name"] if ds_data else dataset_key

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    report_lines = [
        "# Shadowspace Investigation Record",
        f"**Dataset**: {display_name} (`{dataset_key}`)",
        f"**Generated**: {timestamp}",
        f"**Active Representation**: `{rep_id}`",
        f"**Active Metric**: `{metric_id}`",
        f"**Active Catalog View**: `{view_id}`",
        f"**Neighborhood Size (k)**: {k}",
        f"**Selected Target ID**: `{target_id}`",
        "",
        "---",
        "## Session Configuration & Diagnostics",
        f"- Target Point: `{target_id}`",
        f"- View ID: `{view_id}`",
        f"- Metric Space: `{metric_id}` under `{rep_id}`",
        "",
        "---",
        "## Saved View Atlas Snapshots",
    ]

    if saved_views:
        report_lines.append("| Name | Representation | Metric | k | Target ID | Note |")
        report_lines.append("| --- | --- | --- | --- | --- | --- |")
        for sv in saved_views:
            report_lines.append(
                f"| {sv.get('name', 'Untitled')} | {sv.get('representation_id')} | {sv.get('metric_id')} | {sv.get('k')} | {sv.get('target_id')} | {sv.get('note', '')} |"
            )
    else:
        report_lines.append("_No saved views recorded during this session._")

    report_lines.extend(
        [
            "",
            "---",
            "## System Provenance & Data Contract",
            "- **Shadowspace Version**: Sprint 9 (Belief Space Payload & Transition Engine)",
            "- **Reproducibility Guarantee**: Canonical distance metrics and basis projections preserved.",
        ]
    )

    report_content = "\n".join(report_lines)
    return Response(
        report_content,
        mimetype="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=shadowspace-investigation-{dataset_key}.md"
        },
    )


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
            return Response(
                json.dumps({"error": "Invalid k parameter"}),
                status=400,
                mimetype="application/json",
            )

        ds_data = _get_dataset(dataset_key) or _get_dataset("calibration_3class")
        assert ds_data is not None
        if rep_id not in ds_data["representations"]:
            return Response(
                json.dumps({"error": f"Unknown representation {rep_id!r}"}),
                status=400,
                mimetype="application/json",
            )
        if target_id not in ds_data["object_ids"]:
            return Response(
                json.dumps({"error": f"Unknown target_id {target_id!r}"}),
                status=400,
                mimetype="application/json",
            )

        view_id = f"view_{uuid.uuid4().hex[:8]}"
        saved = SavedView(
            id=view_id,
            name=view_name,
            note=note,
            representation_id=rep_id,
            metric_id=metric_id,
            k=k,
            target_id=target_id,
            variance_explained=ds_data["representations"]
            .get(rep_id, {})
            .get("eigenvalues", [0.5, 0.5]),
            metadata={"dataset": dataset_key},
        )
        _SAVED_VIEWS[view_id] = saved
        return Response(
            json.dumps(saved.model_dump(mode="json")), status=201, mimetype="application/json"
        )

    elif request.method == "DELETE":
        view_id = request.args.get("id", "")
        if view_id in _SAVED_VIEWS:
            del _SAVED_VIEWS[view_id]
            return Response(
                json.dumps({"status": "deleted", "id": view_id}), mimetype="application/json"
            )
        return Response(
            json.dumps({"error": f"View {view_id!r} not found"}),
            status=404,
            mimetype="application/json",
        )

    return Response(
        json.dumps({"error": "Method not allowed"}), status=405, mimetype="application/json"
    )


@workbench_bp.route("/api/export-record")
def api_export_record() -> Response:
    """Export complete InvestigationRecord JSON with SHA-256 provenance hashes."""
    ds = _get_dataset("calibration_3class")
    assert ds is not None
    matrix = np.array(ds["raw_matrix"], dtype=np.float64)
    object_ids = ds["object_ids"]
    feature_names = ds["feature_names"]

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
        summary_note="Investigation record exported from Shadowspace Sprint 8 workbench shell.",
    )
    headers = {"Content-Disposition": "attachment; filename=investigation_record.json"}
    return Response(
        json.dumps(record.model_dump(mode="json"), indent=2),
        mimetype="application/json",
        headers=headers,
    )


@workbench_bp.route("/api/tour-path")
def api_tour_path() -> Response:
    """Return Grand Tour sequence of 2D projection frames for active dataset."""
    dataset_key = request.args.get("dataset", "calibration_3class")
    rep_id = request.args.get("representation", "probability")

    try:
        n_frames = int(request.args.get("n_frames", 180))
    except (TypeError, ValueError):
        n_frames = 180

    ds_data = _get_dataset(dataset_key)
    if ds_data is None:
        return Response(
            json.dumps({"error": f"Unknown dataset '{dataset_key}'"}),
            status=404,
            mimetype="application/json",
        )

    raw_matrix = np.array(ds_data["raw_matrix"], dtype=np.float64)
    if rep_id == "sqrt_probability":
        rep_matrix = sqrt_transform(raw_matrix)
    elif rep_id == "clr_probability":
        rep_matrix = clr_transform(raw_matrix)
    else:
        rep_matrix = raw_matrix

    frames_coords, bases_matrices = generate_grand_tour_path(rep_matrix, n_frames=n_frames)

    payload = {
        "dataset": dataset_key,
        "representation": rep_id,
        "n_frames": len(frames_coords),
        "frames": frames_coords,
        "bases": bases_matrices,
        "semantically_valid": True,
        "kind": "linear_projection",
        "geodesic_algorithm": "GLERP",
    }

    return Response(json.dumps(payload), mimetype="application/json")


@workbench_bp.route("/api/optimize-view")
def api_optimize_view() -> Response:
    """Compute an optimal 2D projection basis (class separation or integrity optimal) and return a GLERP geodesic transition."""
    dataset_key = request.args.get("dataset", "calibration_3class")
    rep_id = request.args.get("representation", "probability")
    criterion = request.args.get("criterion", "class_separation")  # "class_separation" | "neighborhood_integrity"
    target_id = request.args.get("target_id", None)

    try:
        n_frames = max(2, min(360, int(request.args.get("n_frames", 60))))
    except (TypeError, ValueError):
        n_frames = 60

    ds_data = _get_dataset(dataset_key)
    if ds_data is None:
        return Response(
            json.dumps({"error": f"Unknown dataset '{dataset_key}'"}),
            status=404,
            mimetype="application/json",
        )

    raw_matrix = np.array(ds_data["raw_matrix"], dtype=np.float64)
    if rep_id == "sqrt_probability":
        rep_matrix = sqrt_transform(raw_matrix)
    elif rep_id == "clr_probability":
        rep_matrix = clr_transform(raw_matrix)
    else:
        rep_matrix = raw_matrix

    # Z-score normalize matrix before projection
    mat_centered = rep_matrix - rep_matrix.mean(axis=0)
    stds = mat_centered.std(axis=0)
    stds[stds < 1e-8] = 1.0
    mat_norm = mat_centered / stds

    # Default start basis: 2D PCA basis of rep_matrix
    mat_centered_pca = rep_matrix - rep_matrix.mean(axis=0)
    _, _, vh_start = np.linalg.svd(mat_centered_pca, full_matrices=False)
    start_basis = validate_orthonormal_basis(vh_start[:2, :].T)

    object_ids = ds_data["object_ids"]
    objects_meta = ds_data.get("objects_meta", [])

    if criterion == "class_separation":
        # Extract finest class/component labels from metadata
        labels = []
        for i in range(len(raw_matrix)):
            meta = objects_meta[i] if (isinstance(objects_meta, list) and i < len(objects_meta)) else None
            if isinstance(meta, dict) and "generator_component" in meta and meta["generator_component"]:
                labels.append(meta["generator_component"])
            elif isinstance(meta, dict) and "pred_class_name" in meta and meta["pred_class_name"]:
                labels.append(meta["pred_class_name"])
            elif isinstance(meta, dict) and "true_label" in meta and meta["true_label"]:
                labels.append(meta["true_label"])
            elif isinstance(meta, dict) and "predicted_label" in meta and meta["predicted_label"]:
                labels.append(meta["predicted_label"])
            else:
                labels.append(int(np.argmax(raw_matrix[i])))
        target_basis = find_discriminative_basis(rep_matrix, labels)

        # If target_basis coincides with start_basis (Grassmannian distance < 0.08 rad),
        # rotate to minor discriminant / orthogonal components so the tour always moves
        if grassmannian_distance(start_basis, target_basis) < 0.08:
            _, _, vh_all = np.linalg.svd(mat_centered_pca, full_matrices=True)
            if vh_all.shape[0] >= 4:
                target_basis = validate_orthonormal_basis(vh_all[[2, 3], :].T)
            elif vh_all.shape[0] >= 3:
                target_basis = validate_orthonormal_basis(vh_all[[1, 2], :].T)
    else:  # "neighborhood_integrity"
        target_indices = []
        if target_id and target_id in object_ids:
            t_idx = object_ids.index(target_id)
            target_indices = [t_idx]
            # Add top k neighbors in raw space
            dists = np.linalg.norm(rep_matrix - rep_matrix[t_idx], axis=1)
            neighbor_indices = np.argsort(dists)[:6].tolist()
            target_indices = neighbor_indices
        target_basis = find_integrity_optimal_basis(rep_matrix, target_indices)

        if grassmannian_distance(start_basis, target_basis) < 0.08:
            _, _, vh_all = np.linalg.svd(mat_centered_pca, full_matrices=True)
            if vh_all.shape[0] >= 3:
                target_basis = validate_orthonormal_basis(vh_all[[1, 2], :].T)

    # Generate smooth GLERP geodesic transition from start_basis to target_basis
    raw_frames = []
    bases_matrices = []
    for step in range(n_frames):
        tau = step / float(max(1, n_frames - 1))
        b_interp = grassmann_geodesic(start_basis, target_basis, tau)
        coords_2d = project(mat_norm, b_interp)
        raw_frames.append(coords_2d)
        bases_matrices.append(b_interp.tolist())

    # Global normalise across transition frames -> [-1, 1]
    all_coords = np.concatenate(raw_frames, axis=0)
    global_min = all_coords.min(axis=0)
    global_max = all_coords.max(axis=0)
    global_range = global_max - global_min
    global_range[global_range < 1e-8] = 1.0

    frames_coords = []
    for frame in raw_frames:
        normed = (frame - global_min) / global_range * 2.0 - 1.0
        frames_coords.append(normed.tolist())

    payload = {
        "dataset": dataset_key,
        "representation": rep_id,
        "criterion": criterion,
        "target_basis": target_basis.tolist(),
        "n_frames": len(frames_coords),
        "frames": frames_coords,
        "bases": bases_matrices,
        "semantically_valid": True,
        "kind": "linear_projection",
        "geodesic_algorithm": "GLERP",
    }

    return Response(json.dumps(payload), mimetype="application/json")


@workbench_bp.route("/api/point-stability")
def api_point_stability() -> Response:
    """Compute per-point stability overlap scores across candidate catalog views."""
    dataset_key = request.args.get("dataset", "calibration_3class")
    rep_id = request.args.get("representation", "probability")
    metric_id = request.args.get("metric", "euclidean")
    k = request.args.get("k", 5, type=int)

    ds_data = _get_dataset(dataset_key)
    if not ds_data:
        return Response(json.dumps({"error": f"Unknown dataset {dataset_key!r}"}), status=404, mimetype="application/json")

    raw_matrix = np.array(ds_data["raw_matrix"], dtype=np.float64)
    if rep_id == "sqrt_probability":
        rep_matrix = sqrt_transform(raw_matrix)
    elif rep_id == "clr_probability":
        rep_matrix = clr_transform(raw_matrix)
    elif rep_id == "logits":
        rep_matrix = logit_transform(raw_matrix)
    else:
        rep_matrix = raw_matrix

    n_pts = len(rep_matrix)

    # Compute source k-NN
    try:
        src_dists = _METRIC_REGISTRY.compute_pairwise_distances(rep_matrix, metric_id, rep_id)
    except (KeyError, ValueError) as err:
        return Response(json.dumps({"error": str(err)}), status=400, mimetype="application/json")

    k_eff = min(k, max(1, n_pts - 1))
    src_knn = np.argsort(src_dists, axis=1)[:, 1:k_eff+1]

    # Collect catalog 2D coords
    catalog = ds_data.get("catalog", {})
    catalog_coords = {}
    for v_id, v_data in catalog.items():
        if "coords" in v_data:
            catalog_coords[v_id] = np.array(v_data["coords"], dtype=np.float64)

    if not catalog_coords:
        if "representations" in ds_data and rep_id in ds_data["representations"]:
            catalog_coords["default"] = np.array(ds_data["representations"][rep_id]["coords"], dtype=np.float64)

    res = compute_point_stability(rep_matrix, catalog_coords, src_knn, k=k_eff)
    res["dataset"] = dataset_key
    res["representation"] = rep_id
    res["metric"] = metric_id
    res["k"] = k_eff

    return Response(json.dumps(res), mimetype="application/json")


@workbench_bp.route("/api/rashomon-set")
def api_rashomon_set() -> Response:
    """Generate a diverse Rashomon candidate set of projection bases."""
    dataset_key = request.args.get("dataset", "calibration_3class")
    rep_id = request.args.get("representation", "probability")
    view_id = request.args.get("view_id", "pca_corners")
    threshold = request.args.get("threshold", 0.50, type=float)

    ds_data = _get_dataset(dataset_key)
    if not ds_data:
        return Response(json.dumps({"error": f"Unknown dataset {dataset_key!r}"}), status=404, mimetype="application/json")

    raw_matrix = np.array(ds_data["raw_matrix"], dtype=np.float64)
    if rep_id == "sqrt_probability":
        rep_matrix = sqrt_transform(raw_matrix)
    elif rep_id == "clr_probability":
        rep_matrix = clr_transform(raw_matrix)
    elif rep_id == "logits":
        rep_matrix = logit_transform(raw_matrix)
    else:
        rep_matrix = raw_matrix

    # Extract labels if present
    objects_meta = ds_data.get("objects_meta", [])
    y_labels = None
    if objects_meta and isinstance(objects_meta, list):
        if "label" in objects_meta[0]:
            y_labels = np.array([o.get("label", 0) for o in objects_meta])
        elif "class_name" in objects_meta[0]:
            classes = sorted(list(set(o.get("class_name", "") for o in objects_meta)))
            c_map = {c: i for i, c in enumerate(classes)}
            y_labels = np.array([c_map.get(o.get("class_name", ""), 0) for o in objects_meta])

    current_basis = None
    if view_id in ds_data.get("catalog", {}):
        v_info = ds_data["catalog"][view_id]
        if "basis" in v_info.get("provenance", {}):
            current_basis = np.array(v_info["provenance"]["basis"], dtype=np.float64)

    candidates = generate_rashomon_set(
        rep_matrix, Y_labels=y_labels, current_basis=current_basis, n_candidates=6, quality_threshold=threshold
    )

    payload = {
        "dataset": dataset_key,
        "representation": rep_id,
        "active_view": view_id,
        "threshold": threshold,
        "n_candidates": len(candidates),
        "candidates": candidates,
    }

    return Response(json.dumps(payload), mimetype="application/json")

