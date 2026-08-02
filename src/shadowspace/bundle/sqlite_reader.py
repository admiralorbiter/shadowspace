"""SQLite Bundle Reader for Shadowspace artifact bundles.

Reads self-contained SQLite database bundles, returning metadata, BLOB representation
matrices, pre-computed k-NN graphs, and catalog view projections.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

try:
    import sqlite_vec
    _HAS_SQLITE_VEC = True
except ImportError:
    _HAS_SQLITE_VEC = False


class SQLiteBundleReader:
    """Reads and queries a single-file SQLite database bundle."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database bundle not found at {self.db_path}")

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if _HAS_SQLITE_VEC:
            try:
                self.conn.enable_load_extension(True)
                sqlite_vec.load(self.conn)
                self.conn.enable_load_extension(False)
            except Exception:
                pass

    def get_manifest(self) -> dict[str, Any]:
        """Load manifest key-value metadata dictionary."""
        rows = self.conn.execute("SELECT key, value FROM manifest").fetchall()
        result = {}
        for r in rows:
            try:
                result[r["key"]] = json.loads(r["value"])
            except Exception:
                result[r["key"]] = r["value"]
        return result

    def get_objects(self) -> pl.DataFrame:
        """Load objects metadata into a Polars DataFrame."""
        rows = self.conn.execute("SELECT object_id, true_label, pred_label, entropy, color, json_meta FROM objects").fetchall()
        records = []
        for r in rows:
            item = {"object_id": r["object_id"]}
            if r["true_label"]:
                item["true_label"] = r["true_label"]
            if r["pred_label"]:
                item["predicted_label"] = r["pred_label"]
            if r["entropy"] is not None:
                item["entropy"] = r["entropy"]
            if r["color"]:
                item["color"] = r["color"]
            if r["json_meta"]:
                try:
                    extra = json.loads(r["json_meta"])
                    item.update(extra)
                except Exception:
                    pass
            records.append(item)

        return pl.DataFrame(records)

    def get_object_ids(self) -> list[str]:
        """Return list of string object IDs in order."""
        rows = self.conn.execute("SELECT object_id FROM objects").fetchall()
        return [r["object_id"] for r in rows]

    def get_representation_matrix(self, rep_id: str) -> tuple[NDArray[np.float64], list[str]]:
        """Return (numpy_matrix, object_ids) for representation_id."""
        row = self.conn.execute(
            "SELECT n_objects, n_features, dtype, data FROM representation_blobs WHERE rep_id = ?",
            (rep_id,)
        ).fetchone()

        if row is None:
            raise KeyError(f"Representation {rep_id!r} not found in database bundle.")

        n_obs = row["n_objects"]
        n_feat = row["n_features"]
        dtype_str = row["dtype"]
        data_bytes = row["data"]

        mat = np.frombuffer(data_bytes, dtype=np.dtype(dtype_str)).reshape((n_obs, n_feat))
        object_ids = self.get_object_ids()
        return mat, object_ids

    def get_knn_graph(self, rep_id: str, metric_id: str, k: int) -> dict[str, list[str]]:
        """Return pre-computed top-k nearest neighbors dict mapping src_id -> [neighbor_ids]."""
        rows = self.conn.execute(
            """
            SELECT src_id, neighbor_id, rank
            FROM knn_edges
            WHERE rep_id = ? AND metric_id = ? AND rank <= ?
            ORDER BY src_id, rank ASC
            """,
            (rep_id, metric_id, k)
        ).fetchall()

        graph: dict[str, list[str]] = {}
        for r in rows:
            src_id = r["src_id"]
            nbr_id = r["neighbor_id"]
            if src_id not in graph:
                graph[src_id] = []
            graph[src_id].append(nbr_id)
        return graph

    def get_catalog_views(self) -> dict[str, dict[str, Any]]:
        """Return catalog views metadata dict with basis arrays."""
        rows = self.conn.execute("SELECT * FROM catalog_views").fetchall()
        catalog = {}
        for r in rows:
            p_dim = r["p_dim"]
            b_mat = np.frombuffer(r["basis_data"], dtype=np.float64).reshape((p_dim, 2))
            catalog[r["view_id"]] = {
                "view_id": r["view_id"],
                "rep_id": r["rep_id"],
                "display_name": r["display_name"],
                "basis": b_mat.tolist(),
                "semantically_valid": bool(r["semantically_valid"]),
                "is_misleading": bool(r["is_misleading"]),
                "description": r["description"],
                "warning_note": r["warning_note"],
            }
        return catalog

    def get_view_coords(self, view_id: str, rep_id: str) -> NDArray[np.float64]:
        """Return (N, 2) array of coordinates for view_id and rep_id."""
        rows = self.conn.execute(
            "SELECT x, y FROM view_coords WHERE view_id = ? AND rep_id = ? ORDER BY rowid",
            (view_id, rep_id)
        ).fetchall()
        if not rows:
            raise KeyError(f"Coordinates for view {view_id!r} and rep {rep_id!r} not found.")
        return np.array([[r["x"], r["y"]] for r in rows], dtype=np.float64)

    def __enter__(self) -> SQLiteBundleReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close SQLite database connection."""
        if hasattr(self, "conn") and self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
