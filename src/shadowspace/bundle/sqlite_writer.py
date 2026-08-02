"""SQLite Bundle Writer for Shadowspace artifact bundles.

Compiles objects, representations, distance metrics, pre-computed k-NN graphs,
and projection catalog views into a single self-contained SQLite database file.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

try:
    import sqlite_vec
    _HAS_SQLITE_VEC = True
except ImportError:
    _HAS_SQLITE_VEC = False


class SQLiteBundleWriter:
    """Creates a single-file SQLite database artifact bundle."""

    def __init__(self, db_path: Path | str, bundle_id: str, description: str = "") -> None:
        self.db_path = Path(db_path)
        self.bundle_id = bundle_id
        self.description = description
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.db_path.exists():
            self.db_path.unlink()

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        if _HAS_SQLITE_VEC:
            try:
                self.conn.enable_load_extension(True)
                sqlite_vec.load(self.conn)
                self.conn.enable_load_extension(False)
            except Exception:
                pass

        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema tables."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS manifest (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) STRICT;
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS objects (
                    object_id TEXT PRIMARY KEY,
                    true_label TEXT,
                    pred_label TEXT,
                    entropy REAL,
                    color TEXT,
                    json_meta TEXT
                ) STRICT;
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS representation_blobs (
                    rep_id TEXT PRIMARY KEY,
                    n_objects INTEGER NOT NULL,
                    n_features INTEGER NOT NULL,
                    dtype TEXT NOT NULL DEFAULT 'float64',
                    data BLOB NOT NULL,
                    sha256 TEXT NOT NULL
                ) STRICT;
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS knn_edges (
                    rep_id TEXT NOT NULL,
                    metric_id TEXT NOT NULL,
                    src_id TEXT NOT NULL,
                    neighbor_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    distance REAL NOT NULL,
                    PRIMARY KEY (rep_id, metric_id, src_id, rank)
                ) STRICT;
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_knn_lookup ON knn_edges(rep_id, metric_id, src_id, rank);")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS catalog_views (
                    view_id TEXT PRIMARY KEY,
                    rep_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    basis_data BLOB NOT NULL,
                    p_dim INTEGER NOT NULL,
                    semantically_valid INTEGER NOT NULL DEFAULT 1,
                    is_misleading INTEGER NOT NULL DEFAULT 0,
                    description TEXT DEFAULT '',
                    warning_note TEXT DEFAULT ''
                ) STRICT;
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS view_coords (
                    view_id TEXT NOT NULL,
                    rep_id TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    PRIMARY KEY (view_id, rep_id, object_id)
                ) STRICT;
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_coords_lookup ON view_coords(view_id, rep_id);")

    def set_objects(self, df: pl.DataFrame) -> None:
        """Populate objects table from a Polars DataFrame."""
        if "object_id" not in df.columns:
            raise ValueError("objects DataFrame must contain 'object_id'")

        rows = []
        for d in df.to_dicts():
            oid = str(d.pop("object_id"))
            t_lbl = str(d.get("true_label", "")) if "true_label" in d else None
            p_lbl = str(d.get("predicted_label", "")) if "predicted_label" in d else None
            ent = float(d.get("entropy", 0.0)) if "entropy" in d else None
            col = str(d.get("color", "")) if "color" in d else None
            json_meta = json.dumps(d)
            rows.append((oid, t_lbl, p_lbl, ent, col, json_meta))

        with self.conn:
            self.conn.executemany(
                "INSERT INTO objects VALUES (?, ?, ?, ?, ?, ?)", rows
            )

    def add_representation(self, rep_id: str, matrix: NDArray[np.float64], object_ids: list[str]) -> None:
        """Add coordinate matrix BLOB and optionally populate sqlite-vec table."""
        mat64 = np.asarray(matrix, dtype=np.float64)
        n_obs, n_feat = mat64.shape
        blob = mat64.tobytes()
        sha256_val = hashlib.sha256(blob).hexdigest()

        with self.conn:
            self.conn.execute(
                "INSERT INTO representation_blobs VALUES (?, ?, ?, 'float64', ?, ?)",
                (rep_id, n_obs, n_feat, blob, sha256_val)
            )

            if _HAS_SQLITE_VEC:
                try:
                    self.conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_{rep_id} USING vec0(embedding float[{n_feat}])")
                    vec_rows = [
                        (i + 1, row.astype(np.float32).tobytes())
                        for i, row in enumerate(mat64)
                    ]
                    self.conn.executemany(
                        f"INSERT INTO vec_{rep_id}(rowid, embedding) VALUES (?, ?)",
                        vec_rows
                    )
                except Exception:
                    pass

    def add_knn_graph(self, rep_id: str, metric_id: str, object_ids: list[str], knn_indices: NDArray[np.int64], distances: NDArray[np.float64]) -> None:
        """Populate pre-computed k-NN graph edges."""
        rows = []
        n_obs, k_eff = knn_indices.shape
        for i in range(n_obs):
            src_id = object_ids[i]
            for rank_idx in range(k_eff):
                nbr_idx = int(knn_indices[i, rank_idx])
                dist_val = float(distances[i, nbr_idx])
                nbr_id = object_ids[nbr_idx]
                rows.append((rep_id, metric_id, src_id, nbr_id, rank_idx + 1, dist_val))

        with self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO knn_edges VALUES (?, ?, ?, ?, ?, ?)", rows)

    def add_catalog_view(self, view_id: str, rep_id: str, display_name: str, basis: NDArray[np.float64], semantically_valid: bool = True, is_misleading: bool = False, description: str = "", warning_note: str = "") -> None:
        """Add projection catalog view basis."""
        basis64 = np.asarray(basis, dtype=np.float64)
        p_dim = basis64.shape[0]
        b_blob = basis64.tobytes()

        with self.conn:
            self.conn.execute(
                "INSERT INTO catalog_views VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (view_id, rep_id, display_name, b_blob, p_dim, int(semantically_valid), int(is_misleading), description, warning_note)
            )

    def add_view_coords(self, view_id: str, rep_id: str, object_ids: list[str], coords_2d: NDArray[np.float64]) -> None:
        """Add pre-projected 2D coordinates for a catalog view."""
        coords = np.asarray(coords_2d, dtype=np.float64)
        rows = [
            (view_id, rep_id, oid, float(coords[i, 0]), float(coords[i, 1]))
            for i, oid in enumerate(object_ids)
        ]
        with self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO view_coords VALUES (?, ?, ?, ?, ?)", rows)

    def finalize(self) -> Path:
        """Write manifest table entries and close connection."""
        created_iso = datetime.now(UTC).isoformat()
        manifest_entries = [
            ("bundle_id", json.dumps(self.bundle_id)),
            ("schema_version", json.dumps("0.2.0-sqlite")),
            ("created_at", json.dumps(created_iso)),
            ("description", json.dumps(self.description)),
        ]
        with self.conn:
            self.conn.executemany("INSERT INTO manifest VALUES (?, ?)", manifest_entries)

        try:
            self.conn.execute("PRAGMA journal_mode = DELETE;")
        except Exception:
            pass

        self.conn.close()
        return self.db_path
