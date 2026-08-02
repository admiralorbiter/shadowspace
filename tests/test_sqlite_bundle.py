"""Tests for SQLiteBundleWriter and SQLiteBundleReader."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from shadowspace.bundle.sqlite_reader import SQLiteBundleReader
from shadowspace.bundle.sqlite_writer import SQLiteBundleWriter


def test_sqlite_bundle_writer_and_reader_roundtrip() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = Path(tmp_dir) / "test_bundle.db"

        # 1. Prepare sample data
        object_ids = [f"obj_{i}" for i in range(10)]
        objects_df = pl.DataFrame({
            "object_id": object_ids,
            "true_label": [f"class_{i % 2}" for i in range(10)],
            "entropy": [0.5] * 10,
        })

        rng = np.random.default_rng(42)
        prob_matrix = rng.dirichlet(alpha=[1, 1, 1], size=10)

        knn_indices = np.argsort(prob_matrix, axis=1)[:, :3]
        knn_distances = np.ones((10, 10), dtype=np.float64) * 0.5

        basis = np.eye(3)[:, :2]
        coords_2d = prob_matrix @ basis

        # 2. Write bundle
        writer = SQLiteBundleWriter(db_path=db_path, bundle_id="test_sqlite_bundle_v1", description="Test SQLite bundle")
        writer.set_objects(objects_df)
        writer.add_representation("probability", prob_matrix, object_ids)
        writer.add_knn_graph("probability", "euclidean", object_ids, knn_indices, knn_distances)
        writer.add_catalog_view("pca_corners", "probability", "PCA Corners", basis)
        writer.add_view_coords("pca_corners", "probability", object_ids, coords_2d)
        writer.finalize()

        assert db_path.exists()

        # 3. Read bundle
        reader = SQLiteBundleReader(db_path)
        manifest = reader.get_manifest()
        assert manifest["bundle_id"] == "test_sqlite_bundle_v1"
        assert manifest["schema_version"] == "0.2.0-sqlite"

        read_ids = reader.get_object_ids()
        assert read_ids == object_ids

        read_objects_df = reader.get_objects()
        assert len(read_objects_df) == 10
        assert "object_id" in read_objects_df.columns

        read_matrix, read_ids2 = reader.get_representation_matrix("probability")
        assert read_ids2 == object_ids
        assert read_matrix.shape == (10, 3)
        np.testing.assert_allclose(read_matrix, prob_matrix)

        read_knn = reader.get_knn_graph("probability", "euclidean", k=3)
        assert len(read_knn) == 10
        assert "obj_0" in read_knn
        assert len(read_knn["obj_0"]) == 3

        read_catalog = reader.get_catalog_views()
        assert "pca_corners" in read_catalog
        assert read_catalog["pca_corners"]["display_name"] == "PCA Corners"
        assert np.array(read_catalog["pca_corners"]["basis"]).shape == (3, 2)

        read_coords = reader.get_view_coords("pca_corners", "probability")
        assert read_coords.shape == (10, 2)
        np.testing.assert_allclose(read_coords, coords_2d)

        reader.close()
        del reader
        del writer
        import gc
        gc.collect()
        if db_path.exists():
            db_path.unlink()
