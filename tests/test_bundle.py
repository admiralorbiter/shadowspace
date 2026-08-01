"""Tests for Shadowspace BundleWriter, BundleReader, and BundleValidator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest

from shadowspace.bundle.reader import BundleReader, BundleValidator
from shadowspace.bundle.writer import BundleWriter
from shadowspace.generators.calibration import generate_calibration_bundle
from shadowspace.models.schemas import RepresentationSpec


def test_bundle_writer_and_reader_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        generate_calibration_bundle(tmp_dir)

        reader = BundleReader(tmp_dir)
        val_res = reader.validate()
        assert val_res.is_valid, f"Validation failed: {val_res.errors}"

        manifest = reader.manifest
        assert manifest.bundle_id == "calibration-3c-v1"
        assert len(manifest.representations) == 2

        obj_df = reader.get_objects()
        assert len(obj_df) == 15
        assert "object_id" in obj_df.columns

        prob_df = reader.get_representation("probability")
        assert len(prob_df) == 15
        assert "p0" in prob_df.columns

        prob_mat, ids = reader.get_representation_matrix("probability")
        assert prob_mat.shape == (15, 3)
        assert len(ids) == 15


def test_bundle_validator_detects_corrupted_sha256() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        generate_calibration_bundle(tmp_dir)

        # Corrupt one byte in probability.parquet
        prob_path = Path(tmp_dir) / "representations" / "probability.parquet"
        data = bytearray(prob_path.read_bytes())
        data[10] ^= 0xFF
        prob_path.write_bytes(data)

        validator = BundleValidator(tmp_dir)
        res = validator.validate()
        assert not res.is_valid
        assert any("SHA-256 mismatch" in err for err in res.errors)


def test_bundle_validator_detects_missing_file() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        generate_calibration_bundle(tmp_dir)

        # Delete objects.parquet
        obj_path = Path(tmp_dir) / "objects.parquet"
        obj_path.unlink()

        validator = BundleValidator(tmp_dir)
        res = validator.validate()
        assert not res.is_valid
        assert any("missing" in err for err in res.errors)


def test_bundle_validator_detects_reordered_ids() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir)
        writer = BundleWriter(out_path, bundle_id="test-reorder")

        obj_df = pl.DataFrame({"object_id": ["id_1", "id_2", "id_3"]})
        rep_df = pl.DataFrame(
            {
                "object_id": ["id_2", "id_1", "id_3"],  # Reordered!
                "f0": [0.5, 0.5, 0.5],
                "f1": [0.5, 0.5, 0.5],
            }
        )

        spec = RepresentationSpec(
            id="rep1",
            path="representations/rep1.parquet",
            dimension=2,
            feature_columns=["f0", "f1"],
            default_metric="euclidean",
        )

        writer.set_objects(obj_df)
        writer.add_representation("rep1", rep_df, spec)
        writer.write()

        validator = BundleValidator(out_path)
        res = validator.validate()
        assert not res.is_valid
        assert any("row order differs" in err for err in res.errors)


def test_bundle_validator_detects_constraint_violations() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir)
        writer = BundleWriter(out_path, bundle_id="test-constraints")

        obj_df = pl.DataFrame({"object_id": ["id_1", "id_2"]})
        # Invalid row sum and negative value
        rep_df = pl.DataFrame(
            {
                "object_id": ["id_1", "id_2"],
                "f0": [-0.1, 0.8],
                "f1": [0.5, 0.8],  # Row 1 sums to 0.4, Row 2 sums to 1.6
            }
        )

        spec = RepresentationSpec(
            id="rep1",
            path="representations/rep1.parquet",
            dimension=2,
            feature_columns=["f0", "f1"],
            constraints=["nonnegative", "row_sum_1"],
            default_metric="euclidean",
        )

        writer.set_objects(obj_df)
        writer.add_representation("rep1", rep_df, spec)
        writer.write()

        validator = BundleValidator(out_path)
        res = validator.validate()
        assert not res.is_valid
        assert any("nonnegative" in err for err in res.errors)
        assert any("sum to 1.0" in err for err in res.errors)


def test_bundle_reader_raises_key_error_for_unknown_rep() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        generate_calibration_bundle(tmp_dir)

        reader = BundleReader(tmp_dir)
        with pytest.raises(KeyError, match="does_not_exist"):
            reader.get_representation("does_not_exist")

        with pytest.raises(KeyError, match="does_not_exist"):
            reader.get_representation_matrix("does_not_exist")


def test_bundle_reader_raises_file_not_found_for_missing_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(FileNotFoundError):
            BundleReader(tmp_dir)


def test_bundle_writer_add_extra_artifact_records_in_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir)
        writer = BundleWriter(out_path, bundle_id="test-extra")

        obj_df = pl.DataFrame({"object_id": ["id_1"]})
        writer.set_objects(obj_df)
        writer.add_extra_artifact("readme", "README.md")
        writer.write()

        reader = BundleReader(tmp_dir)
        assert "readme" in reader.manifest.extra_artifacts
        assert reader.manifest.extra_artifacts["readme"] == "README.md"
