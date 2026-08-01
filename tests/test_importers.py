"""Tests for shadowspace.importers module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from shadowspace.bundle.reader import BundleReader, BundleValidator
from shadowspace.cli import main as cli_main
from shadowspace.importers import (
    ImportValidationError,
    import_csv_bundle,
    import_parquet_bundle,
    validate_import_matrix,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURES_DIR / "sample_3class.csv"


def test_validate_import_matrix_valid() -> None:
    mat = np.array([[0.5, 0.5], [0.1, 0.9]], dtype=np.float64)
    res = validate_import_matrix(mat, n_classes=2)
    assert res.shape == (2, 2)
    assert np.allclose(np.sum(res, axis=1), 1.0)


def test_validate_import_matrix_shape_errors() -> None:
    # 1D matrix
    with pytest.raises(ImportValidationError, match="2D matrix"):
        validate_import_matrix(np.array([0.5, 0.5]))

    # Fewer than 2 features
    with pytest.raises(ImportValidationError, match="at least 2 class columns"):
        validate_import_matrix(np.array([[1.0]]))

    # Dimension mismatch
    with pytest.raises(ImportValidationError, match="Expected 3 class columns"):
        validate_import_matrix(np.array([[0.5, 0.5]]), n_classes=3)


def test_validate_import_matrix_non_finite() -> None:
    mat = np.array([[0.5, np.nan], [0.1, 0.9]])
    with pytest.raises(ImportValidationError, match="NaN or Infinite"):
        validate_import_matrix(mat)


def test_validate_import_matrix_negative() -> None:
    mat = np.array([[-0.1, 1.1], [0.1, 0.9]])
    with pytest.raises(ImportValidationError, match="cannot be negative"):
        validate_import_matrix(mat)


def test_validate_import_matrix_row_sums() -> None:
    mat = np.array([[0.8, 0.8], [0.1, 0.9]])
    with pytest.raises(ImportValidationError, match="probabilities sum to"):
        validate_import_matrix(mat)


def test_validate_import_matrix_normalize_softmax() -> None:
    logits = np.array([[2.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    probs = validate_import_matrix(logits, normalize=True)
    assert probs.shape == (2, 3)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    assert (probs >= 0.0).all()


def test_import_csv_bundle_valid() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "bundle"
        manifest_path = import_csv_bundle(
            csv_path=SAMPLE_CSV,
            output_dir=out_dir,
            id_column="object_id",
            label_column="true_label",
            feature_columns=["p0", "p1", "p2"],
            dataset_name="test_sample_3c",
        )

        assert manifest_path.exists()
        validator = BundleValidator(out_dir)
        val_res = validator.validate()
        assert val_res.is_valid, f"Bundle validation errors: {val_res.errors}"

        reader = BundleReader(out_dir)
        matrix, ids = reader.get_representation_matrix("probability")
        assert matrix.shape == (5, 3)
        assert len(ids) == 5
        assert ids[0] == "sample_0"

        # Check payload_image_bytes schema hook presence
        objects_df = reader.get_objects()
        assert "payload_image_bytes" in objects_df.columns


def test_import_csv_bundle_logits_normalization() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = Path(tmp_dir) / "logits.csv"
        csv_file.write_text("id,l0,l1\nobj1,3.0,1.0\nobj2,-0.5,0.5\n", encoding="utf-8")

        out_dir = Path(tmp_dir) / "bundle"
        manifest_path = import_csv_bundle(
            csv_path=csv_file,
            output_dir=out_dir,
            id_column="id",
            feature_columns=["l0", "l1"],
            normalize=True,
            dataset_name="logits_test",
        )

        assert manifest_path.exists()
        reader = BundleReader(out_dir)
        matrix, _ = reader.get_representation_matrix("probability")
        assert np.allclose(np.sum(matrix, axis=1), 1.0)


def test_import_parquet_bundle_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "csv_bundle"
        import_csv_bundle(
            csv_path=SAMPLE_CSV,
            output_dir=out_dir,
            id_column="object_id",
            feature_columns=["p0", "p1", "p2"],
        )

        # Convert objects.parquet to a standalone test parquet file
        reader = BundleReader(out_dir)
        mat, ids = reader.get_representation_matrix("probability")

        import polars as pl

        pq_df = pl.DataFrame(
            {"id": ids, "feat0": mat[:, 0], "feat1": mat[:, 1], "feat2": mat[:, 2]}
        )
        pq_path = Path(tmp_dir) / "input.parquet"
        pq_df.write_parquet(pq_path)

        pq_bundle_dir = Path(tmp_dir) / "pq_bundle"
        import_parquet_bundle(
            parquet_path=pq_path,
            output_dir=pq_bundle_dir,
            id_column="id",
            feature_columns=["feat0", "feat1", "feat2"],
        )

        val = BundleValidator(pq_bundle_dir).validate()
        assert val.is_valid


def test_cli_import_csv_command() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / "cli_bundle"
        res = cli_main(
            [
                "import-csv",
                "--input",
                str(SAMPLE_CSV),
                "--output",
                str(out_dir),
                "--id-col",
                "object_id",
                "--label-col",
                "true_label",
                "--feature-cols",
                "p0,p1,p2",
                "--name",
                "cli_test",
            ]
        )
        assert res == 0
        assert (out_dir / "manifest.json").exists()
