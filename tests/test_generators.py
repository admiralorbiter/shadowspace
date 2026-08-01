"""Tests for Shadowspace synthetic generators."""

from __future__ import annotations

import tempfile

import numpy as np
import polars.testing as pl_testing

from shadowspace.bundle.reader import BundleReader, BundleValidator
from shadowspace.conventions import PROB_SUM_ATOL
from shadowspace.generators.calibration import generate_calibration_bundle
from shadowspace.generators.synthetic import generate_synthetic_bundle


def test_calibration_generator_bundle() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = generate_calibration_bundle(tmp_dir)
        assert manifest_path.exists()

        validator = BundleValidator(tmp_dir)
        res = validator.validate()
        assert res.is_valid, f"Calibration bundle invalid: {res.errors}"

        reader = BundleReader(tmp_dir)
        objects = reader.get_objects()
        assert len(objects) == 15
        assert "generator_component" in objects.columns

        matrix, ids = reader.get_representation_matrix("probability")
        assert matrix.shape == (15, 3)
        assert len(ids) == 15
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=PROB_SUM_ATOL)


def test_synthetic_generator_determinism() -> None:
    with (
        tempfile.TemporaryDirectory() as tmp_dir1,
        tempfile.TemporaryDirectory() as tmp_dir2,
    ):
        generate_synthetic_bundle(tmp_dir1, seed=20260801, n_samples=500)
        generate_synthetic_bundle(tmp_dir2, seed=20260801, n_samples=500)

        reader1 = BundleReader(tmp_dir1)
        reader2 = BundleReader(tmp_dir2)

        df1 = reader1.get_representation("probability")
        df2 = reader2.get_representation("probability")

        pl_testing.assert_frame_equal(df1, df2)


def test_synthetic_generator_seed_variation() -> None:
    with (
        tempfile.TemporaryDirectory() as tmp_dir1,
        tempfile.TemporaryDirectory() as tmp_dir2,
    ):
        generate_synthetic_bundle(tmp_dir1, seed=20260801, n_samples=500)
        generate_synthetic_bundle(tmp_dir2, seed=999999, n_samples=500)

        reader1 = BundleReader(tmp_dir1)
        reader2 = BundleReader(tmp_dir2)

        mat1, _ = reader1.get_representation_matrix("probability")
        mat2, _ = reader2.get_representation_matrix("probability")

        assert not np.array_equal(mat1, mat2)


def test_synthetic_generator_latent_families_and_invariants() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        generate_synthetic_bundle(tmp_dir, seed=12345, n_samples=1000)

        validator = BundleValidator(tmp_dir)
        res = validator.validate()
        assert res.is_valid, f"Synthetic bundle invalid: {res.errors}"

        reader = BundleReader(tmp_dir)
        objects = reader.get_objects()
        components = set(objects["generator_component"].to_list())

        expected_families = {
            "confident_corner_0",
            "confident_corner_1",
            "confident_corner_2",
            "confident_corner_3",
            "ambiguity_band_01",
            "ambiguity_band_23",
            "high_entropy_center",
            "narrow_bridge_02",
            "evidence_trajectory",
            "isolated_outlier",
            "cluster_outlier",
        }
        assert expected_families.issubset(components)

        prob_mat, ids = reader.get_representation_matrix("probability")
        assert len(prob_mat) == len(objects)
        assert len(ids) == len(objects)
        assert prob_mat.shape[1] == 4
        assert np.all(prob_mat >= 0.0)
        assert np.all(np.isfinite(prob_mat))
        np.testing.assert_allclose(prob_mat.sum(axis=1), 1.0, atol=PROB_SUM_ATOL)
