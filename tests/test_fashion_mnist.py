"""Tests for Sprint 7 — Fashion-MNIST 10-class prediction belief space."""

from __future__ import annotations

import tempfile

import numpy as np

from shadowspace.bundle.reader import BundleReader
from shadowspace.generators.fashion_mnist import FASHION_CLASSES, generate_fashion_mnist_bundle


def test_generate_fashion_mnist_bundle_shape_and_invariants() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = generate_fashion_mnist_bundle(output_dir=tmpdir, seed=42, n_samples=200)
        assert manifest_path.exists()

        reader = BundleReader(tmpdir)
        matrix, object_ids = reader.get_representation_matrix("probability")

        assert matrix.shape == (200, 10)
        assert len(object_ids) == 200

        # Non-negativity invariant
        assert np.all(matrix >= 0.0)

        # Row sum to 1.0 invariant
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-6)

        # Validate object table metadata
        obj_df = reader.get_objects()
        assert "pred_class_name" in obj_df.columns
        assert "is_correct" in obj_df.columns
        assert "confidence" in obj_df.columns
        assert "entropy" in obj_df.columns
        assert len(obj_df) == 200


def test_fashion_mnist_class_labels_count() -> None:
    assert len(FASHION_CLASSES) == 10
    assert "T-shirt/top" in FASHION_CLASSES
    assert "Ankle boot" in FASHION_CLASSES
