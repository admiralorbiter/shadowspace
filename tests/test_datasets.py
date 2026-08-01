"""Tests for shadowspace.datasets package, bundle discovery, and CLI datasets commands."""

import tempfile
from pathlib import Path

import pytest

from shadowspace.bundle.reader import BundleReader, BundleValidator
from shadowspace.cli import main as cli_main
from shadowspace.datasets.bundle_discovery import scan_bundle_dir
from shadowspace.datasets.fetchers.sklearn_datasets import fetch_dataset
from shadowspace.datasets.registry import REGISTRY, DatasetSpec


def test_registry_completeness() -> None:
    """Verify all DatasetSpec entries in REGISTRY have valid attributes."""
    assert "iris_3class" in REGISTRY
    assert "digits_10class" in REGISTRY
    assert "wine_3class" in REGISTRY
    assert "covertype_7class" in REGISTRY

    for key, spec in REGISTRY.items():
        assert isinstance(spec, DatasetSpec)
        assert spec.key == key
        assert len(spec.display_name) > 0
        assert spec.n_classes > 1
        assert len(spec.description) > 0
        assert len(spec.source_fn) > 0


def test_fetch_iris_dataset() -> None:
    """Test fetching Fisher's Iris dataset into a Shadowspace bundle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        manifest_p = fetch_dataset("iris_3class", out_dir, seed=42, force=True)

        assert manifest_p.exists()
        bundle_dir = manifest_p.parent
        val = BundleValidator(bundle_dir).validate()
        assert val.is_valid, f"Iris bundle validation errors: {val.errors}"

        reader = BundleReader(bundle_dir)
        mat, ids = reader.get_representation_matrix("probability")
        assert mat.shape == (150, 3)
        assert len(ids) == 150
        assert np_all_close_1(mat)


def test_fetch_digits_dataset() -> None:
    """Test fetching Digits dataset into a Shadowspace bundle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        manifest_p = fetch_dataset("digits_10class", out_dir, seed=42, force=True)

        assert manifest_p.exists()
        bundle_dir = manifest_p.parent
        val = BundleValidator(bundle_dir).validate()
        assert val.is_valid, f"Digits bundle validation errors: {val.errors}"

        reader = BundleReader(bundle_dir)
        mat, ids = reader.get_representation_matrix("probability")
        assert mat.shape == (1797, 10)
        assert len(ids) == 1797


def test_fetch_wine_dataset() -> None:
    """Test fetching Wine dataset into a Shadowspace bundle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        manifest_p = fetch_dataset("wine_3class", out_dir, seed=42, force=True)

        assert manifest_p.exists()
        bundle_dir = manifest_p.parent
        val = BundleValidator(bundle_dir).validate()
        assert val.is_valid, f"Wine bundle validation errors: {val.errors}"

        reader = BundleReader(bundle_dir)
        mat, ids = reader.get_representation_matrix("probability")
        assert mat.shape == (178, 3)
        assert len(ids) == 178


def test_scan_bundle_dir() -> None:
    """Test scanning a directory for valid artifact bundles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        # Empty scan
        assert scan_bundle_dir(out_dir) == {}
        assert scan_bundle_dir(out_dir / "nonexistent") == {}

        # Fetch one bundle into tmpdir
        fetch_dataset("iris_3class", out_dir, seed=42)

        discovered = scan_bundle_dir(out_dir)
        assert "iris_3class" in discovered
        assert discovered["iris_3class"].name == "manifest.json"


def test_fetch_dataset_unknown_key() -> None:
    """Test that fetching an unknown key raises KeyError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(KeyError, match="Unknown dataset key"):
            fetch_dataset("unknown_key_123", tmpdir)


def test_cli_datasets_list() -> None:
    """Test `shadowspace datasets list` CLI command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        code = cli_main(["datasets", "list", "--bundle-dir", tmpdir])
        assert code == 0


def test_cli_datasets_info() -> None:
    """Test `shadowspace datasets info` CLI command."""
    code = cli_main(["datasets", "info", "iris_3class"])
    assert code == 0

    code_err = cli_main(["datasets", "info", "nonexistent_key"])
    assert code_err == 1


def test_cli_datasets_fetch() -> None:
    """Test `shadowspace datasets fetch` CLI command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        code = cli_main(["datasets", "fetch", "--datasets", "iris_3class", "--output", tmpdir])
        assert code == 0
        assert (Path(tmpdir) / "iris_3class" / "manifest.json").exists()


def np_all_close_1(matrix) -> bool:
    import numpy as np

    return bool(np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5))
