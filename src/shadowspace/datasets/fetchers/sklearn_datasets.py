"""scikit-learn dataset fetchers for Shadowspace benchmark bundle generation."""

import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from shadowspace.datasets.classifier import fit_baseline_classifier
from shadowspace.datasets.registry import REGISTRY, DatasetSpec
from shadowspace.importers.csv_importer import import_csv_bundle


def _build_and_export_bundle(
    spec: DatasetSpec,
    x_mat: NDArray[np.float64],
    y: NDArray[np.int64],
    target_names: list[str],
    output_dir: Path,
    seed: int,
) -> Path:
    """Fit baseline model on x_mat, y and export Shadowspace bundle to output_dir/key."""
    _clf, proba_matrix = fit_baseline_classifier(x_mat, y, seed=seed)

    n_samples, n_classes = proba_matrix.shape
    object_ids = [f"{spec.key}_{i:05d}" for i in range(n_samples)]
    true_labels = [target_names[int(idx)] for idx in y]

    # Build Polars DataFrame dictionary
    data_dict = {
        "object_id": object_ids,
        "true_label": true_labels,
    }
    feature_cols = []
    for k in range(n_classes):
        col_name = f"p_{target_names[k]}" if k < len(target_names) else f"p_{k}"
        data_dict[col_name] = proba_matrix[:, k].tolist()
        feature_cols.append(col_name)

    df = pl.DataFrame(data_dict)

    target_bundle_dir = output_dir / spec.key
    target_bundle_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp_csv_path = Path(tmp.name)

    try:
        df.write_csv(tmp_csv_path)
        manifest_path = import_csv_bundle(
            csv_path=tmp_csv_path,
            output_dir=target_bundle_dir,
            id_column="object_id",
            label_column="true_label",
            feature_columns=feature_cols,
            n_classes=n_classes,
            normalize=False,
            dataset_name=spec.key,
            description=spec.description,
        )
    finally:
        if tmp_csv_path.exists():
            tmp_csv_path.unlink()

    return manifest_path


def fetch_iris(output_dir: Path, seed: int = 20260801) -> Path:
    """Fetch Fisher's Iris dataset, fit baseline model, and write bundle."""
    from sklearn.datasets import load_iris  # type: ignore[import-untyped]

    data = load_iris()
    target_names = [str(name) for name in data.target_names]
    return _build_and_export_bundle(
        REGISTRY["iris_3class"],
        data.data,
        data.target,
        target_names,
        output_dir,
        seed,
    )


def fetch_digits(output_dir: Path, seed: int = 20260801) -> Path:
    """Fetch Handwritten Digits dataset, fit baseline model, and write bundle."""
    from sklearn.datasets import load_digits

    data = load_digits()
    target_names = [f"digit_{name}" for name in data.target_names]
    return _build_and_export_bundle(
        REGISTRY["digits_10class"],
        data.data,
        data.target,
        target_names,
        output_dir,
        seed,
    )


def fetch_wine(output_dir: Path, seed: int = 20260801) -> Path:
    """Fetch Wine Recognition dataset, fit baseline model, and write bundle."""
    from sklearn.datasets import load_wine

    data = load_wine()
    # sklearn only labels these class_0/1/2; give them the actual cultivar designations
    target_names = ["Cultivar_I", "Cultivar_II", "Cultivar_III"]
    return _build_and_export_bundle(
        REGISTRY["wine_3class"],
        data.data,
        data.target,
        target_names,
        output_dir,
        seed,
    )


def fetch_covertype(output_dir: Path, seed: int = 20260801, n_samples: int = 10000) -> Path:
    """Fetch Forest Cover Type dataset, subsample 10,000 stratified, fit model, and write bundle."""
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

    data = fetch_covtype()
    # Labels in covtype are 1..7, map to 0..6
    y_zero_indexed = data.target - 1
    target_names = [f"cover_type_{i + 1}" for i in range(7)]

    # Stratified subsample
    x_sub, _, y_sub, _ = train_test_split(
        data.data,
        y_zero_indexed,
        train_size=min(n_samples, len(data.data)),
        random_state=seed,
        stratify=y_zero_indexed,
    )

    return _build_and_export_bundle(
        REGISTRY["covertype_7class"],
        x_sub,
        y_sub,
        target_names,
        output_dir,
        seed,
    )


def fetch_dataset(
    key: str, output_dir: Path | str, seed: int = 20260801, force: bool = False
) -> Path:
    """Fetch/generate a benchmark dataset bundle by key."""
    if key not in REGISTRY:
        raise KeyError(f"Unknown dataset key '{key}'. Available keys: {list(REGISTRY.keys())}")

    out_p = Path(output_dir)
    spec = REGISTRY[key]
    target_manifest = out_p / spec.key / "manifest.json"

    if target_manifest.exists() and not force:
        return target_manifest

    fetchers: dict[str, Callable[[Path, int], Path]] = {
        "fetch_iris": fetch_iris,
        "fetch_digits": fetch_digits,
        "fetch_wine": fetch_wine,
        "fetch_covertype": fetch_covertype,
    }

    fetcher_fn = fetchers[spec.source_fn]
    return fetcher_fn(out_p, seed)
