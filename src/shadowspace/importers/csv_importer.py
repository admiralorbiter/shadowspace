"""CSV and Parquet data importer for Shadowspace bundle generation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from shadowspace.bundle.writer import BundleWriter
from shadowspace.conventions import CLR_ZERO_DELTA, CLR_ZERO_MATCH, CLR_ZERO_POLICY
from shadowspace.importers.validator import validate_import_matrix
from shadowspace.math.clr import clr_transform
from shadowspace.models.schemas import MetricSpec, RepresentationSpec, ZeroPolicy


def import_csv_bundle(
    csv_path: Path | str,
    output_dir: Path | str,
    id_column: str | None = None,
    label_column: str | None = None,
    feature_columns: Sequence[str] | None = None,
    n_classes: int | None = None,
    normalize: bool = False,
    dataset_name: str = "imported_dataset",
    description: str = "Imported dataset bundle",
) -> Path:
    """Imports a CSV file containing model predictions/probabilities into a Shadowspace bundle.

    Args:
        csv_path: Path to the input CSV file.
        output_dir: Directory where the generated bundle will be stored.
        id_column: Name of the column containing object IDs. If None, auto-generated IDs are used.
        label_column: Optional name of column containing ground truth labels.
        feature_columns: Names of columns containing probabilities/logits. If None, auto-detected.
        n_classes: Expected number of classes K.
        normalize: If True, applies softmax row-wise to feature_columns (for raw logits).
        dataset_name: Bundle identifier string.
        description: Description of the dataset.

    Returns:
        Path: Path to manifest.json of the generated bundle.
    """
    csv_p = Path(csv_path)
    df = pl.read_csv(csv_p)
    return _create_bundle_from_dataframe(
        df=df,
        output_dir=output_dir,
        id_column=id_column,
        label_column=label_column,
        feature_columns=feature_columns,
        n_classes=n_classes,
        normalize=normalize,
        dataset_name=dataset_name,
        description=description,
    )


def import_parquet_bundle(
    parquet_path: Path | str,
    output_dir: Path | str,
    id_column: str | None = None,
    label_column: str | None = None,
    feature_columns: Sequence[str] | None = None,
    n_classes: int | None = None,
    normalize: bool = False,
    dataset_name: str = "imported_dataset",
    description: str = "Imported dataset bundle",
) -> Path:
    """Imports a Parquet file containing model predictions/probabilities into a Shadowspace bundle."""
    parquet_p = Path(parquet_path)
    df = pl.read_parquet(parquet_p)
    return _create_bundle_from_dataframe(
        df=df,
        output_dir=output_dir,
        id_column=id_column,
        label_column=label_column,
        feature_columns=feature_columns,
        n_classes=n_classes,
        normalize=normalize,
        dataset_name=dataset_name,
        description=description,
    )


def _create_bundle_from_dataframe(
    df: pl.DataFrame,
    output_dir: Path | str,
    id_column: str | None,
    label_column: str | None,
    feature_columns: Sequence[str] | None,
    n_classes: int | None,
    normalize: bool,
    dataset_name: str,
    description: str,
) -> Path:
    """Internal helper to convert a Polars DataFrame into a validated bundle."""
    # 1. Resolve object_ids
    if id_column and id_column in df.columns:
        ids = [str(val) for val in df[id_column].to_list()]
    else:
        ids = [f"obj_{i}" for i in range(len(df))]

    # 2. Resolve feature columns
    if feature_columns:
        feat_cols = list(feature_columns)
    else:
        # Exclude metadata columns
        excluded = {id_column, label_column, "object_id", "id"}
        feat_cols = [c for c in df.columns if c not in excluded]

    # Convert features to numpy array
    raw_matrix = df.select(feat_cols).to_numpy().astype(np.float64)

    # 3. Validate and normalize matrix
    prob_matrix = validate_import_matrix(
        raw_matrix,
        n_classes=n_classes,
        normalize=normalize,
    )
    n_rows, k_dim = prob_matrix.shape

    # 4. Compute labels, entropy, confidence
    # Derive class name list from feature column names where possible
    class_names = [c.removeprefix("p_") for c in feat_cols]
    predicted_class_indices = np.argmax(prob_matrix, axis=1)
    predicted_labels = [class_names[idx] if idx < len(class_names) else f"class_{idx}" for idx in predicted_class_indices]

    if label_column and label_column in df.columns:
        true_labels = [str(val) for val in df[label_column].to_list()]
        correct = [t == p for t, p in zip(true_labels, predicted_labels, strict=False)]
    else:
        true_labels = predicted_labels.copy()
        correct = [True] * n_rows

    # Shannon entropy
    safe_p = np.where(prob_matrix > 0, prob_matrix, 1.0)
    entropy = -np.sum(np.where(prob_matrix > 0, prob_matrix * np.log2(safe_p), 0.0), axis=1)
    confidence = np.max(prob_matrix, axis=1)

    # Build objects DataFrame with optional payload_image_bytes schema hook
    objects_df = pl.DataFrame(
        {
            "object_id": ids,
            "generator_component": ["imported"] * n_rows,
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "correct": correct,
            "confidence": confidence.tolist(),
            "entropy": entropy.tolist(),
            "payload_image_bytes": [None] * n_rows,  # Schema hook for Sprint 9+ thumbnails
        }
    )

    # 5. Build Probability Representation Table
    # Use named feature columns when provided (e.g. p_setosa, p_versicolor) otherwise p0..pk
    prob_col_names = feat_cols if feature_columns else [f"p{i}" for i in range(k_dim)]
    prob_data = {"object_id": ids}
    for i, col_name in enumerate(prob_col_names):
        prob_data[col_name] = prob_matrix[:, i].tolist()
    prob_df = pl.DataFrame(prob_data)

    prob_spec = RepresentationSpec(
        id="probability",
        path="representations/probability.parquet",
        dimension=k_dim,
        object_id_column="object_id",
        feature_columns=list(prob_col_names),
        constraints=["finite", "nonnegative", "row_sum_1"],
        compatible_metrics=["euclidean", "fisher_rao"],
        default_metric="fisher_rao",
    )

    # 6. Build CLR Representation Table
    clr_matrix = clr_transform(prob_matrix)
    clr_col_names = [f"clr{i}" for i in range(k_dim)]
    clr_data = {"object_id": ids}
    for i, col_name in enumerate(clr_col_names):
        clr_data[col_name] = clr_matrix[:, i].tolist()
    clr_df = pl.DataFrame(clr_data)

    clr_spec = RepresentationSpec(
        id="clr_probability",
        path="representations/clr_probability.parquet",
        dimension=k_dim,
        object_id_column="object_id",
        feature_columns=clr_col_names,
        constraints=["finite"],
        compatible_metrics=["euclidean", "aitchison"],
        default_metric="aitchison",
        zero_policy=ZeroPolicy(
            policy=CLR_ZERO_POLICY,
            delta=CLR_ZERO_DELTA,
            match=CLR_ZERO_MATCH,
        ),
    )

    # 7. Metrics
    fr_metric = MetricSpec(
        id="fisher_rao",
        display_name="Fisher-Rao Distance",
        representation_ids=["probability"],
        is_metric=True,
        parameters={"scale": 2.0, "convention": "canonical_fisher_information"},
        units_or_scale="radians",
    )

    aitchison_metric = MetricSpec(
        id="aitchison",
        display_name="Aitchison Distance",
        representation_ids=["clr_probability"],
        is_metric=True,
        parameters={"zero_policy": CLR_ZERO_POLICY, "delta": CLR_ZERO_DELTA},
    )

    # 8. Write bundle
    writer = BundleWriter(
        output_dir=Path(output_dir),
        bundle_id=dataset_name,
        description=f"{description} (normalized={normalize})",
    )

    writer.set_objects(objects_df)
    writer.add_representation("probability", prob_df, prob_spec)
    writer.add_representation("clr_probability", clr_df, clr_spec)
    writer.add_metric(fr_metric)
    writer.add_metric(aitchison_metric)

    return writer.write()
