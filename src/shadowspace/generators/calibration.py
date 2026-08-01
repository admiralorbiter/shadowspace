"""3-Class calibration bundle generator.

Wraps the canonical 15-point calibration fixture into a Shadowspace artifact bundle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from shadowspace.bundle.writer import BundleWriter
from shadowspace.conventions import CLR_ZERO_DELTA, CLR_ZERO_MATCH, CLR_ZERO_POLICY
from shadowspace.data.calibration import calibration_fixture
from shadowspace.math.clr import clr_transform
from shadowspace.models.schemas import MetricSpec, RepresentationSpec, ZeroPolicy


def generate_calibration_bundle(output_dir: Path | str) -> Path:
    """Generate the 3-class calibration artifact bundle.

    Returns:
        Path to output manifest.json.
    """
    out_path = Path(output_dir)
    matrix, ids = calibration_fixture()
    n_rows = len(ids)

    # 1. Objects table
    # Assign components to fixture points
    components = []
    true_labels = []
    for i, id_ in enumerate(ids):
        dominant_class = int(np.argmax(matrix[i]))
        if id_.startswith("corner"):
            components.append("corner")
            true_labels.append(f"class_{dominant_class}")
        elif id_.startswith("midpoint"):
            components.append("midpoint")
            true_labels.append(f"class_{dominant_class}")
        elif id_ == "center":
            components.append("uniform_center")
            true_labels.append("ambiguous")
        else:
            components.append("interior")
            true_labels.append(f"class_{dominant_class}")

    predicted_labels = [f"class_{int(np.argmax(matrix[i]))}" for i in range(n_rows)]
    # center point is 'ambiguous' — it has no dominant class
    correct = [
        (t == p if t != "ambiguous" else False)
        for t, p in zip(true_labels, predicted_labels, strict=True)
    ]

    # Calculate Shannon entropy (base 2)
    # p * log2(p) with 0 log2(0) = 0
    safe_p = np.where(matrix > 0, matrix, 1.0)
    entropy = -np.sum(np.where(matrix > 0, matrix * np.log2(safe_p), 0.0), axis=1)

    objects_df = pl.DataFrame(
        {
            "object_id": ids,
            "generator_component": components,
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "correct": correct,
            "entropy": entropy.tolist(),
        }
    )

    # 2. Probability representation table
    prob_df = pl.DataFrame(
        {
            "object_id": ids,
            "p0": matrix[:, 0].tolist(),
            "p1": matrix[:, 1].tolist(),
            "p2": matrix[:, 2].tolist(),
        }
    )

    prob_spec = RepresentationSpec(
        id="probability",
        path="representations/probability.parquet",
        dimension=3,
        object_id_column="object_id",
        feature_columns=["p0", "p1", "p2"],
        constraints=["finite", "nonnegative", "row_sum_1"],
        compatible_metrics=["euclidean", "fisher_rao"],
        default_metric="fisher_rao",
    )

    # 3. CLR representation table
    clr_mat = clr_transform(matrix)
    clr_df = pl.DataFrame(
        {
            "object_id": ids,
            "clr0": clr_mat[:, 0].tolist(),
            "clr1": clr_mat[:, 1].tolist(),
            "clr2": clr_mat[:, 2].tolist(),
        }
    )

    clr_spec = RepresentationSpec(
        id="clr_probability",
        path="representations/clr_probability.parquet",
        dimension=3,
        object_id_column="object_id",
        feature_columns=["clr0", "clr1", "clr2"],
        constraints=["finite"],
        compatible_metrics=["euclidean", "aitchison"],
        default_metric="aitchison",
        zero_policy=ZeroPolicy(
            policy=CLR_ZERO_POLICY,
            delta=CLR_ZERO_DELTA,
            match=CLR_ZERO_MATCH,
        ),
    )

    # 4. Metrics
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

    # Write bundle
    writer = BundleWriter(
        output_dir=out_path,
        bundle_id="calibration-3c-v1",
        description="Canonical 15-point 3-class simplex calibration bundle.",
    )

    writer.set_objects(objects_df)
    writer.add_representation("probability", prob_df, prob_spec)
    writer.add_representation("clr_probability", clr_df, clr_spec)
    writer.add_metric(fr_metric)
    writer.add_metric(aitchison_metric)

    return writer.write()
