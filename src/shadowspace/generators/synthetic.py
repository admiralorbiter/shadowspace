"""4-Class synthetic belief world generator.

Generates separable, ground-truth latent families over 4 probability classes:
- Confident corners (4 classes)
- Pairwise ambiguity bands
- High-entropy center
- Narrow bridge between populations
- Evidence updating trajectory
- Outliers (isolated and cluster)

Guarantees exact determinism when provided with the same seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from shadowspace.bundle.writer import BundleWriter
from shadowspace.conventions import CLR_ZERO_DELTA, CLR_ZERO_MATCH, CLR_ZERO_POLICY
from shadowspace.math.clr import clr_transform
from shadowspace.models.schemas import MetricSpec, RepresentationSpec, ZeroPolicy


def generate_synthetic_bundle(
    output_dir: Path | str,
    n_classes: int = 4,
    seed: int = 20260801,
    n_samples: int = 2000,
) -> Path:
    """Generate a synthetic belief world artifact bundle.

    Args:
        output_dir: Directory to save the bundle.
        n_classes: Number of probability classes (currently supports 4).
        seed: Random seed for exact reproducibility.
        n_samples: Target number of objects. Actual count may differ by up to ~10%
            due to integer rounding across the 7 population segments. The bundle
            manifest records the actual count.

    Returns:
        Path to output manifest.json.
    """
    if n_classes != 4:
        raise ValueError("Synthetic generator currently supports n_classes=4.")

    out_path = Path(output_dir)
    rng = np.random.default_rng(seed)

    probs_list: list[NDArray[np.float64]] = []
    components_list: list[str] = []
    true_labels_list: list[str] = []

    # Calculate proportions
    def _add_dirichlet(alpha: list[float], count: int, comp_name: str, true_c: int) -> None:
        if count <= 0:
            return
        samples = rng.dirichlet(alpha, size=count)
        probs_list.append(samples)
        components_list.extend([comp_name] * count)
        true_labels_list.extend([f"class_{true_c}"] * count)

    # 1. Confident corners (15% each = 60%)
    n_corner = int(n_samples * 0.15)
    _add_dirichlet([50.0, 2.0, 2.0, 2.0], n_corner, "confident_corner_0", 0)
    _add_dirichlet([2.0, 50.0, 2.0, 2.0], n_corner, "confident_corner_1", 1)
    _add_dirichlet([2.0, 2.0, 50.0, 2.0], n_corner, "confident_corner_2", 2)
    _add_dirichlet([2.0, 2.0, 2.0, 50.0], n_corner, "confident_corner_3", 3)

    # 2. Pairwise ambiguity bands (10% each = 20%)
    n_ambig = int(n_samples * 0.10)
    _add_dirichlet([25.0, 25.0, 1.0, 1.0], n_ambig, "ambiguity_band_01", 0)
    _add_dirichlet([1.0, 1.0, 25.0, 25.0], n_ambig, "ambiguity_band_23", 2)

    # 3. High-entropy center (8%)
    # Label is 'ambiguous' because no class is dominant by design.
    n_center = int(n_samples * 0.08)
    if n_center > 0:
        center_samples = rng.dirichlet([8.0, 8.0, 8.0, 8.0], size=n_center)
        probs_list.append(center_samples)
        components_list.extend(["high_entropy_center"] * n_center)
        true_labels_list.extend(["ambiguous"] * n_center)

    # 4. Narrow bridge between Class 0 and Class 2 (5%)
    n_bridge = int(n_samples * 0.05)
    t = np.linspace(0.1, 0.9, n_bridge)
    bridge_probs = np.zeros((n_bridge, 4))
    bridge_probs[:, 0] = 1.0 - t
    bridge_probs[:, 2] = t
    bridge_probs[:, 1] = 0.01
    bridge_probs[:, 3] = 0.01
    bridge_probs /= bridge_probs.sum(axis=1, keepdims=True)
    probs_list.append(bridge_probs)
    components_list.extend(["narrow_bridge_02"] * n_bridge)
    true_labels_list.extend([f"class_{0 if ti < 0.5 else 2}" for ti in t])

    # 5. Evidence updating trajectory (2%)
    n_traj = int(n_samples * 0.02)
    steps = np.linspace(0.0, 1.0, n_traj)
    traj_probs = np.zeros((n_traj, 4))
    for idx, step in enumerate(steps):
        # Starts uniform (0.25 each), transitions to class 1 dominant
        vec = np.array(
            [0.25 * (1 - step), 0.25 + 0.75 * step, 0.25 * (1 - step), 0.25 * (1 - step)]
        )
        traj_probs[idx] = vec / vec.sum()
    probs_list.append(traj_probs)
    components_list.extend(["evidence_trajectory"] * n_traj)
    true_labels_list.extend(["class_1"] * n_traj)

    # 6. Isolated outliers (1%)
    n_outlier = max(1, int(n_samples * 0.01))
    outlier_probs = rng.dirichlet([0.2, 0.2, 0.2, 0.2], size=n_outlier)
    probs_list.append(outlier_probs)
    components_list.extend(["isolated_outlier"] * n_outlier)
    true_labels_list.extend([f"class_{int(np.argmax(op))}" for op in outlier_probs])

    # 7. Cluster outliers (2%)
    n_cluster_outlier = int(n_samples * 0.02)
    _add_dirichlet([1.0, 15.0, 15.0, 1.0], n_cluster_outlier, "cluster_outlier", 1)

    # Combine all generated arrays
    mat = np.vstack(probs_list)
    total_generated = len(mat)
    ids = [f"synth_{i:05d}" for i in range(total_generated)]

    predicted_labels = [f"class_{int(np.argmax(mat[i]))}" for i in range(total_generated)]
    # 'ambiguous' entries (high_entropy_center) have no dominant class and are never 'correct'
    correct = [
        (t == p if t != "ambiguous" else False)
        for t, p in zip(true_labels_list, predicted_labels, strict=True)
    ]

    # Entropy calculation
    safe_p = np.where(mat > 0, mat, 1.0)
    entropy = -np.sum(np.where(mat > 0, mat * np.log2(safe_p), 0.0), axis=1)

    # Objects table
    objects_df = pl.DataFrame(
        {
            "object_id": ids,
            "generator_component": components_list,
            "true_label": true_labels_list,
            "predicted_label": predicted_labels,
            "correct": correct,
            "entropy": entropy.tolist(),
        }
    )

    # Probability representation table
    prob_df = pl.DataFrame(
        {
            "object_id": ids,
            "p0": mat[:, 0].tolist(),
            "p1": mat[:, 1].tolist(),
            "p2": mat[:, 2].tolist(),
            "p3": mat[:, 3].tolist(),
        }
    )

    prob_spec = RepresentationSpec(
        id="probability",
        path="representations/probability.parquet",
        dimension=4,
        object_id_column="object_id",
        feature_columns=["p0", "p1", "p2", "p3"],
        constraints=["finite", "nonnegative", "row_sum_1"],
        compatible_metrics=["euclidean", "fisher_rao"],
        default_metric="fisher_rao",
    )

    # CLR representation table
    clr_mat = clr_transform(mat)
    clr_df = pl.DataFrame(
        {
            "object_id": ids,
            "clr0": clr_mat[:, 0].tolist(),
            "clr1": clr_mat[:, 1].tolist(),
            "clr2": clr_mat[:, 2].tolist(),
            "clr3": clr_mat[:, 3].tolist(),
        }
    )

    clr_spec = RepresentationSpec(
        id="clr_probability",
        path="representations/clr_probability.parquet",
        dimension=4,
        object_id_column="object_id",
        feature_columns=["clr0", "clr1", "clr2", "clr3"],
        constraints=["finite"],
        compatible_metrics=["euclidean", "aitchison"],
        default_metric="aitchison",
        zero_policy=ZeroPolicy(
            policy=CLR_ZERO_POLICY,
            delta=CLR_ZERO_DELTA,
            match=CLR_ZERO_MATCH,
        ),
    )

    # Metrics
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
        bundle_id=f"synthetic-4c-seed{seed}",
        description="Controlled 4-class synthetic probability belief world bundle.",
    )

    writer.set_objects(objects_df)
    writer.add_representation("probability", prob_df, prob_spec)
    writer.add_representation("clr_probability", clr_df, clr_spec)
    writer.add_metric(fr_metric)
    writer.add_metric(aitchison_metric)

    return writer.write()
