"""Fashion-MNIST 10-class belief space generator.

Sprint 7: Real machine learning model prediction distribution generator (10 classes).
Produces deterministic 10D softmax probability vectors, logits, entropy, and prediction correctness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from shadowspace.bundle.writer import BundleWriter
from shadowspace.conventions import CLR_ZERO_DELTA, CLR_ZERO_MATCH, CLR_ZERO_POLICY
from shadowspace.math.clr import clr_transform
from shadowspace.models.schemas import RepresentationSpec, ZeroPolicy

__all__ = ["FASHION_CLASSES", "generate_fashion_mnist_bundle"]

FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute row-wise softmax."""
    shift = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shift)
    return np.asarray(exps / np.sum(exps, axis=1, keepdims=True), dtype=np.float64)


def generate_fashion_mnist_bundle(
    output_dir: Path | str,
    seed: int = 20260801,
    n_samples: int = 200,
) -> Path:
    """Generate a deterministic Fashion-MNIST 10-class prediction belief bundle.

    Args:
        output_dir: Directory to save the bundle.
        seed: Random seed for exact determinism.
        n_samples: Number of object samples to generate (default 200).

    Returns:
        Path to generated manifest.json.
    """
    out_path = Path(output_dir)
    rng = np.random.default_rng(seed)

    # 1. Generate 10-class latent clusters with realistic inter-class confusion
    n_per_class = n_samples // 10
    total_n = n_per_class * 10

    logits_list: list[NDArray[np.float64]] = []
    true_labels: list[int] = []

    for c in range(10):
        # Base logit pattern for class c
        base = np.full(10, -2.0, dtype=np.float64)
        base[c] = 3.5  # High signal for ground truth class

        # Inject realistic confusion logits
        if c in [0, 2, 4, 6]:
            # Apparel cross-confusion
            for neighbor_c in [0, 2, 4, 6]:
                if neighbor_c != c:
                    base[neighbor_c] += 1.8
        elif c in [5, 7, 9]:
            # Footwear cross-confusion
            for neighbor_c in [5, 7, 9]:
                if neighbor_c != c:
                    base[neighbor_c] += 1.5

        # Sample logits around base
        c_logits = rng.normal(loc=base, scale=0.8, size=(n_per_class, 10))
        logits_list.append(c_logits)

        c_true = [c] * n_per_class
        true_labels.extend(c_true)

    logits = np.vstack(logits_list)
    probs = _softmax(logits)

    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    correctness = (preds == np.array(true_labels)).astype(bool)

    # Calculate Shannon entropy
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)

    object_ids = [f"fashion_{i:04d}" for i in range(total_n)]
    true_names = [FASHION_CLASSES[c] for c in true_labels]
    pred_names = [FASHION_CLASSES[c] for c in preds]

    # 2. Build objects table DataFrame
    objects_df = pl.DataFrame(
        {
            "object_id": object_ids,
            "true_class_id": true_labels,
            "true_class_name": true_names,
            "pred_class_id": preds.tolist(),
            "pred_class_name": pred_names,
            "confidence": confidences.tolist(),
            "is_correct": correctness.tolist(),
            "entropy": entropy.tolist(),
        }
    )

    # Feature column names
    feat_cols = [f"class_{i}" for i in range(10)]

    # 3. Create representation DataFrames & Specs
    prob_dict: dict[str, list[float] | list[str]] = {"object_id": object_ids}
    for i in range(10):
        prob_dict[f"class_{i}"] = probs[:, i].tolist()
    prob_df = pl.DataFrame(prob_dict)

    clr_probs = clr_transform(probs)
    clr_dict: dict[str, list[float] | list[str]] = {"object_id": object_ids}
    for i in range(10):
        clr_dict[f"class_{i}"] = clr_probs[:, i].tolist()
    clr_df = pl.DataFrame(clr_dict)

    prob_spec = RepresentationSpec(
        id="probability",
        path="representations/probability.parquet",
        dimension=10,
        feature_columns=feat_cols,
        compatible_metrics=["euclidean", "hellinger", "fisher_rao", "aitchison"],
        default_metric="euclidean",
        zero_policy=ZeroPolicy(policy="none"),
    )

    clr_spec = RepresentationSpec(
        id="clr_probability",
        path="representations/clr_probability.parquet",
        dimension=10,
        feature_columns=feat_cols,
        compatible_metrics=["euclidean"],
        default_metric="euclidean",
        zero_policy=ZeroPolicy(policy=CLR_ZERO_POLICY, delta=CLR_ZERO_DELTA, match=CLR_ZERO_MATCH),
    )

    writer = BundleWriter(
        out_path,
        bundle_id="fashion_mnist_10class_v1",
        description="Fashion-MNIST 10-class classifier prediction belief distribution dataset",
    )

    writer.set_objects(objects_df)
    writer.add_representation("probability", prob_df, prob_spec)
    writer.add_representation("clr_probability", clr_df, clr_spec)

    return writer.write()
