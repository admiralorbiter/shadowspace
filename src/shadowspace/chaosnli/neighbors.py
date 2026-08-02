"""k-Nearest Neighbor extraction and graph storage module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def extract_knn_graph(
    dist_matrix: np.ndarray,
    object_ids: list[str],
    k: int = 10,
    space_id: str = "human_opinion",
    metric_id: str = "hellinger",
) -> tuple[np.ndarray, pl.DataFrame]:
    """Extract top-k nearest neighbors per node from an NxN distance matrix.

    Args:
        dist_matrix: (N, N) distance array.
        object_ids: List of N string object IDs.
        k: Number of nearest neighbors per node.
        space_id: Representation identifier.
        metric_id: Metric identifier.

    Returns:
        knn_indices: (N, k) integer array of neighbor indices.
        neighbor_df: Polars DataFrame of neighbor edge records.
    """
    n = len(dist_matrix)
    if k >= n:
        raise ValueError(f"k ({k}) must be smaller than number of objects ({n})")

    knn_indices = np.zeros((n, k), dtype=np.int32)
    records: list[dict[str, Any]] = []

    for i in range(n):
        row = dist_matrix[i].copy()
        row[i] = np.inf  # Exclude self

        # Top-k smallest distances
        # Partition first for speed, then argsort top k
        top_k_idx = np.argpartition(row, k)[:k]
        top_k_idx = top_k_idx[np.argsort(row[top_k_idx])]

        knn_indices[i] = top_k_idx

        src_id = object_ids[i]
        for rank_idx, neighbor_idx in enumerate(top_k_idx, start=1):
            records.append({
                "source_id": src_id,
                "neighbor_id": object_ids[neighbor_idx],
                "rank": rank_idx,
                "distance": float(dist_matrix[i, neighbor_idx]),
                "space_id": space_id,
                "metric_id": metric_id,
                "k": k,
            })

    neighbor_df = pl.DataFrame(records)
    return knn_indices, neighbor_df


def save_knn_graph(
    df: pl.DataFrame,
    output_dir: Path = Path("data/chaosnli/processed"),
    space_id: str = "human_opinion",
    metric_id: str = "hellinger",
    k: int = 10,
) -> Path:
    """Save k-NN neighbor DataFrame to Parquet file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"neighbors_{space_id}_{metric_id}_k{k:03d}.parquet"
    filepath = output_dir / filename
    df.write_parquet(filepath)
    return filepath
