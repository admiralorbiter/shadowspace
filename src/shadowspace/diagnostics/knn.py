"""shadowspace.diagnostics.knn — k-NN graph computation, deterministic tie breaking, and neighbor classification.

Sprint 4: Local integrity diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NeighborClassification:
    """Classification of neighborhood relationships for a target object."""

    target_id: str
    k: int
    preserved: list[str] = field(default_factory=list)  # In both source and projected k-NN
    torn: list[str] = field(default_factory=list)       # In source k-NN, absent in projected
    false_neighbors: list[str] = field(default_factory=list) # In projected k-NN, absent in source

    @property
    def precision(self) -> float:
        """Local precision: preserved / projected_count."""
        proj_count = len(self.preserved) + len(self.false_neighbors)
        return len(self.preserved) / proj_count if proj_count > 0 else 1.0

    @property
    def recall(self) -> float:
        """Local recall: preserved / source_count."""
        source_count = len(self.preserved) + len(self.torn)
        return len(self.preserved) / source_count if source_count > 0 else 1.0

    @property
    def jaccard_overlap(self) -> float:
        """Jaccard overlap: preserved / union_count."""
        union_count = len(self.preserved) + len(self.torn) + len(self.false_neighbors)
        return len(self.preserved) / union_count if union_count > 0 else 1.0


def compute_knn(
    distance_matrix: NDArray[np.float64],
    k: int,
    object_ids: list[str],
    target_id: str | None = None,
) -> dict[str, list[str]]:
    """Compute top-k nearest neighbors for each object (or target_id only) excluding self.

    Deterministic tie-breaking: when distances are equal, ties are broken
    by object ID string sort order.

    Args:
        distance_matrix: Shape (N, N) symmetric dissimilarity matrix.
        k: Number of nearest neighbors (1 <= k < N).
        object_ids: List of N string object IDs.
        target_id: Optional target object ID string to restrict query to.

    Returns:
        Dict mapping object_id -> list of k nearest neighbor object IDs.

    Raises:
        ValueError: If distance matrix is not square, k is invalid, or target_id unknown.
    """
    n_objects = len(object_ids)
    if distance_matrix.shape != (n_objects, n_objects):
        raise ValueError(
            f"Distance matrix shape {distance_matrix.shape} does not match object count {n_objects}."
        )
    if not (1 <= k < n_objects):
        raise ValueError(f"Neighborhood size k={k} must be in 1 <= k < N ({n_objects}).")

    if target_id is not None and target_id not in object_ids:
        raise ValueError(f"Target object ID {target_id!r} not in object_ids.")

    indices = [object_ids.index(target_id)] if target_id is not None else list(range(n_objects))
    knn_graph: dict[str, list[str]] = {}

    for i in indices:
        src_id = object_ids[i]
        dists = distance_matrix[i].copy()
        dists[i] = np.inf

        candidates = list(zip(dists, object_ids, strict=True))
        candidates.sort(key=lambda pair: (pair[0], pair[1]))
        knn_graph[src_id] = [oid for _, oid in candidates[:k]]

    return knn_graph


def classify_point_neighbors(
    source_knn: list[str],
    proj_knn: list[str],
    target_id: str,
    k: int,
) -> NeighborClassification:
    """Classify neighbors into preserved, torn, and false for a target point.

    Args:
        source_knn: List of k nearest neighbor IDs in high-D source space.
        proj_knn: List of k nearest neighbor IDs in 2D projected space.
        target_id: Target object ID being analyzed.
        k: Neighborhood size.

    Returns:
        NeighborClassification containing preserved, torn, and false neighbor lists,
        plus local precision, recall, and Jaccard overlap metrics.
    """
    source_set = set(source_knn)
    proj_set = set(proj_knn)

    # Preserved: in both source and projected k-NN
    preserved = [oid for oid in source_knn if oid in proj_set]

    # Torn: in source k-NN, absent in projected k-NN
    torn = [oid for oid in source_knn if oid not in proj_set]

    # False neighbors: in projected k-NN, absent in source k-NN
    false_neighbors = [oid for oid in proj_knn if oid not in source_set]

    return NeighborClassification(
        target_id=target_id,
        k=k,
        preserved=preserved,
        torn=torn,
        false_neighbors=false_neighbors,
    )


def compute_point_diagnostics(
    source_dist_matrix: NDArray[np.float64],
    proj_dist_matrix: NDArray[np.float64],
    k: int,
    object_ids: list[str],
    target_id: str,
) -> NeighborClassification:
    """Compute local neighbor classification for a specific target object.

    Args:
        source_dist_matrix: Shape (N, N) high-D distance matrix.
        proj_dist_matrix: Shape (N, N) 2D projected distance matrix.
        k: Neighborhood size k.
        object_ids: List of object IDs.
        target_id: Target object ID to diagnose.

    Returns:
        NeighborClassification for target_id.
    """
    if target_id not in object_ids:
        raise ValueError(f"Target object ID {target_id!r} not in object_ids.")

    src_knn = compute_knn(source_dist_matrix, k, object_ids, target_id=target_id)
    proj_knn = compute_knn(proj_dist_matrix, k, object_ids, target_id=target_id)

    return classify_point_neighbors(
        source_knn=src_knn[target_id],
        proj_knn=proj_knn[target_id],
        target_id=target_id,
        k=k,
    )
