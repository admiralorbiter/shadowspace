"""shadowspace.math.registry — MetricRegistry for managing metrics and compatibility validation.

Enforces valid representation/metric pairs and provides high-level APIs for
pairwise distance matrix computation and k-NN queries.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from shadowspace.math.metrics import (
    pairwise_aitchison,
    pairwise_euclidean,
    pairwise_fisher_rao,
    pairwise_hellinger,
    pairwise_jensen_shannon,
)
from shadowspace.models.schemas import MetricSpec

MetricFunc = Callable[[NDArray[np.float64], NDArray[np.float64] | None], NDArray[np.float64]]


class MetricRegistry:
    """Registry of distance metrics and representation compatibility rules."""

    def __init__(self) -> None:
        self._metrics: dict[str, tuple[MetricSpec, MetricFunc]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default Shadowspace metrics."""
        self.register_metric(
            spec=MetricSpec(
                id="euclidean",
                display_name="Euclidean Distance",
                representation_ids=["probability", "sqrt_probability", "clr_probability", "logit"],
                is_metric=True,
            ),
            func=pairwise_euclidean,
        )

        self.register_metric(
            spec=MetricSpec(
                id="hellinger",
                display_name="Hellinger Distance",
                representation_ids=["probability"],
                is_metric=True,
                units_or_scale="[0, 1]",
            ),
            func=pairwise_hellinger,
        )

        self.register_metric(
            spec=MetricSpec(
                id="fisher_rao",
                display_name="Fisher-Rao Distance",
                representation_ids=["probability"],
                is_metric=True,
                parameters={"scale": 2.0, "convention": "canonical_fisher_information"},
                units_or_scale="radians",
            ),
            func=pairwise_fisher_rao,
        )

        self.register_metric(
            spec=MetricSpec(
                id="aitchison",
                display_name="Aitchison Distance",
                representation_ids=["probability"],
                is_metric=True,
                parameters={"note": "CLR transform applied internally"},
            ),
            func=pairwise_aitchison,
        )

        self.register_metric(
            spec=MetricSpec(
                id="jensen_shannon",
                display_name="Jensen-Shannon Distance",
                representation_ids=["probability"],
                is_metric=True,
                units_or_scale="bits",
            ),
            func=pairwise_jensen_shannon,
        )

    def register_metric(self, spec: MetricSpec, func: MetricFunc) -> None:
        """Register a new metric specification and function."""
        self._metrics[spec.id] = (spec, func)

    def get_spec(self, metric_id: str) -> MetricSpec:
        """Get the spec for a metric ID."""
        if metric_id not in self._metrics:
            raise KeyError(f"Metric {metric_id!r} not found in registry.")
        return self._metrics[metric_id][0]

    def validate_compatibility(self, metric_id: str, representation_id: str) -> None:
        """Check if metric_id is compatible with representation_id.

        Raises:
            KeyError: If metric_id is unknown.
            ValueError: If metric_id is incompatible with representation_id.
        """
        spec = self.get_spec(metric_id)
        if representation_id not in spec.representation_ids:
            raise ValueError(
                f"Metric {metric_id!r} is incompatible with representation {representation_id!r}. "
                f"Compatible representations: {spec.representation_ids}"
            )

    def compute_pairwise_distances(
        self,
        matrix: NDArray[np.float64],
        metric_id: str,
        representation_id: str,
    ) -> NDArray[np.float64]:
        """Compute pairwise distance matrix for coordinates under metric_id.

        Args:
            matrix: Shape (N, K) coordinate matrix.
            metric_id: Metric ID to compute.
            representation_id: Representation ID of the matrix.

        Returns:
            Shape (N, N) distance matrix.
        """
        self.validate_compatibility(metric_id, representation_id)
        _, func = self._metrics[metric_id]
        return func(matrix, None)

    def find_k_nearest_neighbors(
        self,
        matrix: NDArray[np.float64],
        target_idx: int,
        k: int,
        metric_id: str,
        representation_id: str,
        object_ids: list[str] | None = None,
    ) -> list[tuple[str | int, float]]:
        """Find k nearest neighbors for target_idx (excluding self).

        Args:
            matrix: Shape (N, K) coordinate matrix.
            target_idx: Index of target point in matrix.
            k: Number of neighbors to return.
            metric_id: Metric ID.
            representation_id: Representation ID.
            object_ids: Optional string object IDs matching matrix rows.

        Returns:
            List of (id_or_index, distance) tuples sorted by distance.
        """
        if not (0 <= target_idx < len(matrix)):
            raise ValueError(
                f"target_idx {target_idx} out of range for matrix length {len(matrix)}."
            )

        distances = self.compute_pairwise_distances(matrix, metric_id, representation_id)[
            target_idx
        ]

        # Sort indices by distance
        sorted_indices = np.argsort(distances)

        # Exclude self (target_idx)
        neighbor_indices = [idx for idx in sorted_indices if idx != target_idx][:k]

        results: list[tuple[str | int, float]] = []
        for idx in neighbor_indices:
            label: str | int = object_ids[idx] if object_ids is not None else int(idx)
            results.append((label, float(distances[idx])))

        return results
