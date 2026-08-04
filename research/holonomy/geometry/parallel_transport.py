"""Parallel Transport along Path.

Composes sequential transport operators along a path of edges.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from numpy.typing import NDArray

from research.holonomy.geometry.connection import ParallelTransportMap


class PathTransport:
    """Sequential composition of transport maps along a word or edge path."""

    def __init__(self, transport_maps: Sequence[ParallelTransportMap]) -> None:
        self.transport_maps = list(transport_maps)

    def compute_composite_matrix(self) -> NDArray[np.float64]:
        """Composes linear transport matrices sequentially: T = T_k ... T_2 T_1."""
        composite = np.eye(2, dtype=np.float64)
        for t_map in self.transport_maps:
            composite = np.dot(t_map.matrix, composite)
        return composite

    def transport_vector(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        curr = vector.copy()
        for t_map in self.transport_maps:
            curr = t_map.transform(curr)
        return curr
