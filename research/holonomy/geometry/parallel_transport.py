"""Homogeneous Path Parallel Transport.

Composes 3x3 homogeneous transport matrices sequentially along a word path.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from numpy.typing import NDArray

from research.holonomy.geometry.connection import ParallelTransportMap


class PathTransport:
    """Sequential composition of 3x3 homogeneous transport maps along an edge path."""

    def __init__(self, transport_maps: Sequence[ParallelTransportMap]) -> None:
        self.transport_maps = list(transport_maps)

    def compute_homogeneous_matrix(self) -> NDArray[np.float64]:
        """Composes 3x3 homogeneous matrices: T = T_k ... T_1."""
        composite = np.eye(3, dtype=np.float64)
        for t_map in self.transport_maps:
            composite = np.dot(t_map.homogeneous_matrix, composite)
        return composite

    def compute_composite_matrix(self) -> NDArray[np.float64]:
        """Returns 2x2 linear matrix component A_gamma."""
        return self.compute_homogeneous_matrix()[:2, :2]

    def compute_translation_defect(self) -> NDArray[np.float64]:
        """Returns 2D translational defect vector b_gamma."""
        return self.compute_homogeneous_matrix()[:2, 2]

    def transport_vector(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        v_hom = np.array([vector[0], vector[1], 1.0], dtype=np.float64)
        res_hom = np.dot(self.compute_homogeneous_matrix(), v_hom)
        return res_hom[:2]
