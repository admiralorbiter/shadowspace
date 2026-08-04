"""Cellular Sheaf Stalks and Restrictions.

Defines local calibrator parameter spaces Aff(2) attached to open cover patches U,
and restriction maps to overlaps U cap V.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class LocalCalibrator:
    """Affine calibrator f_U(z) = A_U z + b_U on local patch U."""

    patch_id: str
    A_matrix: NDArray[np.float64]  # (2, 2)
    b_vector: NDArray[np.float64]  # (2,)

    def __call__(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.dot(self.A_matrix, z) + self.b_vector

    def to_parameter_vector(self) -> NDArray[np.float64]:
        """Flattens (A, b) into 6-dimensional parameter vector."""
        return np.concatenate([self.A_matrix.ravel(), self.b_vector])

    @classmethod
    def from_parameter_vector(cls, patch_id: str, vec: NDArray[np.float64]) -> LocalCalibrator:
        A = vec[:4].reshape((2, 2))
        b = vec[4:6]
        return cls(patch_id=patch_id, A_matrix=A, b_vector=b)
