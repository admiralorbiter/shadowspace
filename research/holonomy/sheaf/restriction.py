"""Data-Dependent Evaluation Restriction Matrices for Cellular Sheaves.

Restricts local calibrators f_U(z) = A_U z + b_U to item evaluations on overlap items x_1, ..., x_m.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
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
        """Flattens (A_11, A_12, A_21, A_22, b_1, b_2) into 6D vector."""
        return np.concatenate([self.A_matrix.ravel(), self.b_vector])

    @classmethod
    def from_parameter_vector(cls, patch_id: str, vec: NDArray[np.float64]) -> LocalCalibrator:
        A = vec[:4].reshape((2, 2))
        b = vec[4:6]
        return cls(patch_id=patch_id, A_matrix=A, b_vector=b)


def build_evaluation_restriction_matrix(item_ilr_coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """Constructs (2m, 6) evaluation design matrix R_{U -> UV} mapping 6D calibrator params to 2m outputs.

    For each item j with ILR coords z_j = (z1, z2):
    [ z1  z2   0   0   1   0 ]
    [  0   0  z1  z2   0   1 ]
    """
    coords = np.atleast_2d(item_ilr_coords)
    m = coords.shape[0]
    R = np.zeros((2 * m, 6), dtype=np.float64)

    for j, (z1, z2) in enumerate(coords):
        R[2 * j, 0] = z1
        R[2 * j, 1] = z2
        R[2 * j, 4] = 1.0

        R[2 * j + 1, 2] = z1
        R[2 * j + 1, 3] = z2
        R[2 * j + 1, 5] = 1.0

    return R
