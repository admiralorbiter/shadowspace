"""Coboundary Operator delta for Cellular Sheaf.

Constructs 0-cochains, 1-cochains, and coboundary linear operator matrix delta: C^0 -> C^1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

from research.holonomy.sheaf.restriction import LocalCalibrator


@dataclass(frozen=True)
class OverlapEdge:
    """Pair of overlapping patches (U, V)."""

    patch_U: str
    patch_V: str
    overlap_item_ids: Tuple[str, ...]


class CoboundaryOperator:
    """Coboundary operator delta mapping 0-cochain parameter vector to 1-cochain residuals."""

    def __init__(self, patches: List[str], overlaps: List[OverlapEdge]) -> None:
        self.patches = sorted(patches)
        self.overlaps = overlaps
        self.patch_to_idx = {p: i for i, p in enumerate(self.patches)}
        self.num_patches = len(self.patches)
        self.num_overlaps = len(self.overlaps)

    def compute_residuals(
        self, local_calibrators: List[LocalCalibrator]
    ) -> NDArray[np.float64]:
        """Calculates 1-cochain residuals (delta theta)_{U, V} = theta_U - theta_V."""
        calib_map = {c.patch_id: c for c in local_calibrators}
        residuals = []

        for ov in self.overlaps:
            cU = calib_map[ov.patch_U]
            cV = calib_map[ov.patch_V]
            res = cU.to_parameter_vector() - cV.to_parameter_vector()
            residuals.append(res)

        return np.concatenate(residuals) if residuals else np.zeros(0, dtype=np.float64)

    def build_matrix(self, param_dim: int = 6) -> NDArray[np.float64]:
        """Builds explicit sparse/dense boundary matrix delta of shape (num_overlaps * param_dim, num_patches * param_dim)."""
        M = self.num_overlaps * param_dim
        N = self.num_patches * param_dim
        delta_mat = np.zeros((M, N), dtype=np.float64)

        for ov_idx, ov in enumerate(self.overlaps):
            u_idx = self.patch_to_idx[ov.patch_U]
            v_idx = self.patch_to_idx[ov.patch_V]

            row_start = ov_idx * param_dim
            row_end = row_start + param_dim

            u_col_start = u_idx * param_dim
            u_col_end = u_col_start + param_dim

            v_col_start = v_idx * param_dim
            v_col_end = v_col_start + param_dim

            delta_mat[row_start:row_end, u_col_start:u_col_end] = np.eye(param_dim)
            delta_mat[row_start:row_end, v_col_start:v_col_end] = -np.eye(param_dim)

        return delta_mat
