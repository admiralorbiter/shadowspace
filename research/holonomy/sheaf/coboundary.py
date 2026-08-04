"""Data-Dependent Coboundary Operator delta for Cellular Sheaf.

Constructs 0-cochain and 1-cochain evaluation coboundary matrices delta_0: C^0 -> C^1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray

from research.holonomy.sheaf.restriction import LocalCalibrator, build_evaluation_restriction_matrix


@dataclass(frozen=True)
class OverlapEdge:
    """Pair of overlapping patches (U, V) with overlap item ILR coordinates."""

    patch_U: str
    patch_V: str
    overlap_item_coords: NDArray[np.float64]  # (m, 2) ILR coordinates


class CoboundaryOperator:
    """Data-dependent coboundary operator delta_0 mapping 6D local parameter vectors to 2m evaluation residuals."""

    def __init__(self, patches: List[str], overlaps: List[OverlapEdge]) -> None:
        self.patches = sorted(patches)
        self.overlaps = overlaps
        self.patch_to_idx = {p: i for i, p in enumerate(self.patches)}
        self.num_patches = len(self.patches)
        self.num_overlaps = len(self.overlaps)

    def build_matrix(self, param_dim: int = 6) -> NDArray[np.float64]:
        """Builds data-dependent coboundary matrix delta_0 using item evaluation restriction blocks."""
        if not self.overlaps:
            return np.zeros((0, self.num_patches * param_dim), dtype=np.float64)

        total_eval_rows = sum(2 * ov.overlap_item_coords.shape[0] for ov in self.overlaps)
        N = self.num_patches * param_dim
        delta_mat = np.zeros((total_eval_rows, N), dtype=np.float64)

        curr_row = 0
        for ov in self.overlaps:
            u_idx = self.patch_to_idx[ov.patch_U]
            v_idx = self.patch_to_idx[ov.patch_V]

            R_uv = build_evaluation_restriction_matrix(ov.overlap_item_coords)
            num_rows = R_uv.shape[0]

            u_col_start = u_idx * param_dim
            u_col_end = u_col_start + param_dim

            v_col_start = v_idx * param_dim
            v_col_end = v_col_start + param_dim

            delta_mat[curr_row : curr_row + num_rows, u_col_start:u_col_end] = R_uv
            delta_mat[curr_row : curr_row + num_rows, v_col_start:v_col_end] = -R_uv

            curr_row += num_rows

        return delta_mat
