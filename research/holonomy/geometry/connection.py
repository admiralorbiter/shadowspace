"""Transport Connection Estimator.

Estimates local parallel transport linear/affine maps T_{g,x}: F_x -> F_{gx} from paired orbits.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class ParallelTransportMap:
    """Linear parallel transport operator T_{g,x} in GL(2) and optional translation vector b."""

    generator_name: str
    source_id: str
    target_id: str
    matrix: NDArray[np.float64]  # Shape (2, 2)
    bias: NDArray[np.float64]    # Shape (2,)

    def transform(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply transport T(v) = M v + b."""
        return np.dot(self.matrix, vector) + self.bias


class ConnectionEstimator:
    """Estimates parallel transport maps T_{g,x} via multivariate least squares regression or Procrustes."""

    def __init__(self, ridge_alpha: float = 1e-6) -> None:
        self.ridge_alpha = ridge_alpha

    def estimate_linear_transport(
        self,
        generator_name: str,
        source_id: str,
        target_id: str,
        source_coords: NDArray[np.float64],
        target_coords: NDArray[np.float64],
    ) -> ParallelTransportMap:
        """Estimate T_g in GL(2) such that z_target - mean_target approx T_g (z_source - mean_source).

        Args:
            generator_name: Name of generator g.
            source_id: Base vertex ID x.
            target_id: Transformed vertex ID gx.
            source_coords: Shape (M, 2) ILR coordinates near x.
            target_coords: Shape (M, 2) ILR coordinates near gx.

        Returns:
            ParallelTransportMap containing (2, 2) matrix T_g and translation vector.
        """
        Z_src = np.atleast_2d(source_coords)
        Z_tgt = np.atleast_2d(target_coords)

        mean_src = Z_src.mean(axis=0)
        mean_tgt = Z_tgt.mean(axis=0)

        Z_src_c = Z_src - mean_src
        Z_tgt_c = Z_tgt - mean_tgt

        # Ridge regression: T_g = (Z_src_c^T Z_src_c + alpha I)^(-1) Z_src_c^T Z_tgt_c
        d = Z_src_c.shape[1]
        cov = np.dot(Z_src_c.T, Z_src_c) + self.ridge_alpha * np.eye(d)
        cross = np.dot(Z_src_c.T, Z_tgt_c)

        # Matrix T_g operates on column vectors v: T_g_mat @ v
        # Since Z_tgt_c ~ Z_src_c @ T_g_mat^T, T_g_mat^T = cov^(-1) cross
        T_mat_T = np.linalg.solve(cov, cross)
        T_mat = T_mat_T.T

        bias = mean_tgt - np.dot(T_mat, mean_src)

        return ParallelTransportMap(
            generator_name=generator_name,
            source_id=source_id,
            target_id=target_id,
            matrix=T_mat,
            bias=bias,
        )
