"""Homogeneous Affine Parallel Transport Connection (3x3 Representation).

Estimates and composes affine parallel transport operators in 3x3 homogeneous coordinates:
T = [[A, b], [0, 1]]
Provides both OLS and Total Least Squares (TLS) estimators to correct errors-in-variables attenuation bias.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class ParallelTransportMap:
    """Homogeneous parallel transport operator T_g in Aff(2) represented as 3x3 matrix."""

    generator_name: str
    source_id: str
    target_id: str
    matrix_2d: NDArray[np.float64]  # (2, 2) Linear matrix A
    bias_2d: NDArray[np.float64]    # (2,) Translation vector b

    @property
    def homogeneous_matrix(self) -> NDArray[np.float64]:
        """Returns 3x3 homogeneous matrix representation [[A, b], [0, 1]]."""
        H = np.eye(3, dtype=np.float64)
        H[:2, :2] = self.matrix_2d
        H[:2, 2] = self.bias_2d
        return H

    def transform(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply transport T(v) = A v + b."""
        return np.dot(self.matrix_2d, vector) + self.bias_2d


class ConnectionEstimator:
    """Estimates affine parallel transport maps T_{g,x} via OLS or Total Least Squares (TLS)."""

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
        """Estimates affine map T_g via OLS: z_target approx A z_source + b."""
        Z_src = np.atleast_2d(source_coords)
        Z_tgt = np.atleast_2d(target_coords)

        mean_src = Z_src.mean(axis=0)
        mean_tgt = Z_tgt.mean(axis=0)

        Z_src_c = Z_src - mean_src
        Z_tgt_c = Z_tgt - mean_tgt

        d = Z_src_c.shape[1]
        cov = np.dot(Z_src_c.T, Z_src_c) + self.ridge_alpha * np.eye(d)
        cross = np.dot(Z_src_c.T, Z_tgt_c)

        T_mat_T = np.linalg.solve(cov, cross)
        T_mat = T_mat_T.T

        bias = mean_tgt - np.dot(T_mat, mean_src)

        return ParallelTransportMap(
            generator_name=generator_name,
            source_id=source_id,
            target_id=target_id,
            matrix_2d=T_mat,
            bias_2d=bias,
        )

    def estimate_total_least_squares_transport(
        self,
        generator_name: str,
        source_id: str,
        target_id: str,
        source_coords: NDArray[np.float64],
        target_coords: NDArray[np.float64],
    ) -> ParallelTransportMap:
        """Estimates affine map T_g via Total Least Squares (TLS) to correct errors-in-variables attenuation bias."""
        Z_src = np.atleast_2d(source_coords)
        Z_tgt = np.atleast_2d(target_coords)

        mean_src = Z_src.mean(axis=0)
        mean_tgt = Z_tgt.mean(axis=0)

        Z_src_c = Z_src - mean_src
        Z_tgt_c = Z_tgt - mean_tgt

        d_x = Z_src_c.shape[1]
        d_y = Z_tgt_c.shape[1]

        # Stack augmented matrix [X_c | Y_c]
        Aug = np.column_stack([Z_src_c, Z_tgt_c])

        # Perform SVD of augmented matrix
        _, _, Vt = np.linalg.svd(Aug, full_matrices=True)
        V = Vt.T

        # Partition V into [[V11, V12], [V21, V22]]
        V12 = V[:d_x, d_x:]
        V22 = V[d_x:, d_x:]

        # TLS estimate A^T = -V12 V22^(-1) => A = (-V12 V22^(-1))^T
        T_mat_T = -np.dot(V12, np.linalg.inv(V22))
        T_mat = T_mat_T.T

        bias = mean_tgt - np.dot(T_mat, mean_src)

        return ParallelTransportMap(
            generator_name=generator_name,
            source_id=source_id,
            target_id=target_id,
            matrix_2d=T_mat,
            bias_2d=bias,
        )
