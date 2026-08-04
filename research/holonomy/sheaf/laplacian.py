"""Cellular Sheaf Laplacian Operator L_F = delta_0^T delta_0 and Cohomology (H^0, H^1).

Computes data-dependent Sheaf Laplacian matrix, kernel dimension (H^0),
and first cohomology group H^1 = ker(delta_1) / im(delta_0).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from research.holonomy.sheaf.coboundary import CoboundaryOperator


@dataclass
class SheafLaplacianSpectrum:
    """Spectral decomposition and cohomology of Sheaf Laplacian L_F."""

    matrix: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    dim_H0: int  # dim ker(delta_0) = dim ker(L_F)
    dim_H1: int  # dim ker(delta_1) / im(delta_0) (cohomological obstruction)
    fiedler_value: float


class SheafLaplacian:
    """Sheaf Laplacian operator L_F = delta_0^T delta_0."""

    def __init__(self, coboundary: CoboundaryOperator, param_dim: int = 6) -> None:
        self.coboundary = coboundary
        self.param_dim = param_dim
        self.delta0_mat = coboundary.build_matrix(param_dim=param_dim)
        self.L_mat = np.dot(self.delta0_mat.T, self.delta0_mat)

    def compute_spectrum(self, tol: float = 1e-4) -> SheafLaplacianSpectrum:
        """Computes eigenvalues of L_F and cohomology dimensions H^0, H^1."""
        eigvals = np.linalg.eigvalsh(self.L_mat)
        eigvals = np.sort(np.maximum(0.0, eigvals))

        dim_H0 = int(np.sum(eigvals < tol))
        non_zeros = eigvals[eigvals >= tol]
        fiedler = float(non_zeros[0]) if len(non_zeros) > 0 else 0.0

        # Rank of delta_0 image: im(delta_0)
        rank_im_delta0 = int(np.linalg.matrix_rank(self.delta0_mat, tol=tol))
        # Total 1-cochain space dimension: C^1
        num_1_cochains = self.delta0_mat.shape[0]

        # For 1D loop network, H^1 dimension = C^1 - rank(im delta_0)
        dim_H1 = max(0, num_1_cochains - rank_im_delta0)

        return SheafLaplacianSpectrum(
            matrix=self.L_mat,
            eigenvalues=eigvals,
            dim_H0=dim_H0,
            dim_H1=dim_H1,
            fiedler_value=fiedler,
        )
