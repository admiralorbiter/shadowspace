"""Sheaf Laplacian Operator L_F = delta^* delta.

Computes the Sheaf Laplacian matrix, its spectrum, kernel dimension (H^0),
and cohomological obstruction metrics (H^1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

from research.holonomy.sheaf.coboundary import CoboundaryOperator


@dataclass
class SheafLaplacianSpectrum:
    """Spectral decomposition of Sheaf Laplacian L_F."""

    matrix: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    zero_eigenvalues_count: int  # dim ker(L_F) = dim H^0
    fiedler_value: float          # Smallest non-zero eigenvalue


class SheafLaplacian:
    """Sheaf Laplacian operator L_F = delta^T delta."""

    def __init__(self, coboundary: CoboundaryOperator, param_dim: int = 6) -> None:
        self.coboundary = coboundary
        self.param_dim = param_dim
        self.delta_mat = coboundary.build_matrix(param_dim=param_dim)
        self.L_mat = np.dot(self.delta_mat.T, self.delta_mat)

    def compute_spectrum(self, tol: float = 1e-7) -> SheafLaplacianSpectrum:
        """Computes eigenvalues of L_F."""
        eigvals = np.linalg.eigvalsh(self.L_mat)
        eigvals = np.sort(np.maximum(0.0, eigvals))

        zero_count = int(np.sum(eigvals < tol))
        non_zeros = eigvals[eigvals >= tol]
        fiedler = float(non_zeros[0]) if len(non_zeros) > 0 else 0.0

        return SheafLaplacianSpectrum(
            matrix=self.L_mat,
            eigenvalues=eigvals,
            zero_eigenvalues_count=zero_count,
            fiedler_value=fiedler,
        )
