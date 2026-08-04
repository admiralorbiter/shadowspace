"""Gauge Invariants Taxonomy and Similarity Conjugation Flatness Invariants.

Distinguishes:
1. Global Similarity Invariants (GL(2) Conjugation Invariant):
   tr(H), det(H), spec(H), rank_H_minus_I, dim ker(H - I), is_identity_flat (H == I).
2. Frame-Dependent Diagnostics (Require metric-preserving transformations):
   Polar rotation angle theta_polar, Frobenius norm ||log H||_F, anisotropy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class SimilarityInvariants:
    """Exact similarity invariants under GL(2) matrix conjugation H -> A H A^{-1}."""

    trace: float
    determinant: float
    eigenvalues: Tuple[complex, ...]
    rank_H_minus_I: int    # rank(H - I)
    ker_dimension: int     # dim ker(H - I)
    is_identity_flat: bool # H == I

    @classmethod
    def compute(cls, H: NDArray[np.float64], rtol: float = 1e-4) -> SimilarityInvariants:
        tr = float(np.trace(H))
        det = float(np.linalg.det(H))
        eigvals = tuple(np.linalg.eigvals(H))

        H_minus_I = H - np.eye(2)
        rank_h_minus_i = int(np.linalg.matrix_rank(H_minus_I, tol=rtol))
        ker_dim = 2 - rank_h_minus_i
        is_flat = bool(np.allclose(H, np.eye(2), atol=rtol))

        return cls(
            trace=tr,
            determinant=det,
            eigenvalues=eigvals,
            rank_H_minus_I=rank_h_minus_i,
            ker_dimension=ker_dim,
            is_identity_flat=is_flat,
        )


def verify_calibration_holonomy_invariance(
    H_original: NDArray[np.float64],
    jacobian_Df: NDArray[np.float64],
    rtol: float = 1e-4,
) -> bool:
    """Verifies Theorem 1A: H^f = Df * H * Df^(-1) preserves trace, det, spec, and rank(H-I)."""
    Df_inv = np.linalg.inv(jacobian_Df)
    H_recalibrated = np.dot(jacobian_Df, np.dot(H_original, Df_inv))

    inv_orig = SimilarityInvariants.compute(H_original, rtol=rtol)
    inv_recal = SimilarityInvariants.compute(H_recalibrated, rtol=rtol)

    if not np.isclose(inv_orig.trace, inv_recal.trace, rtol=rtol):
        return False

    if not np.isclose(inv_orig.determinant, inv_recal.determinant, rtol=rtol):
        return False

    if inv_orig.rank_H_minus_I != inv_recal.rank_H_minus_I:
        return False

    if inv_orig.is_identity_flat != inv_recal.is_identity_flat:
        return False

    eig_orig = sorted(inv_orig.eigenvalues, key=lambda z: (z.real, z.imag))
    eig_recal = sorted(inv_recal.eigenvalues, key=lambda z: (z.real, z.imag))

    for e1, e2 in zip(eig_orig, eig_recal):
        if not np.isclose(e1, e2, rtol=rtol):
            return False

    return True
