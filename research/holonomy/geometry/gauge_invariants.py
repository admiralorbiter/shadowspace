"""Gauge Invariants and Theorem 1 Recalibration Invariance Utilities.

Verifies that tr(H), det(H), and spectrum spec(H) are strictly invariant under matrix conjugation
H_f = Df H Df^{-1} induced by global recalibration maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class GaugeInvariants:
    """Coordinate-free gauge invariants of a holonomy matrix H_gamma."""

    trace: float
    determinant: float
    eigenvalues: Tuple[complex, ...]

    @classmethod
    def compute(cls, H: NDArray[np.float64]) -> GaugeInvariants:
        tr = float(np.trace(H))
        det = float(np.linalg.det(H))
        eigvals = tuple(np.linalg.eigvals(H))
        return cls(trace=tr, determinant=det, eigenvalues=eigvals)


def verify_calibration_holonomy_invariance(
    H_original: NDArray[np.float64],
    jacobian_Df: NDArray[np.float64],
    rtol: float = 1e-5,
) -> bool:
    """Verifies Theorem 1: H^f = Df * H * Df^(-1) preserves trace, det, and spectrum.

    Args:
        H_original: (2, 2) original holonomy matrix.
        jacobian_Df: (2, 2) invertible recalibration Jacobian matrix Df.
        rtol: Relative tolerance for floating point comparisons.

    Returns:
        True if Theorem 1 invariants hold within rtol.
    """
    Df_inv = np.linalg.inv(jacobian_Df)
    H_recalibrated = np.dot(jacobian_Df, np.dot(H_original, Df_inv))

    inv_orig = GaugeInvariants.compute(H_original)
    inv_recal = GaugeInvariants.compute(H_recalibrated)

    # 1. Trace invariance
    if not np.isclose(inv_orig.trace, inv_recal.trace, rtol=rtol):
        return False

    # 2. Determinant invariance
    if not np.isclose(inv_orig.determinant, inv_recal.determinant, rtol=rtol):
        return False

    # 3. Spectrum invariance (sorted eigenvalues)
    eig_orig = sorted(inv_orig.eigenvalues, key=lambda z: (z.real, z.imag))
    eig_recal = sorted(inv_recal.eigenvalues, key=lambda z: (z.real, z.imag))

    for e1, e2 in zip(eig_orig, eig_recal):
        if not np.isclose(e1, e2, rtol=rtol):
            return False

    return True
