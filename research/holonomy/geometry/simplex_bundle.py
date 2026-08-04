r"""Simplex Fiber Bundle and ILR Coordinates.

Constructs the bundle pi: E -> X with fibers F_x = T_{p(x)} Delta^2 \cong R^2 using
Isometric Log-Ratio (ILR) coordinates with Helmert orthonormal basis.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from shadowspace.math.clr import clr_transform


def get_helmert_basis_3d() -> NDArray[np.float64]:
    """Returns (3, 2) orthonormal basis matrix V for zero-sum subspace 1^perp in R^3."""
    v1 = np.array([1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0], dtype=np.float64)
    v2 = np.array([1.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0)], dtype=np.float64)
    return np.column_stack([v1, v2])


HELMERT_V3 = get_helmert_basis_3d()


def ilr_transform(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """Applies Isometric Log-Ratio (ILR) transform mapping Delta^2 to R^2.

    Args:
        probabilities: Shape (N, 3) or (3,) non-negative probability vectors.

    Returns:
        Shape (N, 2) or (2,) ILR coordinate matrix.
    """
    is_1d = probabilities.ndim == 1
    p = np.atleast_2d(probabilities)

    clr = clr_transform(p)  # Shape (N, 3)
    ilr = np.dot(clr, HELMERT_V3)  # Shape (N, 2)

    return ilr[0] if is_1d else ilr


def ilr_inverse(ilr_coords: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse ILR transform mapping R^2 back to probability simplex Delta^2.

    Args:
        ilr_coords: Shape (N, 2) or (2,) ILR coordinates.

    Returns:
        Shape (N, 3) or (3,) probability vectors in Delta^2.
    """
    is_1d = ilr_coords.ndim == 1
    z = np.atleast_2d(ilr_coords)

    clr = np.dot(z, HELMERT_V3.T)  # Shape (N, 3)
    exp_clr = np.exp(clr)
    p = exp_clr / exp_clr.sum(axis=1, keepdims=True)

    return p[0] if is_1d else p


@dataclass
class SimplexFiber:
    r"""Fiber F_x = T_{p(x)} Delta^2 \cong R^2 at semantic state x."""

    state_id: str
    probability: NDArray[np.float64]
    ilr_coords: NDArray[np.float64]

    @classmethod
    def from_probability(cls, state_id: str, p: NDArray[np.float64]) -> SimplexFiber:
        ilr = ilr_transform(p)
        return cls(state_id=state_id, probability=p, ilr_coords=ilr)
