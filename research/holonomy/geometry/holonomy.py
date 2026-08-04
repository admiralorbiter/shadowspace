"""Loop Holonomy and Affine Polar Decomposition.

Evaluates homogeneous holonomy H_gamma in Aff(2), computes linear polar decomposition H = R * U,
and distinguishes linear flatness (A_gamma == I_2) from affine flatness (H_gamma == I_3).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import polar

from research.holonomy.geometry.parallel_transport import PathTransport


@dataclass
class HolonomyResult:
    """Holonomy evaluation result over a closed loop gamma."""

    loop_name: str
    matrix: NDArray[np.float64]  # (2, 2) Linear Holonomy matrix A_gamma
    homogeneous_matrix: NDArray[np.float64]  # (3, 3) Homogeneous matrix H_gamma
    translation_defect: NDArray[np.float64]  # (2,) Translation defect b_gamma
    linear_is_flat: bool  # A_gamma == I_2
    affine_is_flat: bool  # H_gamma == I_3 (A_gamma == I_2 and b_gamma == 0)
    orthogonal_R: NDArray[np.float64]  # (2, 2) Rotation / Reflection component
    stretch_U: NDArray[np.float64]  # (2, 2) Shear / Scaling component
    rotation_angle: float  # Polar rotation angle theta_gamma in radians
    curvature_magnitude: float  # ||log A_gamma||_F
    volume_distortion: float  # log |det A_gamma|
    anisotropy: float  # log(sigma_min / sigma_max)


def evaluate_holonomy(loop_name: str, path_transport: PathTransport, tol: float = 1e-4) -> HolonomyResult:
    """Evaluates 3x3 homogeneous holonomy matrix and extracts affine/linear invariants."""
    H_hom = path_transport.compute_homogeneous_matrix()
    A = H_hom[:2, :2]
    b = H_hom[:2, 2]

    linear_flat = bool(np.allclose(A, np.eye(2), atol=tol))
    affine_flat = bool(np.allclose(H_hom, np.eye(3), atol=tol))

    # Polar decomposition A = R * U
    R, U = polar(A)

    cos_theta = np.clip(0.5 * np.trace(R), -1.0, 1.0)
    rotation_angle = float(np.arccos(cos_theta))

    try:
        from scipy.linalg import logm
        log_A = logm(A)
        curvature_magnitude = float(np.linalg.norm(log_A, "fro"))
    except Exception:
        curvature_magnitude = float(np.linalg.norm(A - np.eye(2), "fro"))

    det_A = float(np.linalg.det(A))
    volume_distortion = float(np.log(np.abs(det_A) + 1e-15))

    s = np.linalg.svd(A, compute_uv=False)
    sigma_max, sigma_min = s[0], s[-1]
    anisotropy = float(np.log((sigma_min + 1e-15) / (sigma_max + 1e-15)))

    return HolonomyResult(
        loop_name=loop_name,
        matrix=A,
        homogeneous_matrix=H_hom,
        translation_defect=b,
        linear_is_flat=linear_flat,
        affine_is_flat=affine_flat,
        orthogonal_R=R,
        stretch_U=U,
        rotation_angle=rotation_angle,
        curvature_magnitude=curvature_magnitude,
        volume_distortion=volume_distortion,
        anisotropy=anisotropy,
    )
