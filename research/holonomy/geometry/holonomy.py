"""Loop Holonomy and Polar Decomposition.

Evaluates loop holonomy matrix H_gamma and computes polar decomposition H_gamma = R * U.
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
    matrix: NDArray[np.float64]  # (2, 2) Holonomy matrix H_gamma
    orthogonal_R: NDArray[np.float64]  # (2, 2) Rotation / Reflection component
    stretch_U: NDArray[np.float64]  # (2, 2) Shear / Scaling component
    rotation_angle: float  # Polar rotation angle theta_gamma in radians
    curvature_magnitude: float  # ||log H_gamma||_F
    volume_distortion: float  # log |det H_gamma|
    anisotropy: float  # log(sigma_min / sigma_max)


def evaluate_holonomy(loop_name: str, path_transport: PathTransport) -> HolonomyResult:
    """Evaluates holonomy matrix H_gamma and extracts polar invariants."""
    H = path_transport.compute_composite_matrix()

    # Polar decomposition H = R * U
    R, U = polar(H)

    # 1. Rotation angle theta_gamma = arccos(tr(R) / 2) clamped to [-1, 1]
    cos_theta = np.clip(0.5 * np.trace(R), -1.0, 1.0)
    rotation_angle = float(np.arccos(cos_theta))

    # 2. Curvature magnitude: Frobenius norm of matrix log
    try:
        from scipy.linalg import logm
        log_H = logm(H)
        curvature_magnitude = float(np.linalg.norm(log_H, "fro"))
    except Exception:
        curvature_magnitude = float(np.linalg.norm(H - np.eye(2), "fro"))

    # 3. Volume distortion: log |det H|
    det_H = float(np.linalg.det(H))
    volume_distortion = float(np.log(np.abs(det_H) + 1e-15))

    # 4. Anisotropy: log(sigma_min / sigma_max)
    s = np.linalg.svd(H, compute_uv=False)
    sigma_max, sigma_min = s[0], s[-1]
    anisotropy = float(np.log((sigma_min + 1e-15) / (sigma_max + 1e-15)))

    return HolonomyResult(
        loop_name=loop_name,
        matrix=H,
        orthogonal_R=R,
        stretch_U=U,
        rotation_angle=rotation_angle,
        curvature_magnitude=curvature_magnitude,
        volume_distortion=volume_distortion,
        anisotropy=anisotropy,
    )
