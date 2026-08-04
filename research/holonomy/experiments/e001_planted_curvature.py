"""Experiment E001: Planted Curvature Recovery.

Validates that when a planted non-zero rotation theta is injected into the CurvedWorld,
the holonomy estimator recovers the planted rotation angle theta_gamma within tolerance.
"""

from __future__ import annotations

import numpy as np

from research.holonomy.geometry.connection import ParallelTransportMap
from research.holonomy.geometry.holonomy import evaluate_holonomy
from research.holonomy.geometry.parallel_transport import PathTransport


def run_e001_planted_curvature_experiment(planted_angle: float = np.pi / 6) -> bool:
    """Runs E001 experiment verifying recovery of planted holonomy rotation angle."""
    c, s = np.cos(planted_angle), np.sin(planted_angle)
    R_planted = np.array([[c, -s], [s, c]], dtype=np.float64)

    # Construct loop path transport where composite matrix equals R_planted
    T_map = ParallelTransportMap(
        generator_name="planted_rot",
        source_id="x0",
        target_id="x0",
        matrix=R_planted,
        bias=np.zeros(2),
    )

    path_transport = PathTransport([T_map])
    res = evaluate_holonomy("PlantedRotationLoop", path_transport)

    # Check that estimated rotation angle matches planted_angle
    angle_recovered = np.isclose(res.rotation_angle, abs(planted_angle), atol=1e-5)
    return angle_recovered


if __name__ == "__main__":
    success = run_e001_planted_curvature_experiment()
    print(f"E001 Planted Curvature Experiment Passed: {success}")
