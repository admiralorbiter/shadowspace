"""Experiment E003: Calibration-Holonomy Invariance (Theorem 1 Verification).

Empirically confirms that global invertible affine recalibrations f(z) = A z + b
preserve trace, determinant, and spectrum of loop holonomy H_gamma across curved worlds.
"""

from __future__ import annotations

import numpy as np

from research.holonomy.geometry.gauge_invariants import verify_calibration_holonomy_invariance


def run_e003_calibration_invariance_experiment() -> bool:
    """Runs E003 experiment testing Theorem 1 across random holonomy matrices and Jacobians."""
    np.random.seed(123)

    for _ in range(20):
        # 1. Random non-trivial holonomy matrix H in GL(2)
        angle = np.random.uniform(0.1, np.pi / 2)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        scale = np.diag([np.random.uniform(0.5, 2.0), np.random.uniform(0.5, 2.0)])
        H_orig = np.dot(R, scale)

        # 2. Random invertible calibration Jacobian Df in GL(2)
        A = np.random.normal(0, 1, (2, 2))
        while abs(np.linalg.det(A)) < 0.1:
            A = np.random.normal(0, 1, (2, 2))

        # 3. Verify Theorem 1 invariants
        if not verify_calibration_holonomy_invariance(H_orig, A):
            return False

    return True


if __name__ == "__main__":
    success = run_e003_calibration_invariance_experiment()
    print(f"E003 Calibration-Holonomy Invariance Experiment Passed: {success}")
