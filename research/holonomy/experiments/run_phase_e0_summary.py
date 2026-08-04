"""Phase E0.7 Synthetic Laboratory Summary Runner & Enriched Manifest Exporter.

Executes all Phase E0.7 synthetic benchmarks (Flat World, 4-Corner Planted Curvature,
Monte Carlo FPR/Power Sweeps, TLS Errors-in-Variables Bias Correction, Theorem 1A & Prop 1B Invariance,
Data-Dependent Sheaf Cohomology H0/H1, GlueOOD Solver) and exports machine-readable execution manifest.
Enforces strict gate assertions and exits with SystemExit(1) on any failure.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
import numpy as np
import scipy

from research.holonomy.experiments.e000_flat_world import run_e000_flat_world_experiment
from research.holonomy.experiments.e001_planted_curvature import (
    run_e001_monte_carlo_sweeps,
    run_e001_planted_curvature_experiment,
)
from research.holonomy.experiments.e003_calibration_invariance import run_e003_calibration_invariance_experiment
from research.holonomy.geometry.connection import ParallelTransportMap
from research.holonomy.sheaf.coboundary import CoboundaryOperator, OverlapEdge
from research.holonomy.sheaf.laplacian import SheafLaplacian
from research.holonomy.sheaf.ood_gluing import compute_glue_ood_score


def get_git_commit_sha() -> str:
    try:
        import subprocess
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd()).decode("utf-8").strip()
        return out
    except Exception:
        return "UNKNOWN"


def main() -> None:
    print("================================================================================")
    print("PHASE E0.7: FINITE AMBIGUITY LABORATORY — MONTE CARLO & TLS SUMMARY REPORT")
    print("================================================================ algorithm: 3x3 Homogeneous Affine Connections & Data-Dependent Sheaves\n")

    # 1. Flat World Benchmark
    flat_pass = run_e000_flat_world_experiment()
    print(f"[1] Flat World Zero Holonomy (H_gamma == I_2): {'PASSED' if flat_pass else 'FAILED'}")

    # 2. 4-Corner Planted Curvature Recovery & Monte Carlo Sweeps
    print("\n[2] 4-Corner Planted Curvature Recovery & Monte Carlo Inference:")
    angles = [np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 3]
    curvature_results = []
    for angle in angles:
        passed = run_e001_planted_curvature_experiment(angle)
        curvature_results.append(passed)
        print(f"    - Rotation angle {angle:.4f} rad ({np.degrees(angle):.1f}°): {'PASSED' if passed else 'FAILED'}")

    mc_sweeps = run_e001_monte_carlo_sweeps(num_seeds=50)
    print(f"\n    - Monte Carlo Statistical Sweeps (50 Seeds per config):")
    for sw in mc_sweeps:
        print(
            f"      [N={sw.sample_size}, r={sw.perturbation_radius}, sigma={sw.measurement_noise}] "
            f"FPR: {sw.false_positive_rate:.2f}, Power: {sw.detection_power:.2f} | "
            f"OLS Bias: {sw.ols_matrix_bias_norm:.4f}, TLS Bias: {sw.tls_matrix_bias_norm:.4f}"
        )

    # 3. Theorem 1A & Proposition 1B Invariance
    thm1_pass = run_e003_calibration_invariance_experiment()
    print(f"\n[3] Theorem 1A & Proposition 1B Invariance (Smooth Nonlinear Recalibration Bounds): {'PASSED' if thm1_pass else 'FAILED'}")

    # 4. Data-Dependent Sheaf Laplacian Cohomology
    patches = [f"U{i}" for i in range(4)]
    item_coords = np.array([[0.5, -0.3], [0.1, 0.4], [-0.2, 0.8]], dtype=np.float64)
    overlaps = [OverlapEdge(f"U{i}", f"U{i+1}", item_coords) for i in range(3)]
    cob = CoboundaryOperator(patches, overlaps)
    lap = SheafLaplacian(cob, param_dim=6)
    spec = lap.compute_spectrum()
    h0_pass = spec.dim_H0 == 6
    print(f"\n[4] Data-Dependent Sheaf Laplacian Cohomology:")
    print(f"    - Patches: {len(patches)}, Overlaps: {len(overlaps)}")
    print(f"    - Kernel dimension dim H^0: {spec.dim_H0} (Expected: 6) -> {'PASSED' if h0_pass else 'FAILED'}")
    print(f"    - Cohomological obstruction dim H^1: {spec.dim_H1}")
    print(f"    - Algebraic connectivity (Fiedler): {spec.fiedler_value:.6f}")

    # 5. GlueOOD Least-Squares Solver
    s1 = np.array([0.5, -0.2])
    s2 = np.array([0.5, -0.2])
    s3 = np.array([-1.0, 2.0])

    R_id = ParallelTransportMap("id", "u", "v", np.eye(2), np.zeros(2))
    score_coherent = compute_glue_ood_score([s1, s2], [R_id, R_id])
    score_incoherent = compute_glue_ood_score([s1, s3], [R_id, R_id])

    glue_pass = (score_coherent < 1e-4) and (score_incoherent > 0.5)
    print(f"\n[5] GlueOOD Ridge-Regularized Least-Squares Consensus Solver:")
    print(f"    - Coherent extension GlueOOD score: {score_coherent:.6f}")
    print(f"    - Incoherent extension GlueOOD score: {score_incoherent:.6f} -> {'PASSED' if glue_pass else 'FAILED'}")

    all_passed = bool(
        flat_pass and all(curvature_results) and thm1_pass and h0_pass and glue_pass
    )

    # Machine-readable manifest export
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "E0.7",
        "git_commit_sha": get_git_commit_sha(),
        "environment": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
        "status": "PASSED" if all_passed else "FAILED",
        "gates": {
            "flat_world": flat_pass,
            "planted_curvature_4corner": all(curvature_results),
            "theorem1_prop1b_invariance": thm1_pass,
            "sheaf_cohomology_h0": h0_pass,
            "glue_ood_solver": glue_pass,
        },
        "cohomology": {
            "dim_H0": spec.dim_H0,
            "dim_H1": spec.dim_H1,
            "fiedler_value": float(spec.fiedler_value),
        },
        "glue_ood": {
            "coherent_score": float(score_coherent),
            "incoherent_score": float(score_incoherent),
        },
        "monte_carlo_sweeps": [
            {
                "sample_size": sw.sample_size,
                "perturbation_radius": sw.perturbation_radius,
                "measurement_noise": sw.measurement_noise,
                "ols_matrix_bias_norm": sw.ols_matrix_bias_norm,
                "tls_matrix_bias_norm": sw.tls_matrix_bias_norm,
                "ols_holonomy_rmse": sw.ols_holonomy_rmse,
                "tls_holonomy_rmse": sw.tls_holonomy_rmse,
                "false_positive_rate": sw.false_positive_rate,
                "detection_power": sw.detection_power,
            }
            for sw in mc_sweeps
        ],
    }

    manifest_dir = "results/holonomy"
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, "phase_e0_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[6] Machine-Readable Manifest Exported to: {manifest_path}")

    print("\n================================================================================")
    if all_passed:
        print("ALL PHASE E0.7 MATHEMATICAL HARDENING GATES PASSED CLEANLY")
        print("================================================================================")
    else:
        print("PHASE E0.7 GATES FAILED — CHECK MANIFEST FOR DETAILS")
        print("================================================================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
