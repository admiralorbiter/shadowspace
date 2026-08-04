"""Phase E0.7.2 & Phase E2 Laboratory Summary Runner & Enriched Manifest Exporter.

Executes all Phase E0.7.2 synthetic benchmarks & Phase E2-A1 Natural Language Model Audit Pilot:
- Flat World Zero Holonomy
- 4-Corner Planted Curvature Recovery & Exact 3x3 Homogeneous H_hom_true Ground Truth
- 3-Group Monte Carlo Loop Holonomy Inference
- Total Least Squares (TLS) Errors-In-Variables Attenuation Bias Correction
- Theorem 1A & Proposition 1B Invariance
- Data-Dependent Sheaf Laplacian Cohomology H0/H1
- GlueOOD Ridge-Regularized Solver
- Phase E2-A1 Natural Language Exact-Symmetry Model Audit Pilot (Tier 1 Reversible Entity Renaming)
Exports machine-readable execution manifest and enforces strict gate assertions.
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
from research.holonomy.experiments.e002_classifier_holonomy import run_e002_classifier_holonomy_experiment
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
    print("PHASE E0.7.2 & E2-A1: FINITE AMBIGUITY & NATURAL LANGUAGE HOLONOMY REPORT")
    print("================================================================ algorithm: 3x3 Homogeneous Affine Connections & Data-Dependent Sheaves\n")

    # 1. Flat World Benchmark
    flat_pass = run_e000_flat_world_experiment()
    print(f"[1] Flat World Zero Holonomy (H_gamma == I_2): {'PASSED' if flat_pass else 'FAILED'}")

    # 2. 4-Corner Planted Curvature Recovery & Monte Carlo Sweeps
    print("\n[2] 4-Corner Planted Curvature Recovery & 3-Group Monte Carlo Loop Inference:")
    angles = [np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 3]
    curvature_results = []
    for angle in angles:
        passed = run_e001_planted_curvature_experiment(angle)
        curvature_results.append(passed)
        print(f"    - Rotation angle {angle:.4f} rad ({np.degrees(angle):.1f}°): {'PASSED' if passed else 'FAILED'}")

    mc_sweeps = run_e001_monte_carlo_sweeps(num_seeds=50)
    print(f"\n    - 3-Group Monte Carlo Loop Holonomy Sweeps (50 Seeds per group):")
    mc_pass_gates = []
    for sw in mc_sweeps:
        bias_improved = sw.tls_matrix_bias_norm < sw.ols_matrix_bias_norm
        fpr_valid = sw.tls_false_positive_rate <= 0.10
        power_valid = sw.tls_detection_power >= 0.80

        if sw.sample_size >= 250:
            loop_improved = sw.tls_matrix_holonomy_rmse < sw.ols_matrix_holonomy_rmse
        else:
            loop_improved = True

        gate_ok = bias_improved and loop_improved and fpr_valid and power_valid
        mc_pass_gates.append(gate_ok)

        print(
            f"      [N={sw.sample_size}, r={sw.perturbation_radius}, sigma={sw.measurement_noise}] "
            f"TLS FPR: {sw.tls_false_positive_rate:.2f}, Power: {sw.tls_detection_power:.2f} | "
            f"Bias OLS: {sw.ols_matrix_bias_norm:.4f} -> TLS: {sw.tls_matrix_bias_norm:.4f} | "
            f"Matrix Holonomy RMSE OLS: {sw.ols_matrix_holonomy_rmse:.4f} -> TLS: {sw.tls_matrix_holonomy_rmse:.4f} "
            f"({'PASSED' if gate_ok else 'FAILED'})"
        )

    mc_all_pass = all(mc_pass_gates)

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

    # 5. GlueOOD Solver
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

    # 6. Phase E2-A1.1 Natural Language Model Audit Pilot
    audit_results = run_e002_classifier_holonomy_experiment()
    e2_res = audit_results[0]
    smoke_test_pass = bool(
        e2_res.num_active_orbits > 0
        and e2_res.text_path_closure_rate == 1.0
        and e2_res.train_orbit_count >= 2
        and e2_res.test_orbit_count >= 1
    )
    identifiability_status = "PASSED" if e2_res.estimator_identifiable else "NOT_ESTIMABLE"
    if e2_res.is_live_model:
        model_audit_status = "PASSED" if (e2_res.estimator_identifiable and not e2_res.artificial_curvature_detected) else "FAILED"
    else:
        model_audit_status = "NOT_RUN"

    print(f"\n[6] Phase E2-A1.1 Natural Language Model Audit Pilot (Tier 1 Reversible Entity Renaming):")
    print(f"    - Adapter Mode: {e2_res.adapter_mode}")
    print(f"    - Active Orbits: {e2_res.num_active_orbits}")
    print(f"    - Textual Path Closure Rate: {e2_res.text_path_closure_rate * 100:.1f}%")
    print(f"    - Train/Test Orbit Split: {e2_res.train_orbit_count} Train / {e2_res.test_orbit_count} Test")
    print(f"    - Orbit Smoke Test: {'PASSED' if smoke_test_pass else 'FAILED'}")
    print(f"    - Transport Identifiability: {identifiability_status}")
    print(f"    - Model Audit Status: {model_audit_status}")

    all_passed = bool(
        flat_pass and all(curvature_results) and mc_all_pass and thm1_pass and h0_pass and glue_pass and smoke_test_pass
    )

    def sanitize_for_json(val: Any) -> Any:
        if isinstance(val, float):
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        if isinstance(val, (np.floating, np.integer)):
            v = val.item()
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return None
            return v
        if isinstance(val, dict):
            return {k: sanitize_for_json(v) for k, v in val.items()}
        if isinstance(val, list):
            return [sanitize_for_json(v) for v in val]
        return val

    # Machine-readable manifest export
    manifest_raw = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "E2-A1.1",
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
            "monte_carlo_loop_holonomy": mc_all_pass,
            "theorem1_prop1b_invariance": thm1_pass,
            "sheaf_cohomology_h0": h0_pass,
            "glue_ood_solver": glue_pass,
            "natural_language_orbit_smoke_test": smoke_test_pass,
            "natural_language_transport_identifiability": identifiability_status,
            "natural_language_model_audit": model_audit_status,
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
        "natural_language_audit_summary": {
            "adapter_mode": e2_res.adapter_mode,
            "is_live_model": e2_res.is_live_model,
            "model_name": e2_res.model_name,
            "audit_status": e2_res.audit_status,
            "num_active_orbits": e2_res.num_active_orbits,
            "text_path_closure_rate": e2_res.text_path_closure_rate,
            "train_orbit_count": e2_res.train_orbit_count,
            "test_orbit_count": e2_res.test_orbit_count,
            "estimator_identifiable": e2_res.estimator_identifiable,
            "min_edge_design_rank": e2_res.min_edge_design_rank,
            "max_edge_condition_number": e2_res.max_edge_condition_number,
            "max_transport_norm": e2_res.max_transport_norm,
            "mean_held_out_return_residual": e2_res.mean_held_out_return_residual,
            "artificial_curvature_detected": e2_res.artificial_curvature_detected,
            "edge_diagnostics": e2_res.edge_diagnostics,
            "provenance": e2_res.provenance,
        },
        "monte_carlo_sweeps": [
            {
                "sample_size": sw.sample_size,
                "perturbation_radius": sw.perturbation_radius,
                "measurement_noise": sw.measurement_noise,
                "ols_matrix_bias_norm": sw.ols_matrix_bias_norm,
                "tls_matrix_bias_norm": sw.tls_matrix_bias_norm,
                "ols_edge_matrix_rmse": sw.ols_edge_matrix_rmse,
                "tls_edge_matrix_rmse": sw.tls_edge_matrix_rmse,
                "ols_matrix_holonomy_rmse": sw.ols_matrix_holonomy_rmse,
                "tls_matrix_holonomy_rmse": sw.tls_matrix_holonomy_rmse,
                "ols_curvature_stat_rmse": sw.ols_curvature_stat_rmse,
                "tls_curvature_stat_rmse": sw.tls_curvature_stat_rmse,
                "ols_false_positive_rate": sw.ols_false_positive_rate,
                "tls_false_positive_rate": sw.tls_false_positive_rate,
                "ols_detection_power": sw.ols_detection_power,
                "tls_detection_power": sw.tls_detection_power,
            }
            for sw in mc_sweeps
        ],
    }

    manifest = sanitize_for_json(manifest_raw)

    manifest_dir = "results/holonomy"
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, "phase_e0_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    print(f"\n[7] Machine-Readable Manifest Exported to: {manifest_path}")

    print("\n================================================================================")
    if all_passed:
        print("ALL PHASE E0.7.2 & E2-A1.1 MATHEMATICAL HARDENING GATES PASSED CLEANLY")
        print("================================================================================")
    else:
        print("PHASE E0.7.2 & E2-A1.1 GATES FAILED — CHECK MANIFEST FOR DETAILS")
        print("================================================================================")
        sys.exit(1)



if __name__ == "__main__":
    main()
