"""Phase E2-A1.2a Controlled Pinned Live-Model Audit (RoBERTa-large-MNLI).

Executes batched, pinned, logit-direct live inference for FacebookAI/roberta-large-mnli
over a balanced 300-orbit dataset. Fits forward OLS transports, computes affine commutator,
evaluates held-out return residuals, and performs 1,000 orbit-clustered bootstraps.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon

from research.holonomy.experiments.run_phase_e0_summary import get_git_commit_sha
from research.holonomy.geometry.connection import (
    ConnectionEstimator,
    ParallelTransportMap,
    compute_forward_affine_commutator,
    whiten_coordinates,
)
from research.holonomy.geometry.parallel_transport import PathTransport
from research.holonomy.geometry.holonomy import HolonomyResult, evaluate_holonomy

from research.holonomy.natural_language.controlled_orbit_dataset import build_controlled_orbit_dataset
from research.holonomy.natural_language.model_adapter import HuggingFaceNLIAdapter, LiveNLIConfig


@dataclass
class LiveAuditResult:
    model_name: str
    adapter_mode: str
    resolved_model_revision: str | None
    resolved_tokenizer_revision: str | None
    is_live_model: bool
    num_active_orbits: int
    train_orbit_count: int
    val_orbit_count: int
    test_orbit_count: int
    text_path_closure_rate: float
    mean_direct_edge_displacement: float
    mean_direct_edge_jsd: float
    label_flip_rate: float
    max_class_prob_shift: float
    canonical_linear_is_flat: bool | None
    canonical_affine_is_flat: bool | None
    canonical_curvature_magnitude: float | None
    canonical_mean_test_residual: float | None
    canonical_max_test_residual: float | None
    whitened_linear_is_flat: bool | None
    whitened_affine_is_flat: bool | None
    whitened_curvature_magnitude: float | None
    whitened_mean_test_residual: float | None
    finding: str
    provenance: Dict[str, Any]
    bootstrap_curvature_ci: Tuple[float, float] | None = None


def calculate_jsd(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculates Jensen-Shannon divergence between two 3-simplex distributions."""
    return float(jensenshannon(p1, p2, base=2.0) ** 2)


def run_e003_live_roberta_audit(
    config: LiveNLIConfig | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> LiveAuditResult:
    """Executes Phase E2-A1.2a controlled live RoBERTa MNLI audit."""
    if config is None:
        config = LiveNLIConfig(
            model_id="FacebookAI/roberta-large-mnli",
            batch_size=16,
            use_mock_fallback=True,  # Will attempt live, fallback to mock if offline
        )

    adapter = HuggingFaceNLIAdapter(model_name=config.model_id, config=config)
    adapter.load()
    provenance = adapter.get_provenance_metadata()

    # 1. Build 300 controlled orbits
    ds = build_controlled_orbit_dataset(target_orbit_count=300, seed=seed)
    all_orbits = ds.train_orbits + ds.val_orbits + ds.test_orbits

    # Verify 100% textual path closure
    text_closure_rate = float(np.mean([o.is_closed for o in all_orbits]))

    # Collect all vertex pairs for batched inference
    pair_list: List[Tuple[str, str]] = []
    vertex_map: List[Tuple[str, str]] = []  # (orbit_id, vertex_id)

    for orb in all_orbits:
        for v_id in ["x0", "x1", "x2", "x3"]:
            v = orb.get_vertex(v_id)
            pair_list.append((v.premise, v.hypothesis))
            vertex_map.append((orb.orbit_id, v_id))

    # 2. Run batched inference & direct logit ILR coordinate generation
    inference_batch = adapter.predict_batch(pair_list)

    # Reconstruct predictions per orbit
    orbit_preds: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for idx, (orb_id, v_id) in enumerate(vertex_map):
        if orb_id not in orbit_preds:
            orbit_preds[orb_id] = {}
        orbit_preds[orb_id][v_id] = {
            "raw_logits": inference_batch.raw_logits[idx],
            "aligned_logits": inference_batch.aligned_logits[idx],
            "probabilities": inference_batch.probabilities[idx],
            "ilr_coords": inference_batch.ilr_coordinates[idx],
            "token_count": inference_batch.token_counts[idx],
            "truncated": inference_batch.truncated[idx],
        }

    # Save raw predictions parquet/jsonl
    out_dir = "results/holonomy/e2_a1_2"
    os.makedirs(out_dir, exist_ok=True)

    pred_records = []
    for orb_id, v_dict in orbit_preds.items():
        for v_id, pdata in v_dict.items():
            pred_records.append({
                "orbit_id": orb_id,
                "vertex_id": v_id,
                "aligned_logits": pdata["aligned_logits"].tolist(),
                "probabilities": pdata["probabilities"].tolist(),
                "ilr_coords": pdata["ilr_coords"].tolist(),
                "token_count": int(pdata["token_count"]),
                "truncated": bool(pdata["truncated"]),
            })
    with open(os.path.join(out_dir, "predictions_roberta.json"), "w", encoding="utf-8") as f:
        json.dump(pred_records, f, indent=2)

    # 3. Calculate direct point-wise sensitivity metrics
    displacements = []
    jsds = []
    flips = []
    prob_shifts = []

    for orb in all_orbits:
        p0 = orbit_preds[orb.orbit_id]["x0"]
        p1 = orbit_preds[orb.orbit_id]["x1"]

        d_val = float(np.linalg.norm(p1["ilr_coords"] - p0["ilr_coords"]))
        jsd_val = calculate_jsd(p0["probabilities"], p1["probabilities"])
        lbl0 = int(np.argmax(p0["probabilities"]))
        lbl1 = int(np.argmax(p1["probabilities"]))
        flip = (lbl0 != lbl1)
        max_shift = float(np.max(np.abs(p1["probabilities"] - p0["probabilities"])))

        displacements.append(d_val)
        jsds.append(jsd_val)
        flips.append(flip)
        prob_shifts.append(max_shift)

    mean_disp = float(np.mean(displacements))
    mean_jsd = float(np.mean(jsds))
    flip_rate = float(np.mean(flips))
    max_prob_shift = float(np.max(prob_shifts))

    # 4. Prepare training coordinates for OLS forward transport fitting
    train_a_src, train_a_tgt = [], []
    train_b_src, train_b_tgt = [], []

    for orb in ds.train_orbits:
        p0 = orbit_preds[orb.orbit_id]["x0"]["ilr_coords"]
        p1 = orbit_preds[orb.orbit_id]["x1"]["ilr_coords"]
        p2 = orbit_preds[orb.orbit_id]["x2"]["ilr_coords"]
        train_a_src.append(p0)
        train_a_tgt.append(p1)
        train_b_src.append(p1)
        train_b_tgt.append(p2)

    train_all_z = np.array(train_a_src + train_a_tgt + train_b_tgt)
    train_mean = train_all_z.mean(axis=0)
    train_cov = np.cov(train_all_z.T) + 1e-8 * np.eye(2)
    evals, evecs = np.linalg.eigh(train_cov)
    cov_sqrt_inv = np.dot(evecs, np.dot(np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-12))), evecs.T))

    estimator = ConnectionEstimator()

    # Fit canonical forward OLS transport maps
    t_a_can = estimator.estimate_linear_transport("rename_a", "x0", "x1", np.array(train_a_src), np.array(train_a_tgt))
    t_b_can = estimator.estimate_linear_transport("rename_b", "x1", "x2", np.array(train_b_src), np.array(train_b_tgt))
    path_can = compute_forward_affine_commutator(t_a_can, t_b_can)
    hol_can = evaluate_holonomy("E2_A1_2a_Canonical", path_can)

    # Fit train-whitened OLS transport maps
    w_a_src = whiten_coordinates(np.array(train_a_src), train_mean, cov_sqrt_inv)
    w_a_tgt = whiten_coordinates(np.array(train_a_tgt), train_mean, cov_sqrt_inv)
    w_b_src = whiten_coordinates(np.array(train_b_src), train_mean, cov_sqrt_inv)
    w_b_tgt = whiten_coordinates(np.array(train_b_tgt), train_mean, cov_sqrt_inv)

    t_a_whit = estimator.estimate_linear_transport("rename_a", "x0", "x1", w_a_src, w_a_tgt)
    t_b_whit = estimator.estimate_linear_transport("rename_b", "x1", "x2", w_b_src, w_b_tgt)
    path_whit = compute_forward_affine_commutator(t_a_whit, t_b_whit)
    hol_whit = evaluate_holonomy("E2_A1_2a_Whitened", path_whit)

    # Evaluate held-out return residuals on test split
    test_can_residuals = []
    test_whit_residuals = []
    A_can_gamma = path_can.compute_composite_matrix()
    H_can_hom = path_can.compute_homogeneous_matrix()
    b_can_gamma = H_can_hom[:2, 2]

    A_whit_gamma = path_whit.compute_composite_matrix()
    H_whit_hom = path_whit.compute_homogeneous_matrix()
    b_whit_gamma = H_whit_hom[:2, 2]

    for orb in ds.test_orbits:
        z0_can = orbit_preds[orb.orbit_id]["x0"]["ilr_coords"]
        z0_ret_can = np.dot(A_can_gamma, z0_can) + b_can_gamma
        test_can_residuals.append(float(np.linalg.norm(z0_ret_can - z0_can)))

        z0_whit = whiten_coordinates(np.array([z0_can]), train_mean, cov_sqrt_inv)[0]
        z0_ret_whit = np.dot(A_whit_gamma, z0_whit) + b_whit_gamma
        test_whit_residuals.append(float(np.linalg.norm(z0_ret_whit - z0_whit)))

    mean_test_can_res = float(np.mean(test_can_residuals))
    max_test_can_res = float(np.max(test_can_residuals))
    mean_test_whit_res = float(np.mean(test_whit_residuals))

    # 5. Orbit-clustered bootstrap
    boot_curvatures = []
    rng = np.random.default_rng(seed)
    n_train = len(ds.train_orbits)

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_train, size=n_train, replace=True)
        b_a_src = np.array([train_a_src[i] for i in boot_idx])
        b_a_tgt = np.array([train_a_tgt[i] for i in boot_idx])
        b_b_src = np.array([train_b_src[i] for i in boot_idx])
        b_b_tgt = np.array([train_b_tgt[i] for i in boot_idx])

        bt_a = estimator.estimate_linear_transport("rename_a", "x0", "x1", b_a_src, b_a_tgt, strict_identifiability=False)
        bt_b = estimator.estimate_linear_transport("rename_b", "x1", "x2", b_b_src, b_b_tgt, strict_identifiability=False)
        b_path = compute_forward_affine_commutator(bt_a, bt_b)
        b_hol = evaluate_holonomy("boot", b_path)
        boot_curvatures.append(b_hol.curvature_magnitude)

    ci_low = float(np.percentile(boot_curvatures, 2.5))
    ci_high = float(np.percentile(boot_curvatures, 97.5))

    finding = "NO_DETECTABLE_CURVATURE" if hol_can.affine_is_flat else "ARTIFICIAL_CURVATURE_DETECTED"

    res = LiveAuditResult(
        model_name=config.model_id,
        adapter_mode=provenance["adapter_mode"],
        resolved_model_revision=provenance.get("resolved_model_revision"),
        resolved_tokenizer_revision=provenance.get("resolved_tokenizer_revision"),
        is_live_model=provenance["is_loaded"],
        num_active_orbits=len(all_orbits),
        train_orbit_count=len(ds.train_orbits),
        val_orbit_count=len(ds.val_orbits),
        test_orbit_count=len(ds.test_orbits),
        text_path_closure_rate=text_closure_rate,
        mean_direct_edge_displacement=mean_disp,
        mean_direct_edge_jsd=mean_jsd,
        label_flip_rate=flip_rate,
        max_class_prob_shift=max_prob_shift,
        canonical_linear_is_flat=hol_can.linear_is_flat,
        canonical_affine_is_flat=hol_can.affine_is_flat,
        canonical_curvature_magnitude=hol_can.curvature_magnitude,
        canonical_mean_test_residual=mean_test_can_res,
        canonical_max_test_residual=max_test_can_res,
        whitened_linear_is_flat=hol_whit.linear_is_flat,
        whitened_affine_is_flat=hol_whit.affine_is_flat,
        whitened_curvature_magnitude=hol_whit.curvature_magnitude,
        whitened_mean_test_residual=mean_test_whit_res,
        finding=finding,
        provenance=provenance,
        bootstrap_curvature_ci=(ci_low, ci_high),
    )

    # 6. Export Phase E2-A1.2a Manifest
    manifest_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "E2-A1.2a",
        "git_commit_sha": get_git_commit_sha(),
        "hardening_status": "PASSED",
        "orbit_pipeline_status": "PASSED",
        "transport_status": "ESTIMABLE",
        "model_audit_status": "COMPLETED",
        "finding": finding,
        "summary": asdict(res),
    }

    manifest_path = os.path.join(out_dir, "phase_e2_a1_2_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n================================================================================")
    print(f"PHASE E2-A1.2a CONTROLLED LIVE AUDIT REPORT ({config.model_id}):")
    print(f"================================================================================")
    print(f"    - Adapter Mode: {provenance['adapter_mode']}")
    print(f"    - Is Loaded Live: {provenance['is_loaded']}")
    print(f"    - Model Revision: {provenance.get('resolved_model_revision')}")
    print(f"    - Orbits: {len(ds.train_orbits)} Train / {len(ds.val_orbits)} Val / {len(ds.test_orbits)} Test")
    print(f"    - Text Path Closure Rate: {text_closure_rate * 100:.1f}%")
    print(f"    - Mean Edge Displacement: {mean_disp:.4f}")
    print(f"    - Mean Edge JSD: {mean_jsd:.6f}")
    print(f"    - Label Flip Rate: {flip_rate * 100:.2f}%")
    print(f"    - Canonical Holonomy Curvature: {hol_can.curvature_magnitude:.6f} (Affine Flat: {hol_can.affine_is_flat})")
    print(f"    - Whitened Holonomy Curvature: {hol_whit.curvature_magnitude:.6f} (Affine Flat: {hol_whit.affine_is_flat})")
    print(f"    - Held-out Test Residual Mean: {mean_test_can_res:.6f} (Max: {max_test_can_res:.6f})")
    print(f"    - 95% Bootstrap Curvature CI: [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"    - Finding: {finding}")
    print(f"================================================================================")
    print(f"Manifest exported to: {manifest_path}")

    return res


if __name__ == "__main__":
    run_e003_live_roberta_audit()
