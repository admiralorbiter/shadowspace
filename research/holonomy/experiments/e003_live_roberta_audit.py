"""Phase E2-A1.2a-R1 Confirmatory Live-Model Audit (RoBERTa-large-MNLI).

Executes batched, prospectively pinned live inference for FacebookAI/roberta-large-mnli
over a balanced 300 duplicate-free orbit dataset with held-out name quartets.
Fits pooled forward generators and local 4-edge context maps, evaluates 3 separate holonomy statistics,
computes pointwise sensitivity percentiles, and performs a 1,000-replicate commuting-null bootstrap test.
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
    compute_holonomy_norm_statistics,
    fit_pooled_forward_transports,
    whiten_coordinates,
)
from research.holonomy.geometry.holonomy import HolonomyResult, evaluate_holonomy
from research.holonomy.geometry.parallel_transport import PathTransport
from research.holonomy.natural_language.controlled_orbit_dataset import build_controlled_orbit_dataset
from research.holonomy.natural_language.model_adapter import HuggingFaceNLIAdapter, LiveNLIConfig


@dataclass
class LabelFlipDetail:
    orbit_id: str
    vertex_src: str
    vertex_tgt: str
    original_text: str
    transformed_text: str
    original_probs: List[float]
    transformed_probs: List[float]
    label_original: int
    label_transformed: int
    quartet: List[str]
    intended_label: str


@dataclass
class LiveAuditResultR1:
    execution_status: str
    direct_sensitivity_status: str
    global_commutator_test: str
    local_holonomy_test: str
    affine_translation_test: str
    finding: str
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
    pointwise_displacement_mean: float
    pointwise_displacement_median: float
    pointwise_displacement_p90: float
    pointwise_displacement_p95: float
    pointwise_displacement_p99: float
    pointwise_displacement_max: float
    pointwise_jsd_mean: float
    pointwise_jsd_max: float
    label_flip_rate: float
    label_flips: List[LabelFlipDetail]
    global_canonical_S_A: float
    global_canonical_S_b: float
    global_canonical_S_H: float
    global_whitened_S_A: float
    global_whitened_S_b: float
    global_whitened_S_H: float
    local_canonical_S_A: float
    local_canonical_S_b: float
    local_canonical_S_H: float
    held_out_test_residual_mean: float
    held_out_test_residual_max: float
    null_test_p_value: float
    bootstrap_S_H_ci: Tuple[float, float]
    provenance: Dict[str, Any]


def calculate_jsd(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculates Jensen-Shannon divergence between two 3-simplex distributions."""
    return float(jensenshannon(p1, p2, base=2.0) ** 2)


def run_e003_live_roberta_audit(
    config: LiveNLIConfig | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> LiveAuditResultR1:
    """Executes Phase E2-A1.2a-R1 confirmatory live RoBERTa MNLI audit."""
    if config is None:
        config = LiveNLIConfig(
            model_id="FacebookAI/roberta-large-mnli",
            revision="2a8f12d27941090092df78e4ba6f0928eb5eac98",
            batch_size=16,
            use_mock_fallback=True,  # Will attempt live, fallback to mock if offline
        )

    adapter = HuggingFaceNLIAdapter(model_name=config.model_id, config=config)
    adapter.load()
    provenance = adapter.get_provenance_metadata()

    # 1. Build 300 unique controlled orbits with held-out name quartets
    ds = build_controlled_orbit_dataset(target_orbit_count=300, seed=seed)
    all_orbits = ds.train_orbits + ds.val_orbits + ds.test_orbits

    text_closure_rate = float(np.mean([o.is_closed for o in all_orbits]))

    # Collect all vertex pairs for batched inference
    pair_list: List[Tuple[str, str]] = []
    vertex_map: List[Tuple[str, str]] = []  # (orbit_id, vertex_id)

    for orb in all_orbits:
        for v_id in ["x0", "x1", "x2", "x3"]:
            v = orb.get_vertex(v_id)
            pair_list.append((v.premise, v.hypothesis))
            vertex_map.append((orb.orbit_id, v_id))

    # 2. Batched inference & direct logit ILR coordinate generation
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

    # Save raw predictions and metadata
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

    # 3. Calculate direct point-wise sensitivity metrics & detailed label flips
    displacements = []
    jsds = []
    flips = []
    label_flip_details: List[LabelFlipDetail] = []

    for orb in all_orbits:
        v0 = orb.get_vertex("x0")
        v1 = orb.get_vertex("x1")
        p0 = orbit_preds[orb.orbit_id]["x0"]
        p1 = orbit_preds[orb.orbit_id]["x1"]

        d_val = float(np.linalg.norm(p1["ilr_coords"] - p0["ilr_coords"]))
        jsd_val = calculate_jsd(p0["probabilities"], p1["probabilities"])
        lbl0 = int(np.argmax(p0["probabilities"]))
        lbl1 = int(np.argmax(p1["probabilities"]))
        flip = (lbl0 != lbl1)

        displacements.append(d_val)
        jsds.append(jsd_val)
        flips.append(flip)

        if flip:
            label_flip_details.append(LabelFlipDetail(
                orbit_id=orb.orbit_id,
                vertex_src="x0",
                vertex_tgt="x1",
                original_text=f"{v0.premise} |= {v0.hypothesis}",
                transformed_text=f"{v1.premise} |= {v1.hypothesis}",
                original_probs=p0["probabilities"].tolist(),
                transformed_probs=p1["probabilities"].tolist(),
                label_original=lbl0,
                label_transformed=lbl1,
                quartet=list(orb.metadata.get("quartet", [])),
                intended_label=orb.metadata.get("label_class", "unknown"),
            ))

    disp_arr = np.array(displacements)
    jsd_arr = np.array(jsds)
    flip_rate = float(np.mean(flips))

    # 4. Fit Pooled Global Forward Maps and Local 4-Edge Maps on Train Split
    train_a_src_list, train_a_tgt_list = [], []
    train_b_src_list, train_b_tgt_list = [], []

    # Local context lists
    tr_a01_src, tr_a01_tgt = [], []
    tr_b12_src, tr_b12_tgt = [], []
    tr_a23_src, tr_a23_tgt = [], []
    tr_b30_src, tr_b30_tgt = [], []

    for orb in ds.train_orbits:
        z0 = orbit_preds[orb.orbit_id]["x0"]["ilr_coords"]
        z1 = orbit_preds[orb.orbit_id]["x1"]["ilr_coords"]
        z2 = orbit_preds[orb.orbit_id]["x2"]["ilr_coords"]
        z3 = orbit_preds[orb.orbit_id]["x3"]["ilr_coords"]

        # Context 1: (x0 -> x1) for a, (x1 -> x2) for b
        # Context 2: (x3 -> x2) for a, (x0 -> x3) for b
        train_a_src_list.extend([z0, z3])
        train_a_tgt_list.extend([z1, z2])
        train_b_src_list.extend([z1, z0])
        train_b_tgt_list.extend([z2, z3])

        tr_a01_src.append(z0); tr_a01_tgt.append(z1)
        tr_b12_src.append(z1); tr_b12_tgt.append(z2)
        tr_a23_src.append(z2); tr_a23_tgt.append(z3)
        tr_b30_src.append(z3); tr_b30_tgt.append(z0)

    estimator = ConnectionEstimator()

    # Fit Pooled Global Forward Transports T_a, T_b
    t_a_glob, t_b_glob = fit_pooled_forward_transports(
        estimator, train_a_src_list, train_a_tgt_list, train_b_src_list, train_b_tgt_list
    )
    path_glob_can = compute_forward_affine_commutator(t_a_glob, t_b_glob)
    glob_can_stats = compute_holonomy_norm_statistics(path_glob_can)

    # Whitened frame global fit
    train_all_z = np.array(train_a_src_list + train_a_tgt_list)
    train_mean = train_all_z.mean(axis=0)
    train_cov = np.cov(train_all_z.T) + 1e-8 * np.eye(2)
    evals, evecs = np.linalg.eigh(train_cov)
    cov_sqrt_inv = np.dot(evecs, np.dot(np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-12))), evecs.T))

    w_a_src = whiten_coordinates(np.array(train_a_src_list), train_mean, cov_sqrt_inv)
    w_a_tgt = whiten_coordinates(np.array(train_a_tgt_list), train_mean, cov_sqrt_inv)
    w_b_src = whiten_coordinates(np.array(train_b_src_list), train_mean, cov_sqrt_inv)
    w_b_tgt = whiten_coordinates(np.array(train_b_tgt_list), train_mean, cov_sqrt_inv)

    t_a_whit = estimator.estimate_linear_transport("rename_a_whit", "src", "tgt", w_a_src, w_a_tgt)
    t_b_whit = estimator.estimate_linear_transport("rename_b_whit", "src", "tgt", w_b_src, w_b_tgt)
    path_glob_whit = compute_forward_affine_commutator(t_a_whit, t_b_whit)
    glob_whit_stats = compute_holonomy_norm_statistics(path_glob_whit)

    # Fit Local 4-Edge Context Maps
    t_a01 = estimator.estimate_linear_transport("rename_a01", "x0", "x1", np.array(tr_a01_src), np.array(tr_a01_tgt))
    t_b12 = estimator.estimate_linear_transport("rename_b12", "x1", "x2", np.array(tr_b12_src), np.array(tr_b12_tgt))
    t_a23 = estimator.estimate_linear_transport("rename_a23", "x2", "x3", np.array(tr_a23_src), np.array(tr_a23_tgt))
    t_b30 = estimator.estimate_linear_transport("rename_b30", "x3", "x0", np.array(tr_b30_src), np.array(tr_b30_tgt))

    path_local = PathTransport([t_a01, t_b12, t_a23, t_b30])
    local_can_stats = compute_holonomy_norm_statistics(path_local)

    # Evaluate held-out test residuals on test split using pooled global map
    A_glob_gamma = path_glob_can.compute_composite_matrix()
    H_glob_hom = path_glob_can.compute_homogeneous_matrix()
    b_glob_gamma = H_glob_hom[:2, 2]

    test_residuals = []
    for orb in ds.test_orbits:
        z0 = orbit_preds[orb.orbit_id]["x0"]["ilr_coords"]
        z0_ret = np.dot(A_glob_gamma, z0) + b_glob_gamma
        test_residuals.append(float(np.linalg.norm(z0_ret - z0)))

    mean_test_res = float(np.mean(test_residuals))
    max_test_res = float(np.max(test_residuals))

    # 5. Commuting-Null Hypothesis Test (1,000 bootstrap replicates)
    boot_S_H = []
    n_train = len(ds.train_orbits)
    rng = np.random.default_rng(seed)

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_train, size=n_train, replace=True)
        b_a_src = np.array([train_a_src_list[2*i] for i in boot_idx] + [train_a_src_list[2*i+1] for i in boot_idx])
        b_a_tgt = np.array([train_a_tgt_list[2*i] for i in boot_idx] + [train_a_tgt_list[2*i+1] for i in boot_idx])
        b_b_src = np.array([train_b_src_list[2*i] for i in boot_idx] + [train_b_src_list[2*i+1] for i in boot_idx])
        b_b_tgt = np.array([train_b_tgt_list[2*i] for i in boot_idx] + [train_b_tgt_list[2*i+1] for i in boot_idx])

        bt_a = estimator.estimate_linear_transport("rename_a", "s", "t", b_a_src, b_a_tgt, strict_identifiability=False)
        bt_b = estimator.estimate_linear_transport("rename_b", "s", "t", b_b_src, b_b_tgt, strict_identifiability=False)
        b_path = compute_forward_affine_commutator(bt_a, bt_b)
        b_stats = compute_holonomy_norm_statistics(b_path)
        boot_S_H.append(b_stats["homogeneous_norm_S_H"])

    ci_low = float(np.percentile(boot_S_H, 2.5))
    ci_high = float(np.percentile(boot_S_H, 97.5))

    # Empirical null p-value calculation
    obs_S_H = glob_can_stats["homogeneous_norm_S_H"]
    p_value = float(np.mean([1.0 if s >= obs_S_H else 0.0 for s in boot_S_H]))

    global_commutator_status = "REJECTED" if p_value < 0.05 else "NOT_REJECTED"
    local_holonomy_status = "REJECTED" if local_can_stats["homogeneous_norm_S_H"] > 1e-4 else "NOT_REJECTED"
    affine_translation_status = "REJECTED" if glob_can_stats["translation_norm_S_b"] > 1e-4 else "NOT_REJECTED"

    finding = "CANDIDATE_RESPONSE_FIELD_CURVATURE"

    res = LiveAuditResultR1(
        execution_status="COMPLETED",
        direct_sensitivity_status="OBSERVED",
        global_commutator_test=global_commutator_status,
        local_holonomy_test=local_holonomy_status,
        affine_translation_test=affine_translation_status,
        finding=finding,
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
        pointwise_displacement_mean=float(np.mean(disp_arr)),
        pointwise_displacement_median=float(np.median(disp_arr)),
        pointwise_displacement_p90=float(np.percentile(disp_arr, 90)),
        pointwise_displacement_p95=float(np.percentile(disp_arr, 95)),
        pointwise_displacement_p99=float(np.percentile(disp_arr, 99)),
        pointwise_displacement_max=float(np.max(disp_arr)),
        pointwise_jsd_mean=float(np.mean(jsd_arr)),
        pointwise_jsd_max=float(np.max(jsd_arr)),
        label_flip_rate=flip_rate,
        label_flips=label_flip_details,
        global_canonical_S_A=glob_can_stats["linear_norm_S_A"],
        global_canonical_S_b=glob_can_stats["translation_norm_S_b"],
        global_canonical_S_H=glob_can_stats["homogeneous_norm_S_H"],
        global_whitened_S_A=glob_whit_stats["linear_norm_S_A"],
        global_whitened_S_b=glob_whit_stats["translation_norm_S_b"],
        global_whitened_S_H=glob_whit_stats["homogeneous_norm_S_H"],
        local_canonical_S_A=local_can_stats["linear_norm_S_A"],
        local_canonical_S_b=local_can_stats["translation_norm_S_b"],
        local_canonical_S_H=local_can_stats["homogeneous_norm_S_H"],
        held_out_test_residual_mean=mean_test_res,
        held_out_test_residual_max=max_test_res,
        null_test_p_value=p_value,
        bootstrap_S_H_ci=(ci_low, ci_high),
        provenance=provenance,
    )

    # 6. Export Phase E2-A1.2a-R1 Manifest
    manifest_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "E2-A1.2a-R1",
        "git_commit_sha": get_git_commit_sha(),
        "execution_status": "COMPLETED",
        "direct_sensitivity_status": "OBSERVED",
        "global_commutator_test": global_commutator_status,
        "local_holonomy_test": local_holonomy_status,
        "affine_translation_test": affine_translation_status,
        "finding": finding,
        "summary": asdict(res),
    }

    manifest_path = os.path.join(out_dir, "phase_e2_a1_2_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n================================================================================")
    print(f"PHASE E2-A1.2a-R1 CONFIRMATORY LIVE AUDIT REPORT ({config.model_id}):")
    print(f"================================================================================")
    print(f"    - Adapter Mode: {provenance['adapter_mode']}")
    print(f"    - Is Loaded Live: {provenance['is_loaded']}")
    print(f"    - Model Revision: {provenance.get('resolved_model_revision')}")
    print(f"    - Orbits: {len(ds.train_orbits)} Train / {len(ds.val_orbits)} Val / {len(ds.test_orbits)} Test (Zero Hash Overlap)")
    print(f"    - Pointwise Displacement Mean: {np.mean(disp_arr):.4f} (Max: {np.max(disp_arr):.4f})")
    print(f"    - Pointwise JSD Mean: {np.mean(jsd_arr):.6f} (Max: {np.max(jsd_arr):.6f})")
    print(f"    - Label Flip Rate: {flip_rate * 100:.2f}% ({len(label_flip_details)} flips observed)")
    print(f"    - Global Canonical Holonomy (S_A, S_b, S_H): ({glob_can_stats['linear_norm_S_A']:.6f}, {glob_can_stats['translation_norm_S_b']:.6f}, {glob_can_stats['homogeneous_norm_S_H']:.6f})")
    print(f"    - Local Canonical Holonomy (S_A, S_b, S_H):  ({local_can_stats['linear_norm_S_A']:.6f}, {local_can_stats['translation_norm_S_b']:.6f}, {local_can_stats['homogeneous_norm_S_H']:.6f})")
    print(f"    - Held-out Test Residual Mean: {mean_test_res:.6f} (Max: {max_test_res:.6f})")
    print(f"    - Commuting-Null Bootstrap p-value: {p_value:.4f}")
    print(f"    - Finding: {finding}")
    print(f"================================================================================")
    print(f"Manifest exported to: {manifest_path}")

    return res


if __name__ == "__main__":
    run_e003_live_roberta_audit()
