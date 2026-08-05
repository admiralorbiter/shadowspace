"""Phase E2-A1.2a-R1.1 Confirmatory Live-Model Audit (RoBERTa-large-MNLI).

Executes batched, prospectively pinned live inference for FacebookAI/roberta-large-mnli
over a balanced 300 duplicate-free orbit dataset with held-out name quartets and formal vs bias tracks.
Evaluates held-out edge predictive skill (RMSE, MAE, R2, skill vs identity), 4-edge sensitivity percentiles,
constrained commuting-null bootstrap, and direct rename-context interaction permutation testing.
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
    compute_rename_context_interaction_test,
    evaluate_edge_predictive_skill,
    fit_constrained_commuting_transports,
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
    flip_description: str


@dataclass
class LiveAuditResultR11:
    execution_status: str
    direct_sensitivity_status: str
    global_commutator_test: str
    local_holonomy_status: str
    affine_translation_status: str
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
    generalization_axis: str
    template_ood: bool
    name_ood: bool
    edge_predictive_skill_ta_rmse: float
    edge_predictive_skill_ta_mae: float
    edge_predictive_skill_ta_r2: float
    edge_predictive_skill_ta_skill_vs_identity: float
    edge_predictive_skill_tb_rmse: float
    edge_predictive_skill_tb_mae: float
    edge_predictive_skill_tb_r2: float
    edge_predictive_skill_tb_skill_vs_identity: float
    all_edges_displacement_mean: float
    all_edges_displacement_median: float
    all_edges_displacement_p90: float
    all_edges_displacement_p95: float
    all_edges_displacement_p99: float
    all_edges_displacement_max: float
    formal_track_displacement_mean: float
    formal_track_displacement_max: float
    bias_track_displacement_mean: float
    bias_track_displacement_max: float
    pointwise_jsd_mean: float
    pointwise_jsd_max: float
    label_flip_rate: float
    label_flips: List[LabelFlipDetail]
    top_10_sensitive_orbits: List[Dict[str, Any]]
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
    constrained_null_p_value: float
    rename_context_interaction_norm: float
    rename_context_interaction_p_value: float
    bootstrap_S_H_ci: Tuple[float, float]
    provenance: Dict[str, Any]


def calculate_jsd(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculates Jensen-Shannon divergence between two 3-simplex distributions."""
    return float(jensenshannon(p1, p2, base=2.0) ** 2)


def run_e003_live_roberta_audit(
    config: LiveNLIConfig | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> LiveAuditResultR11:
    """Executes Phase E2-A1.2a-R1.1 confirmatory live RoBERTa MNLI audit."""
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

    # Save rich prediction records JSON/Parquet
    out_dir = "results/holonomy/e2_a1_2"
    os.makedirs(out_dir, exist_ok=True)

    pred_records = []
    for orb in all_orbits:
        for v_id in ["x0", "x1", "x2", "x3"]:
            v = orb.get_vertex(v_id)
            pdata = orbit_preds[orb.orbit_id][v_id]
            pred_records.append({
                "orbit_id": orb.orbit_id,
                "vertex_id": v_id,
                "premise": v.premise,
                "hypothesis": v.hypothesis,
                "raw_logits": pdata["raw_logits"].tolist(),
                "aligned_logits": pdata["aligned_logits"].tolist(),
                "probabilities": pdata["probabilities"].tolist(),
                "ilr_coords": pdata["ilr_coords"].tolist(),
                "token_count": int(pdata["token_count"]),
                "truncated": bool(pdata["truncated"]),
                "track": orb.metadata.get("track", "unknown"),
                "label_class": orb.metadata.get("label_class", "unknown"),
            })
    with open(os.path.join(out_dir, "predictions_roberta.json"), "w", encoding="utf-8") as f:
        json.dump(pred_records, f, indent=2)

    # 3. Calculate 4-Edge Direct Pointwise Sensitivity & Formal vs Bias Tracks
    all_edge_displacements = []
    formal_track_displacements = []
    bias_track_displacements = []
    jsds = []
    flips = []
    label_flip_details: List[LabelFlipDetail] = []
    orbit_sensitivity_scores = []

    for orb in all_orbits:
        v0 = orb.get_vertex("x0")
        v1 = orb.get_vertex("x1")
        v2 = orb.get_vertex("x2")
        v3 = orb.get_vertex("x3")

        p0 = orbit_preds[orb.orbit_id]["x0"]
        p1 = orbit_preds[orb.orbit_id]["x1"]
        p2 = orbit_preds[orb.orbit_id]["x2"]
        p3 = orbit_preds[orb.orbit_id]["x3"]

        # All 4 edge displacements
        d_a1 = float(np.linalg.norm(p1["ilr_coords"] - p0["ilr_coords"]))
        d_b1 = float(np.linalg.norm(p2["ilr_coords"] - p1["ilr_coords"]))
        d_a2 = float(np.linalg.norm(p3["ilr_coords"] - p2["ilr_coords"]))
        d_b2 = float(np.linalg.norm(p0["ilr_coords"] - p3["ilr_coords"]))

        orb_disps = [d_a1, d_b1, d_a2, d_b2]
        all_edge_displacements.extend(orb_disps)

        track = orb.metadata.get("track", "unknown")
        if track == "formal_invariance":
            formal_track_displacements.extend(orb_disps)
        else:
            bias_track_displacements.extend(orb_disps)

        jsd_val = calculate_jsd(p0["probabilities"], p1["probabilities"])
        lbl0 = int(np.argmax(p0["probabilities"]))
        lbl1 = int(np.argmax(p1["probabilities"]))
        flip = (lbl0 != lbl1)

        jsds.append(jsd_val)
        flips.append(flip)
        max_orb_disp = float(np.max(orb_disps))

        orbit_sensitivity_scores.append({
            "orbit_id": orb.orbit_id,
            "max_displacement": max_orb_disp,
            "mean_displacement": float(np.mean(orb_disps)),
            "jsd": jsd_val,
            "base_text": f"{v0.premise} |= {v0.hypothesis}",
            "track": track,
            "label_class": orb.metadata.get("label_class", "unknown"),
        })

        if flip:
            # Map index to class string
            class_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
            lbl0_str = class_map.get(lbl0, str(lbl0))
            lbl1_str = class_map.get(lbl1, str(lbl1))
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
                flip_description=f"{lbl0_str} -> {lbl1_str}",
            ))

    disp_arr = np.array(all_edge_displacements)
    formal_disp_arr = np.array(formal_track_displacements) if formal_track_displacements else disp_arr
    bias_disp_arr = np.array(bias_track_displacements) if bias_track_displacements else disp_arr
    jsd_arr = np.array(jsds)
    flip_rate = float(np.mean(flips))

    # Top 10 most sensitive orbits
    orbit_sensitivity_scores.sort(key=lambda x: x["max_displacement"], reverse=True)
    top_10_orbits = orbit_sensitivity_scores[:10]

    # 4. Fit Pooled Global Forward Transports T_a, T_b on Train Split
    train_a_src_list, train_a_tgt_list = [], []
    train_b_src_list, train_b_tgt_list = [], []
    tr_a01_src, tr_a01_tgt = [], []
    tr_b12_src, tr_b12_tgt = [], []
    tr_a23_src, tr_a23_tgt = [], []
    tr_b30_src, tr_b30_tgt = [], []

    for orb in ds.train_orbits:
        z0 = orbit_preds[orb.orbit_id]["x0"]["ilr_coords"]
        z1 = orbit_preds[orb.orbit_id]["x1"]["ilr_coords"]
        z2 = orbit_preds[orb.orbit_id]["x2"]["ilr_coords"]
        z3 = orbit_preds[orb.orbit_id]["x3"]["ilr_coords"]

        train_a_src_list.extend([z0, z3])
        train_a_tgt_list.extend([z1, z2])
        train_b_src_list.extend([z1, z0])
        train_b_tgt_list.extend([z2, z3])

        tr_a01_src.append(z0); tr_a01_tgt.append(z1)
        tr_b12_src.append(z1); tr_b12_tgt.append(z2)
        tr_a23_src.append(z2); tr_a23_tgt.append(z3)
        tr_b30_src.append(z3); tr_b30_tgt.append(z0)

    estimator = ConnectionEstimator()

    t_a_glob, t_b_glob = fit_pooled_forward_transports(
        estimator, train_a_src_list, train_a_tgt_list, train_b_src_list, train_b_tgt_list
    )
    path_glob_can = compute_forward_affine_commutator(t_a_glob, t_b_glob)
    glob_can_stats = compute_holonomy_norm_statistics(path_glob_can)

    # 5. Evaluate Held-Out Edge Predictive Skill on Test Split
    test_a_src = np.array([orbit_preds[o.orbit_id]["x0"]["ilr_coords"] for o in ds.test_orbits] + [orbit_preds[o.orbit_id]["x3"]["ilr_coords"] for o in ds.test_orbits])
    test_a_tgt = np.array([orbit_preds[o.orbit_id]["x1"]["ilr_coords"] for o in ds.test_orbits] + [orbit_preds[o.orbit_id]["x2"]["ilr_coords"] for o in ds.test_orbits])
    test_b_src = np.array([orbit_preds[o.orbit_id]["x1"]["ilr_coords"] for o in ds.test_orbits] + [orbit_preds[o.orbit_id]["x0"]["ilr_coords"] for o in ds.test_orbits])
    test_b_tgt = np.array([orbit_preds[o.orbit_id]["x2"]["ilr_coords"] for o in ds.test_orbits] + [orbit_preds[o.orbit_id]["x3"]["ilr_coords"] for o in ds.test_orbits])

    skill_ta = evaluate_edge_predictive_skill(t_a_glob, test_a_src, test_a_tgt)
    skill_tb = evaluate_edge_predictive_skill(t_b_glob, test_b_src, test_b_tgt)

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

    # Local 4-Edge Context Maps
    t_a01 = estimator.estimate_linear_transport("rename_a01", "x0", "x1", np.array(tr_a01_src), np.array(tr_a01_tgt))
    t_b12 = estimator.estimate_linear_transport("rename_b12", "x1", "x2", np.array(tr_b12_src), np.array(tr_b12_tgt))
    t_a23 = estimator.estimate_linear_transport("rename_a23", "x2", "x3", np.array(tr_a23_src), np.array(tr_a23_tgt))
    t_b30 = estimator.estimate_linear_transport("rename_b30", "x3", "x0", np.array(tr_b30_src), np.array(tr_b30_tgt))

    path_local = PathTransport([t_a01, t_b12, t_a23, t_b30])
    local_can_stats = compute_holonomy_norm_statistics(path_local)

    # Evaluate held-out test residuals
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

    # 6. Constrained Commuting-Null Bootstrap & Direct Interaction Test
    t_a_c, t_b_c = fit_constrained_commuting_transports(
        np.array(train_a_src_list), np.array(train_a_tgt_list),
        np.array(train_b_src_list), np.array(train_b_tgt_list)
    )

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

    obs_S_H = glob_can_stats["homogeneous_norm_S_H"]
    constrained_p_val = float((1.0 + np.sum(np.array(boot_S_H) >= obs_S_H)) / (n_bootstrap + 1.0))

    # Direct rename-context interaction permutation test
    orbit_coords_map = {
        orb.orbit_id: {
            "x0": orbit_preds[orb.orbit_id]["x0"]["ilr_coords"],
            "x1": orbit_preds[orb.orbit_id]["x1"]["ilr_coords"],
            "x2": orbit_preds[orb.orbit_id]["x2"]["ilr_coords"],
            "x3": orbit_preds[orb.orbit_id]["x3"]["ilr_coords"],
        }
        for orb in all_orbits
    }
    interaction_test = compute_rename_context_interaction_test(orbit_coords_map, n_permutations=1000, seed=seed)

    global_commutator_status = "NOT_REJECTED" if constrained_p_val > 0.05 else "REJECTED"
    local_holonomy_status = "DESCRIPTIVE_ONLY"
    affine_translation_status = "DESCRIPTIVE_ONLY"
    finding = "CANDIDATE_RESPONSE_FIELD_CURVATURE"

    res = LiveAuditResultR11(
        execution_status="COMPLETED",
        direct_sensitivity_status="OBSERVED",
        global_commutator_test=global_commutator_status,
        local_holonomy_status=local_holonomy_status,
        affine_translation_status=affine_translation_status,
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
        generalization_axis="held_out_name_quartets",
        template_ood=False,
        name_ood=True,
        edge_predictive_skill_ta_rmse=skill_ta["rmse_affine"],
        edge_predictive_skill_ta_mae=skill_ta["mae_affine"],
        edge_predictive_skill_ta_r2=skill_ta["r2_affine"],
        edge_predictive_skill_ta_skill_vs_identity=skill_ta["relative_skill_vs_identity"],
        edge_predictive_skill_tb_rmse=skill_tb["rmse_affine"],
        edge_predictive_skill_tb_mae=skill_tb["mae_affine"],
        edge_predictive_skill_tb_r2=skill_tb["r2_affine"],
        edge_predictive_skill_tb_skill_vs_identity=skill_tb["relative_skill_vs_identity"],
        all_edges_displacement_mean=float(np.mean(disp_arr)),
        all_edges_displacement_median=float(np.median(disp_arr)),
        all_edges_displacement_p90=float(np.percentile(disp_arr, 90)),
        all_edges_displacement_p95=float(np.percentile(disp_arr, 95)),
        all_edges_displacement_p99=float(np.percentile(disp_arr, 99)),
        all_edges_displacement_max=float(np.max(disp_arr)),
        formal_track_displacement_mean=float(np.mean(formal_disp_arr)),
        formal_track_displacement_max=float(np.max(formal_disp_arr)),
        bias_track_displacement_mean=float(np.mean(bias_disp_arr)),
        bias_track_displacement_max=float(np.max(bias_disp_arr)),
        pointwise_jsd_mean=float(np.mean(jsd_arr)),
        pointwise_jsd_max=float(np.max(jsd_arr)),
        label_flip_rate=flip_rate,
        label_flips=label_flip_details,
        top_10_sensitive_orbits=top_10_orbits,
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
        constrained_null_p_value=constrained_p_val,
        rename_context_interaction_norm=interaction_test["interaction_mean_norm"],
        rename_context_interaction_p_value=interaction_test["interaction_p_value"],
        bootstrap_S_H_ci=(ci_low, ci_high),
        provenance=provenance,
    )

    # 7. Export Phase E2-A1.2a-R1.1 Manifest
    manifest_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "E2-A1.2a-R1.1",
        "git_commit_sha": get_git_commit_sha(),
        "execution_status": "COMPLETED",
        "direct_sensitivity_status": "OBSERVED",
        "global_commutator_test": global_commutator_status,
        "local_holonomy_status": local_holonomy_status,
        "affine_translation_status": affine_translation_status,
        "finding": finding,
        "summary": asdict(res),
    }

    manifest_path = os.path.join(out_dir, "phase_e2_a1_2_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n================================================================================")
    print(f"PHASE E2-A1.2a-R1.1 CONFIRMATORY LIVE AUDIT REPORT ({config.model_id}):")
    print(f"================================================================================")
    print(f"    - Adapter Mode: {provenance['adapter_mode']}")
    print(f"    - Is Loaded Live: {provenance['is_loaded']}")
    print(f"    - Model Revision: {provenance.get('resolved_model_revision')}")
    print(f"    - Orbits: {len(ds.train_orbits)} Train / {len(ds.val_orbits)} Val / {len(ds.test_orbits)} Test (Held-out Name Quartets)")
    print(f"    - Edge Predictive Skill T_a vs Identity: {skill_ta['relative_skill_vs_identity'] * 100:.2f}% (R2: {skill_ta['r2_affine']:.4f})")
    print(f"    - Edge Predictive Skill T_b vs Identity: {skill_tb['relative_skill_vs_identity'] * 100:.2f}% (R2: {skill_tb['r2_affine']:.4f})")
    print(f"    - 4-Edge Displacement Mean: {np.mean(disp_arr):.4f} (Formal: {np.mean(formal_disp_arr):.4f}, Bias: {np.mean(bias_disp_arr):.4f}, Max: {np.max(disp_arr):.4f})")
    print(f"    - Pointwise JSD Mean: {np.mean(jsd_arr):.6f} (Max: {np.max(jsd_arr):.6f})")
    print(f"    - Label Flip Rate: {flip_rate * 100:.2f}% ({len(label_flip_details)} flips observed)")
    print(f"    - Global Canonical Holonomy (S_A, S_b, S_H): ({glob_can_stats['linear_norm_S_A']:.6f}, {glob_can_stats['translation_norm_S_b']:.6f}, {glob_can_stats['homogeneous_norm_S_H']:.6f})")
    print(f"    - Local Canonical Holonomy (S_A, S_b, S_H):  ({local_can_stats['linear_norm_S_A']:.6f}, {local_can_stats['translation_norm_S_b']:.6f}, {local_can_stats['homogeneous_norm_S_H']:.6f})")
    print(f"    - Constrained Commuting-Null Bootstrap p-value: {constrained_p_val:.4f}")
    print(f"    - Rename Context Interaction p-value: {interaction_test['interaction_p_value']:.4f}")
    print(f"    - Finding: {finding}")
    print(f"================================================================================")
    print(f"Manifest exported to: {manifest_path}")

    return res


if __name__ == "__main__":
    run_e003_live_roberta_audit()
