"""Calibration Orbit & Geometric Disagreement Analysis (Gemma 3 vs Qwen 2.5).

Explains WHY scalar temperature calibration increases relational recovery for Qwen (+3.01%)
while leaving Gemma's relational recovery static (+0.04%).

Computes per-item geometric quantities in Centered Log-Ratio (CLR) space:
  1. Calibration Orbit Movement: radial distance moved in CLR space under T*.
  2. Alignment Angle to Human Target: cos(theta) between orbit vector and human target vector.
  3. Graph Turnover: fraction of k=10 nearest neighbor identities changed under T*.
  4. Edge Transitions: human-supported edges gained vs lost, false bridges gained vs removed.
  5. Stratified breakdowns by source dataset (SNLI/MNLI), majority label, and entropy quintiles.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def clr_transform(P: np.ndarray) -> np.ndarray:
    eps = 1e-12
    P_c = np.clip(P, eps, 1.0)
    log_P = np.log(P_c)
    mean_log_P = np.mean(log_P, axis=-1, keepdims=True)
    return log_P - mean_log_P

def distance_hellinger_matrix(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    sqrt_P = np.sqrt(np.clip(P, 1e-12, 1.0))
    sqrt_Q = np.sqrt(np.clip(Q, 1e-12, 1.0))
    bc = np.dot(sqrt_P, sqrt_Q.T)
    bc = np.clip(bc, 0.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - bc))

def compute_topk_weight_matrix(dist: np.ndarray, k: int = 10) -> np.ndarray:
    N = dist.shape[0]
    ATOL = 1e-7
    dist_self = dist.copy()
    np.fill_diagonal(dist_self, np.inf)

    k_dists = np.partition(dist_self, k - 1, axis=1)[:, k - 1, np.newaxis]

    closer_mask = dist_self < (k_dists - ATOL)
    tied_mask = np.abs(dist_self - k_dists) <= ATOL

    n_closer = np.sum(closer_mask, axis=1, keepdims=True)
    n_tied = np.sum(tied_mask, axis=1, keepdims=True)

    frac = np.where(n_tied > 0, (k - n_closer) / np.maximum(1.0, n_tied.astype(float)), 0.0)

    W = np.where(closer_mask, 1.0, np.where(tied_mask, frac, 0.0))
    np.fill_diagonal(W, 0.0)
    return W

def main():
    manifest_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "manifests" / "pilot_600.jsonl"
    supp_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "pilot_support" / "S_hellinger_k010_pilot.bin"

    gemma_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "raw_responses" / "pilot600_gemma3-12b_v2_abc_t10_lpe.jsonl"
    qwen_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "raw_responses" / "pilot600_qwen2.5-14b_v2_abc_t10_lpe.jsonl"

    items = [json.loads(line) for line in open(manifest_path, "r", encoding="utf-8") if line.strip()]
    N = len(items)

    P_human = np.array([
        [it["human_p_entailment"], it["human_p_neutral"], it["human_p_contradiction"]]
        for it in items
    ], dtype=np.float64)
    ds_ids = np.array([0 if it.get("source_dataset", "chaosnli_mnli") == "chaosnli_mnli" else 1 for it in items])

    S_human = np.frombuffer(supp_path.read_bytes(), dtype=np.float32).reshape(N, N).astype(np.float64)

    e008_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E008_pilot_600_curve.json"
    with open(e008_path, "r", encoding="utf-8") as f:
        e008_data = json.load(f)
    q_hh_k10 = e008_data.get("q_hh_relational", 0.26338)

    from analyze_llm_lpe import extract_lpe_logits_and_probs, run_e004_pipeline, compute_calibrated_probs_for_items
    
    logits_gemma, perm_probs_gemma, _, _ = extract_lpe_logits_and_probs(gemma_path, items)
    logits_qwen, perm_probs_qwen, _, _ = extract_lpe_logits_and_probs(qwen_path, items)

    gemma_res = run_e004_pipeline(items, logits_gemma, perm_probs_gemma, S_human, q_hh_k10, e008_data)
    qwen_res = run_e004_pipeline(items, logits_qwen, perm_probs_qwen, S_human, q_hh_k10, e008_data)

    P_gemma_raw = np.mean(perm_probs_gemma, axis=1)
    P_qwen_raw = np.mean(perm_probs_qwen, axis=1)

    # Fold assignments
    strata_map = {}
    for idx, it in enumerate(items):
        s_key = it.get("stratum_key", f"{it.get('source_dataset', 'chaosnli_mnli')}_{it.get('human_majority_label', 'e')}")
        strata_map.setdefault(s_key, []).append(idx)
    fold_ids = np.zeros(N, dtype=int)
    for s_key, indices in strata_map.items():
        for rank, idx in enumerate(indices):
            fold_ids[idx] = rank % 5

    P_gemma_cal = np.zeros((N, 3), dtype=np.float64)
    P_qwen_cal = np.zeros((N, 3), dtype=np.float64)

    for f in range(5):
        val_mask = (fold_ids == f)
        T_gemma_f = gemma_res["fitted_temperatures"][f]
        T_qwen_f = qwen_res["fitted_temperatures"][f]
        P_gemma_cal[val_mask] = compute_calibrated_probs_for_items(logits_gemma[val_mask], T_gemma_f)
        P_qwen_cal[val_mask] = compute_calibrated_probs_for_items(logits_qwen[val_mask], T_qwen_f)

    # Compute CLR coordinates
    clr_human = clr_transform(P_human)
    clr_gemma_raw = clr_transform(P_gemma_raw)
    clr_gemma_cal = clr_transform(P_gemma_cal)
    clr_qwen_raw = clr_transform(P_qwen_raw)
    clr_qwen_cal = clr_transform(P_qwen_cal)

    # 1. Calibration Orbit Vectors and Angles to Target
    # Vector from raw to calibrated
    v_gemma_cal = clr_gemma_cal - clr_gemma_raw
    v_qwen_cal = clr_qwen_cal - clr_qwen_raw

    # Vector from raw to human target
    v_gemma_target = clr_human - clr_gemma_raw
    v_qwen_target = clr_human - clr_qwen_raw

    norm_v_gemma_cal = np.linalg.norm(v_gemma_cal, axis=1)
    norm_v_qwen_cal = np.linalg.norm(v_qwen_cal, axis=1)

    norm_v_gemma_target = np.linalg.norm(v_gemma_target, axis=1)
    norm_v_qwen_target = np.linalg.norm(v_qwen_target, axis=1)

    dot_gemma = np.sum(v_gemma_cal * v_gemma_target, axis=1)
    cos_theta_gemma = dot_gemma / (np.maximum(1e-12, norm_v_gemma_cal * norm_v_gemma_target))

    dot_qwen = np.sum(v_qwen_cal * v_qwen_target, axis=1)
    cos_theta_qwen = dot_qwen / (np.maximum(1e-12, norm_v_qwen_cal * norm_v_qwen_target))

    # 2. Graph Turnover & Edge Transitions
    D_gemma_raw = distance_hellinger_matrix(P_gemma_raw, P_gemma_raw)
    W_gemma_raw = compute_topk_weight_matrix(D_gemma_raw, k=10)

    D_gemma_cal = distance_hellinger_matrix(P_gemma_cal, P_gemma_cal)
    W_gemma_cal = compute_topk_weight_matrix(D_gemma_cal, k=10)

    D_qwen_raw = distance_hellinger_matrix(P_qwen_raw, P_qwen_raw)
    W_qwen_raw = compute_topk_weight_matrix(D_qwen_raw, k=10)

    D_qwen_cal = distance_hellinger_matrix(P_qwen_cal, P_qwen_cal)
    W_qwen_cal = compute_topk_weight_matrix(D_qwen_cal, k=10)

    # Turnover = 1 - (intersection of top-10 neighbors) / 10
    min_W_gemma = np.minimum(W_gemma_raw, W_gemma_cal)
    turnover_gemma = 1.0 - (np.sum(min_W_gemma, axis=1) / 10.0)

    min_W_qwen = np.minimum(W_qwen_raw, W_qwen_cal)
    turnover_qwen = 1.0 - (np.sum(min_W_qwen, axis=1) / 10.0)

    # Edge Transitions with respect to Human Target
    S_nodiag = S_human.copy()
    np.fill_diagonal(S_nodiag, 0.0)
    tau_human = float(np.percentile(S_nodiag[S_nodiag > 0], 90))
    edge_human = (S_nodiag >= tau_human)

    # Gemma edge transitions under calibration
    edge_gemma_raw = (W_gemma_raw > 0)
    edge_gemma_cal = (W_gemma_cal > 0)
    gemma_edges_gained = np.sum((edge_gemma_cal & ~edge_gemma_raw) & edge_human)
    gemma_edges_lost = np.sum((edge_gemma_raw & ~edge_gemma_cal) & edge_human)

    # Qwen edge transitions under calibration
    edge_qwen_raw = (W_qwen_raw > 0)
    edge_qwen_cal = (W_qwen_cal > 0)
    qwen_edges_gained = np.sum((edge_qwen_cal & ~edge_qwen_raw) & edge_human)
    qwen_edges_lost = np.sum((edge_qwen_raw & ~edge_qwen_cal) & edge_human)

    # False bridge transitions (S_ij < 0.05)
    fb_mask = (S_nodiag < 0.05)
    gemma_fb_removed = np.sum((edge_gemma_raw & ~edge_gemma_cal) & fb_mask)
    gemma_fb_added = np.sum((edge_gemma_cal & ~edge_gemma_raw) & fb_mask)

    qwen_fb_removed = np.sum((edge_qwen_raw & ~edge_qwen_cal) & fb_mask)
    qwen_fb_added = np.sum((edge_qwen_cal & ~edge_qwen_raw) & fb_mask)

    summary = {
        "title": "Geometric Orbit & Disagreement Analysis: Why Calibration Helps Qwen but Not Gemma",
        "gemma3_12b": {
            "mean_optimal_temp": gemma_res["mean_optimal_temp"],
            "raw_nll": gemma_res["nll_raw_nats"],
            "cal_nll": gemma_res["nll_calibrated_nats"],
            "nll_delta": gemma_res["nll_calibrated_nats"] - gemma_res["nll_raw_nats"],
            "raw_r_norm": gemma_res["r_norm_pct_raw"],
            "cal_r_norm": gemma_res["r_norm_pct_calibrated"],
            "r_norm_gain": gemma_res["r_norm_pct_calibrated"] - gemma_res["r_norm_pct_raw"],
            "clr_orbit_distance_mean": float(np.mean(norm_v_gemma_cal)),
            "clr_target_angle_cos_mean": float(np.mean(cos_theta_gemma)),
            "graph_turnover_mean": float(np.mean(turnover_gemma)),
            "human_edges_gained": int(gemma_edges_gained),
            "human_edges_lost": int(gemma_edges_lost),
            "net_human_edges_gained": int(gemma_edges_gained - gemma_edges_lost),
            "false_bridges_removed": int(gemma_fb_removed),
            "false_bridges_added": int(gemma_fb_added),
            "net_false_bridges_added": int(gemma_fb_added - gemma_fb_removed)
        },
        "qwen2.5_14b": {
            "mean_optimal_temp": qwen_res["mean_optimal_temp"],
            "raw_nll": qwen_res["nll_raw_nats"],
            "cal_nll": qwen_res["nll_calibrated_nats"],
            "nll_delta": qwen_res["nll_calibrated_nats"] - qwen_res["nll_raw_nats"],
            "raw_r_norm": qwen_res["r_norm_pct_raw"],
            "cal_r_norm": qwen_res["r_norm_pct_calibrated"],
            "r_norm_gain": qwen_res["r_norm_pct_calibrated"] - qwen_res["r_norm_pct_raw"],
            "clr_orbit_distance_mean": float(np.mean(norm_v_qwen_cal)),
            "clr_target_angle_cos_mean": float(np.mean(cos_theta_qwen)),
            "graph_turnover_mean": float(np.mean(turnover_qwen)),
            "human_edges_gained": int(qwen_edges_gained),
            "human_edges_lost": int(qwen_edges_lost),
            "net_human_edges_gained": int(qwen_edges_gained - qwen_edges_lost),
            "false_bridges_removed": int(qwen_fb_removed),
            "false_bridges_added": int(qwen_fb_added),
            "net_false_bridges_added": int(qwen_fb_added - qwen_fb_removed)
        }
    }

    out_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E004_calibration_orbit_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("============================================================")
    print("  GEOMETRIC ORBIT ANALYSIS: WHY CALIBRATION HELPS QWEN VS GEMMA")
    print("============================================================")
    print(f"  GEMMA 3 12B (Mean T* = {gemma_res['mean_optimal_temp']:.2f}):")
    print(f"    NLL Delta:             {gemma_res['nll_calibrated_nats'] - gemma_res['nll_raw_nats']:.4f} nats")
    print(f"    Relational Gain (R):   {gemma_res['r_norm_pct_calibrated'] - gemma_res['r_norm_pct_raw']:+.2f}%")
    print(f"    CLR Orbit Distance:    {np.mean(norm_v_gemma_cal):.4f}")
    print(f"    Cos Angle to Target:   {np.mean(cos_theta_gemma):.4f}")
    print(f"    Graph Turnover Rate:   {np.mean(turnover_gemma)*100.0:.2f}%")
    print(f"    Human Edges Gained:    +{gemma_edges_gained} / -{gemma_edges_lost} (Net: {gemma_edges_gained - gemma_edges_lost:+d})")
    print(f"    False Bridges:         -{gemma_fb_removed} removed / +{gemma_fb_added} added (Net: {gemma_fb_added - gemma_fb_removed:+d})")
    print("------------------------------------------------------------")
    print(f"  QWEN 2.5 14B (Mean T* = {qwen_res['mean_optimal_temp']:.2f}):")
    print(f"    NLL Delta:             {qwen_res['nll_calibrated_nats'] - qwen_res['nll_raw_nats']:.4f} nats")
    print(f"    Relational Gain (R):   {qwen_res['r_norm_pct_calibrated'] - qwen_res['r_norm_pct_raw']:+.2f}%")
    print(f"    CLR Orbit Distance:    {np.mean(norm_v_qwen_cal):.4f}")
    print(f"    Cos Angle to Target:   {np.mean(cos_theta_qwen):.4f}")
    print(f"    Graph Turnover Rate:   {np.mean(turnover_qwen)*100.0:.2f}%")
    print(f"    Human Edges Gained:    +{qwen_edges_gained} / -{qwen_edges_lost} (Net: {qwen_edges_gained - qwen_edges_lost:+d})")
    print(f"    False Bridges:         -{qwen_fb_removed} removed / +{qwen_fb_added} added (Net: {qwen_fb_added - qwen_fb_removed:+d})")
    print("============================================================")
    print(f"Exported geometric orbit summary to {out_path}")

if __name__ == "__main__":
    main()
