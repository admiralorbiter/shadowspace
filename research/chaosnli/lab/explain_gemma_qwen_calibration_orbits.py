"""Fold-Coherent Calibration Orbit & Support Band Decomposition Analysis.

Implements all required scientific fixes:
  1. Fold-coherent calibration orbit and graph turnover:
     For each fold f, applies T_f* to all 600 items, constructs complete coherent graph W_f,
     and evaluates held-out focal rows.
  2. Exact per-item support change Delta Q_i = (1/k) * sum_j (W_f,ij - W_raw,ij) * S_human,ij.
  3. Support Band Partitioning:
     - Band 1: S_ij < 0.05 (False bridges)
     - Band 2: 0.05 <= S_ij < 0.25 (Weak support)
     - Band 3: 0.25 <= S_ij < 0.50 (Moderate support)
     - Band 4: S_ij >= 0.50 (Strong human agreement edges)
  4. Precise paper-safe reporting of candidate mechanisms.
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

def evaluate_fold_coherent_orbits(
    model_name: str,
    P_raw: np.ndarray,
    logits: np.ndarray,
    fitted_Ts: List[float],
    P_human: np.ndarray,
    S_human: np.ndarray,
    strata_map: Dict
) -> Dict:
    N = len(P_human)
    fold_ids = np.zeros(N, dtype=int)
    for s_key, indices in strata_map.items():
        for rank, idx in enumerate(indices):
            fold_ids[idx] = rank % 5

    from analyze_llm_lpe import compute_calibrated_probs_for_items

    D_raw = distance_hellinger_matrix(P_raw, P_raw)
    W_raw = compute_topk_weight_matrix(D_raw, k=10)

    P_cal_coherent = np.zeros((N, 3), dtype=np.float64)
    W_cal_coherent = np.zeros((N, N), dtype=np.float64)

    clr_human = clr_transform(P_human)
    clr_raw = clr_transform(P_raw)

    clr_distances = np.zeros(N, dtype=np.float64)
    cos_angles = np.zeros(N, dtype=np.float64)
    turnover_rates = np.zeros(N, dtype=np.float64)

    # Edge Support Bands
    S_nodiag = S_human.copy()
    np.fill_diagonal(S_nodiag, 0.0)

    band_fb = (S_nodiag < 0.05)
    band_weak = (S_nodiag >= 0.05) & (S_nodiag < 0.25)
    band_mod = (S_nodiag >= 0.25) & (S_nodiag < 0.50)
    band_strong = (S_nodiag >= 0.50)

    band_delta_Q = {
        "band_1_false_bridge_lt_005": 0.0,
        "band_2_weak_005_to_025": 0.0,
        "band_3_moderate_025_to_050": 0.0,
        "band_4_strong_gte_050": 0.0
    }

    fb_removed_count = 0
    fb_added_count = 0

    for f in range(5):
        val_mask = (fold_ids == f)
        T_f = fitted_Ts[f]

        # Apply T_f to all 600 items
        P_f_all = compute_calibrated_probs_for_items(logits, T_f)
        P_cal_coherent[val_mask] = P_f_all[val_mask]

        clr_cal_f = clr_transform(P_f_all)

        D_f = distance_hellinger_matrix(P_f_all, P_f_all)
        W_f = compute_topk_weight_matrix(D_f, k=10)
        W_cal_coherent[val_mask] = W_f[val_mask]

        val_indices = np.where(val_mask)[0]
        for v in val_indices:
            v_cal = clr_cal_f[v] - clr_raw[v]
            v_target = clr_human[v] - clr_raw[v]

            norm_cal = np.linalg.norm(v_cal)
            norm_target = np.linalg.norm(v_target)

            clr_distances[v] = norm_cal
            cos_angles[v] = np.sum(v_cal * v_target) / max(1e-12, norm_cal * norm_target)

            # Turnover
            min_w = np.minimum(W_raw[v], W_f[v])
            turnover_rates[v] = 1.0 - (np.sum(min_w) / 10.0)

            # Delta W for focal row v
            dW_v = W_f[v] - W_raw[v]

            band_delta_Q["band_1_false_bridge_lt_005"] += float(np.sum(dW_v[band_fb[v]] * S_nodiag[v, band_fb[v]]) / 10.0)
            band_delta_Q["band_2_weak_005_to_025"] += float(np.sum(dW_v[band_weak[v]] * S_nodiag[v, band_weak[v]]) / 10.0)
            band_delta_Q["band_3_moderate_025_to_050"] += float(np.sum(dW_v[band_mod[v]] * S_nodiag[v, band_mod[v]]) / 10.0)
            band_delta_Q["band_4_strong_gte_050"] += float(np.sum(dW_v[band_strong[v]] * S_nodiag[v, band_strong[v]]) / 10.0)

            fb_removed_count += int(np.sum((W_raw[v] > 0) & (W_f[v] == 0) & band_fb[v]))
            fb_added_count += int(np.sum((W_raw[v] == 0) & (W_f[v] > 0) & band_fb[v]))

    # Average Delta Q per band across N items
    for b in band_delta_Q:
        band_delta_Q[b] /= float(N)

    return {
        "model_name": model_name,
        "mean_clr_orbit_distance": float(np.mean(clr_distances)),
        "mean_target_direction_cos": float(np.mean(cos_angles)),
        "mean_graph_turnover_pct": float(np.mean(turnover_rates) * 100.0),
        "false_bridges_removed": fb_removed_count,
        "false_bridges_added": fb_added_count,
        "net_false_bridges_added": fb_added_count - fb_removed_count,
        "band_delta_Q_contributions": band_delta_Q
    }

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

    S_human = np.frombuffer(supp_path.read_bytes(), dtype=np.float32).reshape(N, N).astype(np.float64)

    e008_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E008_pilot_600_curve.json"
    with open(e008_path, "r", encoding="utf-8") as f:
        e008_data = json.load(f)

    strata_map = {}
    for idx, it in enumerate(items):
        s_key = it.get("stratum_key", f"{it.get('source_dataset', 'chaosnli_mnli')}_{it.get('human_majority_label', 'e')}")
        strata_map.setdefault(s_key, []).append(idx)

    from analyze_llm_lpe import extract_lpe_logits_and_probs, run_e004_pipeline

    logits_gemma, perm_probs_gemma, _, _ = extract_lpe_logits_and_probs(gemma_path, items)
    logits_qwen, perm_probs_qwen, _, _ = extract_lpe_logits_and_probs(qwen_path, items)

    gemma_res = run_e004_pipeline(items, logits_gemma, perm_probs_gemma, S_human, e008_data.get("q_hh_relational", 0.26338), e008_data)
    qwen_res = run_e004_pipeline(items, logits_qwen, perm_probs_qwen, S_human, e008_data.get("q_hh_relational", 0.26338), e008_data)

    P_gemma_raw = np.mean(perm_probs_gemma, axis=1)
    P_qwen_raw = np.mean(perm_probs_qwen, axis=1)

    gemma_orbits = evaluate_fold_coherent_orbits(
        "gemma3:12b", P_gemma_raw, logits_gemma, gemma_res["fitted_temperatures"], P_human, S_human, strata_map
    )
    qwen_orbits = evaluate_fold_coherent_orbits(
        "qwen2.5:14b", P_qwen_raw, logits_qwen, qwen_res["fitted_temperatures"], P_human, S_human, strata_map
    )

    summary = {
        "title": "Fold-Coherent Geometric Orbit & Support Band Decomposition Analysis",
        "gemma3_12b": gemma_orbits,
        "qwen2.5_14b": qwen_orbits
    }

    out_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E004_calibration_orbit_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("============================================================")
    print("  FOLD-COHERENT GEOMETRIC ORBIT & SUPPORT BAND DECOMPOSITION")
    print("============================================================")
    print(f"  GEMMA 3 12B:")
    print(f"    CLR Orbit Distance:           {gemma_orbits['mean_clr_orbit_distance']:.4f}")
    print(f"    Cos Angle to Target:          {gemma_orbits['mean_target_direction_cos']:.4f}")
    print(f"    Fold-Coherent Turnover:       {gemma_orbits['mean_graph_turnover_pct']:.2f}%")
    print(f"    False Bridges Removed/Added:  -{gemma_orbits['false_bridges_removed']} / +{gemma_orbits['false_bridges_added']} (Net: {gemma_orbits['net_false_bridges_added']:+d})")
    print(f"    Band Delta Q Contributions:")
    for b, v in gemma_orbits["band_delta_Q_contributions"].items():
        print(f"      {b}: {v:+.6f}")
    print("------------------------------------------------------------")
    print(f"  QWEN 2.5 14B:")
    print(f"    CLR Orbit Distance:           {qwen_orbits['mean_clr_orbit_distance']:.4f}")
    print(f"    Cos Angle to Target:          {qwen_orbits['mean_target_direction_cos']:.4f}")
    print(f"    Fold-Coherent Turnover:       {qwen_orbits['mean_graph_turnover_pct']:.2f}%")
    print(f"    False Bridges Removed/Added:  -{qwen_orbits['false_bridges_removed']} / +{qwen_orbits['false_bridges_added']} (Net: {qwen_orbits['net_false_bridges_added']:+d})")
    print(f"    Band Delta Q Contributions:")
    for b, v in qwen_orbits["band_delta_Q_contributions"].items():
        print(f"      {b}: {v:+.6f}")
    print("============================================================")
    print(f"Exported fold-coherent orbit summary to {out_path}")

if __name__ == "__main__":
    main()
