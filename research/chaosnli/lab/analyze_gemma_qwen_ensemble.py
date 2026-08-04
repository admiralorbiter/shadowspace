"""E004 Gemma 3 & Qwen 2.5 Equal-Weight Ensemble Coalition & Edge Recovery Analysis.

Evaluates the equal-weight ensemble distribution P_ensemble = 0.5 * P_Gemma + 0.5 * P_Qwen
against human posterior support S_human (k=10, Q_HH = 0.26338) on the 600-item ChaosNLI pilot.

Computes:
  1. Pointwise NLL, JSD (nats & bits), Brier score for Raw (T=1) and Calibrated ensemble.
  2. Relational recovery Q_support, Q_null, R_norm (%), effective bits b, and K_eff prototypes.
  3. 30-stratum focal-row bootstrap 95% CIs.
  4. Detailed Graph Edge Decomposition:
     - Human-supported edges (S_ij > tau_human) recovered by Gemma alone, Qwen alone, both, and ensemble.
     - False bridges (S_ij < 0.05) introduced by each model and ensemble.
     - Tests whether ensemble relational recovery exceeds Qwen alone (R_ensemble > R_Qwen).
"""

from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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

def compute_e007_block_density_null(W_model: np.ndarray, S_human: np.ndarray, ds_ids: np.ndarray, k: int = 10) -> float:
    N = len(ds_ids)
    blocks = [0, 1]  # 0: MNLI, 1: SNLI
    block_masks = [ds_ids == b for b in blocks]
    block_sizes = [int(np.sum(m)) for m in block_masks]

    q_null = 0.0
    for a in range(2):
        for b in range(2):
            mask_a = block_masks[a]
            mask_b = block_masks[b]

            W_sub = W_model[mask_a][:, mask_b].copy()
            S_sub = S_human[mask_a][:, mask_b].copy()

            if a == b:
                np.fill_diagonal(W_sub, 0.0)
                np.fill_diagonal(S_sub, 0.0)
                n_pairs = block_sizes[a] * (block_sizes[a] - 1)
            else:
                n_pairs = block_sizes[a] * block_sizes[b]

            w_ab = (np.sum(W_sub) / float(n_pairs)) if n_pairs > 0 else 0.0
            s_sum_ab = np.sum(S_sub)

            q_null += w_ab * s_sum_ab

    return float(q_null / (N * float(k)))

def compute_jsd_nats(P: np.ndarray, Q: np.ndarray) -> float:
    eps = 1e-12
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    M = 0.5 * (P + Q)
    kl_pm = np.sum(P * np.log(P / M), axis=1)
    kl_qm = np.sum(Q * np.log(Q / M), axis=1)
    return float(np.mean(0.5 * kl_pm + 0.5 * kl_qm))

def main():
    manifest_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "manifests" / "pilot_600.jsonl"
    supp_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "pilot_support" / "S_hellinger_k010_pilot.bin"
    
    gemma_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "raw_responses" / "pilot600_gemma3-12b_v2_abc_t10_lpe.jsonl"
    qwen_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "raw_responses" / "pilot600_qwen2.5-14b_v2_abc_t10_lpe.jsonl"

    items = [json.loads(line) for line in open(manifest_path, "r", encoding="utf-8") if line.strip()]
    N = len(items)
    
    human_p = np.array([
        [it["human_p_entailment"], it["human_p_neutral"], it["human_p_contradiction"]]
        for it in items
    ], dtype=np.float64)
    ds_ids = np.array([0 if it.get("source_dataset", "chaosnli_mnli") == "chaosnli_mnli" else 1 for it in items])

    S_human = np.frombuffer(supp_path.read_bytes(), dtype=np.float32).reshape(N, N).astype(np.float64)

    e008_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E008_pilot_600_curve.json"
    with open(e008_path, "r", encoding="utf-8") as f:
        e008_data = json.load(f)
    q_hh_k10 = e008_data.get("q_hh_relational", 0.26338)

    # Re-run pipeline for Gemma 3 and Qwen 2.5 to get P_model for both
    from analyze_llm_lpe import extract_lpe_logits_and_probs, run_e004_pipeline
    
    logits_gemma, perm_probs_gemma, _, _ = extract_lpe_logits_and_probs(gemma_path, items)
    logits_qwen, perm_probs_qwen, _, _ = extract_lpe_logits_and_probs(qwen_path, items)

    gemma_res = run_e004_pipeline(items, logits_gemma, perm_probs_gemma, S_human, q_hh_k10, e008_data)
    qwen_res = run_e004_pipeline(items, logits_qwen, perm_probs_qwen, S_human, q_hh_k10, e008_data)

    P_gemma_raw = np.mean(perm_probs_gemma, axis=1)
    P_qwen_raw = np.mean(perm_probs_qwen, axis=1)

    # 1. Equal-Weight Ensemble (Raw T=1.0)
    P_ens_raw = 0.5 * (P_gemma_raw + P_qwen_raw)
    P_ens_raw = np.clip(P_ens_raw, 1e-12, 1.0)
    P_ens_raw /= np.sum(P_ens_raw, axis=1, keepdims=True)

    eps = 1e-12
    nll_ens_raw = float(-np.mean(np.sum(human_p * np.log(np.clip(P_ens_raw, eps, 1.0)), axis=1)))
    brier_ens_raw = float(np.mean(np.sum((P_ens_raw - human_p) ** 2, axis=1)))
    jsd_ens_raw_nats = compute_jsd_nats(P_ens_raw, human_p)
    jsd_ens_raw_bits = float(jsd_ens_raw_nats / math.log(2.0))

    D_ens_raw = distance_hellinger_matrix(P_ens_raw, P_ens_raw)
    W_ens_raw = compute_topk_weight_matrix(D_ens_raw, k=10)
    q_rows_ens_raw = np.sum(W_ens_raw * S_human, axis=1) / 10.0
    q_supp_ens_raw = float(np.mean(q_rows_ens_raw))
    q_null_ens_raw = compute_e007_block_density_null(W_ens_raw, S_human, ds_ids, k=10)
    r_norm_ens_raw = float((q_supp_ens_raw - q_null_ens_raw) / (q_hh_k10 - q_null_ens_raw) * 100.0)

    from sprint1_rate_distortion_and_summary import interpolate_log_linear_bits
    k_eff_ens_raw, b_bits_ens_raw = interpolate_log_linear_bits(r_norm_ens_raw, e008_data["prototype_ladder"])

    # 2. Equal-Weight Ensemble (Calibrated)
    # Average calibrated Gemma and calibrated Qwen probabilities
    # Note: Using individual model calibrated probabilities
    # We obtain calibrated probs for Gemma and Qwen across folds
    from analyze_llm_lpe import compute_calibrated_probs_for_items
    
    # 30-stratum fold assignments
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

    P_ens_cal = 0.5 * (P_gemma_cal + P_qwen_cal)
    P_ens_cal = np.clip(P_ens_cal, 1e-12, 1.0)
    P_ens_cal /= np.sum(P_ens_cal, axis=1, keepdims=True)

    nll_ens_cal = float(-np.mean(np.sum(human_p * np.log(np.clip(P_ens_cal, eps, 1.0)), axis=1)))
    brier_ens_cal = float(np.mean(np.sum((P_ens_cal - human_p) ** 2, axis=1)))
    jsd_ens_cal_nats = compute_jsd_nats(P_ens_cal, human_p)
    jsd_ens_cal_bits = float(jsd_ens_cal_nats / math.log(2.0))

    D_ens_cal = distance_hellinger_matrix(P_ens_cal, P_ens_cal)
    W_ens_cal = compute_topk_weight_matrix(D_ens_cal, k=10)
    q_rows_ens_cal = np.sum(W_ens_cal * S_human, axis=1) / 10.0
    q_supp_ens_cal = float(np.mean(q_rows_ens_cal))
    q_null_ens_cal = compute_e007_block_density_null(W_ens_cal, S_human, ds_ids, k=10)
    r_norm_ens_cal = float((q_supp_ens_cal - q_null_ens_cal) / (q_hh_k10 - q_null_ens_cal) * 100.0)

    k_eff_ens_cal, b_bits_ens_cal = interpolate_log_linear_bits(r_norm_ens_cal, e008_data["prototype_ladder"])

    # 3. 30-Stratum Focal-Row Bootstrap CIs for Ensemble & Contrasts
    rng = np.random.default_rng(20260803)
    strata_indices = {s: np.where(np.array([it.get("stratum_key", f"{it.get('source_dataset', 'chaosnli_mnli')}_{it.get('human_majority_label', 'e')}") for it in items]) == s)[0] for s in strata_map.keys()}

    boot_r_ens_raw = []
    boot_r_ens_cal = []
    boot_ens_minus_qwen_raw = []
    boot_ens_minus_qwen_cal = []

    q_rows_gemma_raw = gemma_res["q_rows_raw"]
    q_rows_qwen_raw = qwen_res["q_rows_raw"]
    q_rows_qwen_cal = qwen_res["q_rows_cal"]
    null_qwen_cal = qwen_res["null_by_item_cal"]

    for _ in range(1000):
        boot_idx_list = []
        for s, s_idx in strata_indices.items():
            sampled_s = rng.choice(s_idx, size=len(s_idx), replace=True)
            boot_idx_list.extend(sampled_s)
        idx_boot = np.array(boot_idx_list)

        # Ensemble Raw R
        q_s_ens_raw_b = float(np.mean(q_rows_ens_raw[idx_boot]))
        r_ens_raw_b = float((q_s_ens_raw_b - q_null_ens_raw) / (q_hh_k10 - q_null_ens_raw) * 100.0)
        boot_r_ens_raw.append(r_ens_raw_b)

        # Qwen Raw R
        q_s_qwen_raw_b = float(np.mean(q_rows_qwen_raw[idx_boot]))
        r_qwen_raw_b = float((q_s_qwen_raw_b - qwen_res["q_null_raw"]) / (q_hh_k10 - qwen_res["q_null_raw"]) * 100.0)
        boot_ens_minus_qwen_raw.append(r_ens_raw_b - r_qwen_raw_b)

        # Ensemble Cal R
        q_s_ens_cal_b = float(np.mean(q_rows_ens_cal[idx_boot]))
        r_ens_cal_b = float((q_s_ens_cal_b - q_null_ens_cal) / (q_hh_k10 - q_null_ens_cal) * 100.0)
        boot_r_ens_cal.append(r_ens_cal_b)

        # Qwen Cal R
        q_s_qwen_cal_b = float(np.mean(q_rows_qwen_cal[idx_boot]))
        null_qwen_b = float(np.mean(null_qwen_cal[idx_boot]))
        r_qwen_cal_b = float((q_s_qwen_cal_b - null_qwen_b) / (q_hh_k10 - null_qwen_b) * 100.0)
        boot_ens_minus_qwen_cal.append(r_ens_cal_b - r_qwen_cal_b)

    ci_ens_raw = [float(np.percentile(boot_r_ens_raw, 2.5)), float(np.percentile(boot_r_ens_raw, 97.5))]
    ci_ens_cal = [float(np.percentile(boot_r_ens_cal, 2.5)), float(np.percentile(boot_r_ens_cal, 97.5))]
    ci_ens_minus_qwen_raw = [float(np.percentile(boot_ens_minus_qwen_raw, 2.5)), float(np.percentile(boot_ens_minus_qwen_raw, 97.5))]
    ci_ens_minus_qwen_cal = [float(np.percentile(boot_ens_minus_qwen_cal, 2.5)), float(np.percentile(boot_ens_minus_qwen_cal, 97.5))]

    # 4. Graph Edge Recovery & Decomposition
    # Define human-supported edges threshold (top 10% or median positive S_ij)
    S_nodiag = S_human.copy()
    np.fill_diagonal(S_nodiag, 0.0)
    tau_human = float(np.percentile(S_nodiag[S_nodiag > 0], 90))  # High human support threshold

    D_gemma_raw = distance_hellinger_matrix(P_gemma_raw, P_gemma_raw)
    W_gemma_raw = compute_topk_weight_matrix(D_gemma_raw, k=10)

    D_qwen_raw = distance_hellinger_matrix(P_qwen_raw, P_qwen_raw)
    W_qwen_raw = compute_topk_weight_matrix(D_qwen_raw, k=10)

    # Binary top-10 edge masks (W > 0)
    edge_gemma = (W_gemma_raw > 0)
    edge_qwen = (W_qwen_raw > 0)
    edge_ens = (W_ens_raw > 0)
    edge_human = (S_nodiag >= tau_human)

    # Edge recovery breakdown
    gemma_supported = edge_gemma & edge_human
    qwen_supported = edge_qwen & edge_human
    ens_supported = edge_ens & edge_human

    n_human_supported_edges = int(np.sum(edge_human))
    n_gemma_supported = int(np.sum(gemma_supported))
    n_qwen_supported = int(np.sum(qwen_supported))
    n_ens_supported = int(np.sum(ens_supported))

    n_gemma_only_supported = int(np.sum(gemma_supported & ~qwen_supported))
    n_qwen_only_supported = int(np.sum(qwen_supported & ~gemma_supported))
    n_both_supported = int(np.sum(gemma_supported & qwen_supported))
    n_ens_only_supported = int(np.sum(ens_supported & ~(gemma_supported | qwen_supported)))

    # False bridges (S_ij < 0.05)
    false_bridge_mask = (S_nodiag < 0.05)
    n_fb_gemma = int(np.sum(edge_gemma & false_bridge_mask))
    n_fb_qwen = int(np.sum(edge_qwen & false_bridge_mask))
    n_fb_ens = int(np.sum(edge_ens & false_bridge_mask))
    n_fb_gemma_only = int(np.sum(edge_gemma & false_bridge_mask & ~edge_qwen))
    n_fb_qwen_only = int(np.sum(edge_qwen & false_bridge_mask & ~edge_gemma))

    summary = {
        "title": "Gemma 3 & Qwen 2.5 Equal-Weight Ensemble Coalition & Edge Recovery",
        "num_items": N,
        "human_relational_target": {
            "q_hh_relational": q_hh_k10,
            "tau_human_edge_threshold": tau_human,
            "total_human_supported_edges": n_human_supported_edges
        },
        "ensemble_raw_metrics": {
            "nll_nats": nll_ens_raw,
            "brier": brier_ens_raw,
            "jsd_nats": jsd_ens_raw_nats,
            "jsd_bits": jsd_ens_raw_bits,
            "q_support": q_supp_ens_raw,
            "q_null": q_null_ens_raw,
            "r_norm_pct": r_norm_ens_raw,
            "r_norm_95ci": ci_ens_raw,
            "effective_bits": b_bits_ens_raw,
            "k_eff_prototypes": k_eff_ens_raw
        },
        "ensemble_calibrated_metrics": {
            "nll_nats": nll_ens_cal,
            "brier": brier_ens_cal,
            "jsd_nats": jsd_ens_cal_nats,
            "jsd_bits": jsd_ens_cal_bits,
            "q_support": q_supp_ens_cal,
            "q_null": q_null_ens_cal,
            "r_norm_pct": r_norm_ens_cal,
            "r_norm_95ci": ci_ens_cal,
            "effective_bits": b_bits_ens_cal,
            "k_eff_prototypes": k_eff_ens_cal
        },
        "ensemble_vs_single_model_contrasts": {
            "ensemble_minus_qwen_raw_pct": r_norm_ens_raw - qwen_res["r_norm_pct_raw"],
            "ensemble_minus_qwen_raw_95ci": ci_ens_minus_qwen_raw,
            "ensemble_minus_qwen_cal_pct": r_norm_ens_cal - qwen_res["r_norm_pct_calibrated"],
            "ensemble_minus_qwen_cal_95ci": ci_ens_minus_qwen_cal,
            "ensemble_exceeds_qwen_raw": r_norm_ens_raw > qwen_res["r_norm_pct_raw"],
            "ensemble_exceeds_qwen_cal": r_norm_ens_cal > qwen_res["r_norm_pct_calibrated"]
        },
        "graph_edge_decomposition": {
            "gemma_supported_edges": n_gemma_supported,
            "qwen_supported_edges": n_qwen_supported,
            "ensemble_supported_edges": n_ens_supported,
            "gemma_only_supported_edges": n_gemma_only_supported,
            "qwen_only_supported_edges": n_qwen_only_supported,
            "both_models_supported_edges": n_both_supported,
            "ensemble_only_supported_edges": n_ens_only_supported,
            "gemma_false_bridges": n_fb_gemma,
            "qwen_false_bridges": n_fb_qwen,
            "ensemble_false_bridges": n_fb_ens,
            "gemma_only_false_bridges": n_fb_gemma_only,
            "qwen_only_false_bridges": n_fb_qwen_only
        }
    }

    out_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E004_gemma_qwen_ensemble_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("============================================================")
    print("  GEMMA 3 & QWEN 2.5 EQUAL-WEIGHT ENSEMBLE COALITION RESULTS")
    print("============================================================")
    print(f"  Ensemble Raw NLL:             {nll_ens_raw:.4f} nats  (Gemma: 3.8277, Qwen: 5.2986)")
    print(f"  Ensemble Calibrated NLL:      {nll_ens_cal:.4f} nats  (Gemma: 0.9308, Qwen: 0.8837)")
    print(f"  Ensemble Raw R_norm:          {r_norm_ens_raw:.2f}%  (95% CI: [{ci_ens_raw[0]:.2f}%, {ci_ens_raw[1]:.2f}%])")
    print(f"                                (Gemma: 9.72%, Qwen: 11.85%)")
    print(f"  Ensemble Calibrated R_norm:   {r_norm_ens_cal:.2f}%  (95% CI: [{ci_ens_cal[0]:.2f}%, {ci_ens_cal[1]:.2f}%])")
    print(f"                                (Gemma: 9.76%, Qwen: 14.86%)")
    print(f"  Ensemble Resolution (Raw):    {b_bits_ens_raw:.3f} bits (K_eff = {k_eff_ens_raw:.2f})")
    print(f"  Ensemble Resolution (Cal):    {b_bits_ens_cal:.3f} bits (K_eff = {k_eff_ens_cal:.2f})")
    print("------------------------------------------------------------")
    print("  ENSEMBLE VS SINGLE MODEL CONTRASTS:")
    print(f"    Raw Ens - Qwen:             {r_norm_ens_raw - qwen_res['r_norm_pct_raw']:+.2f}% (95% CI: [{ci_ens_minus_qwen_raw[0]:+.2f}%, {ci_ens_minus_qwen_raw[1]:+.2f}%])")
    print(f"    Cal Ens - Qwen:             {r_norm_ens_cal - qwen_res['r_norm_pct_calibrated']:+.2f}% (95% CI: [{ci_ens_minus_qwen_cal[0]:+.2f}%, {ci_ens_minus_qwen_cal[1]:+.2f}%])")
    print(f"    Ensemble Exceeds Qwen Alone (Raw/Cal): {r_norm_ens_raw > qwen_res['r_norm_pct_raw']} / {r_norm_ens_cal > qwen_res['r_norm_pct_calibrated']}")
    print("------------------------------------------------------------")
    print("  GRAPH EDGE DECOMPOSITION:")
    print(f"    Human-Supported Edges Total: {n_human_supported_edges}")
    print(f"    Recovered by Gemma Alone:   {n_gemma_only_supported}")
    print(f"    Recovered by Qwen Alone:    {n_qwen_only_supported}")
    print(f"    Recovered by Both:          {n_both_supported}")
    print(f"    Recovered by Ensemble:      {n_ens_supported} ({n_ens_only_supported} ensemble-unique)")
    print(f"    False Bridges (Gemma/Qwen/Ens): {n_fb_gemma} / {n_fb_qwen} / {n_fb_ens}")
    print("============================================================")
    print(f"Exported ensemble summary to {out_path}")

if __name__ == "__main__":
    main()
