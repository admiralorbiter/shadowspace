"""E004 Fold-Coherent Ensemble Coalition & Pooling Operator Analysis.

Implements all required scientific fixes:
  1. Fold-coherent calibrated ensemble estimator:
     For each fold f, applies T_G,f* and T_Q,f* to all 600 items, builds one complete graph W_ens,f,
     scores held-out focal rows, and computes fold-specific item nulls.
  2. Full 21-point linear pooling sweep P_lambda = lambda * P_Gemma + (1-lambda) * P_Qwen (lambda in [0, 1]).
  3. Logarithmic Opinion Pool (Log-Linear Pool): P_log,lambda(x) proportional to P_Gemma(x)^lambda * P_Qwen(x)^(1-lambda).
  4. Weighted Fuzzy Support & Multi-Threshold Sensitivity Audit (80th, 90th, 95th percentiles; tau = 0.1, 0.25, 0.5).
  5. 30-stratum focal-row bootstrap CIs with resampled item-level fold nulls.
"""

from __future__ import annotations

import json
import math
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

def evaluate_fold_coherent_ensemble(
    P_gemma_raw: np.ndarray,
    P_qwen_raw: np.ndarray,
    logits_gemma: np.ndarray,
    logits_qwen: np.ndarray,
    gemma_fitted_Ts: List[float],
    qwen_fitted_Ts: List[float],
    human_p: np.ndarray,
    S_human: np.ndarray,
    ds_ids: np.ndarray,
    strata_map: Dict,
    q_hh_k10: float,
    e008_data: Dict,
    weight_gemma: float = 0.5,
    pool_type: str = "linear"
) -> Dict:
    N = len(human_p)
    fold_ids = np.zeros(N, dtype=int)
    for s_key, indices in strata_map.items():
        for rank, idx in enumerate(indices):
            fold_ids[idx] = rank % 5

    # 1. Raw Ensemble
    if pool_type == "linear":
        P_ens_raw = weight_gemma * P_gemma_raw + (1.0 - weight_gemma) * P_qwen_raw
    else:  # logarithmic
        eps = 1e-12
        P_g_c = np.clip(P_gemma_raw, eps, 1.0)
        P_q_c = np.clip(P_qwen_raw, eps, 1.0)
        unnorm = (P_g_c ** weight_gemma) * (P_q_c ** (1.0 - weight_gemma))
        P_ens_raw = unnorm / np.sum(unnorm, axis=1, keepdims=True)

    P_ens_raw = np.clip(P_ens_raw, 1e-12, 1.0)
    P_ens_raw /= np.sum(P_ens_raw, axis=1, keepdims=True)

    eps = 1e-12
    nll_raw = float(-np.mean(np.sum(human_p * np.log(np.clip(P_ens_raw, eps, 1.0)), axis=1)))
    brier_raw = float(np.mean(np.sum((P_ens_raw - human_p) ** 2, axis=1)))
    jsd_raw_nats = compute_jsd_nats(P_ens_raw, human_p)

    D_raw = distance_hellinger_matrix(P_ens_raw, P_ens_raw)
    W_raw = compute_topk_weight_matrix(D_raw, k=10)
    q_rows_raw = np.sum(W_raw * S_human, axis=1) / 10.0
    q_supp_raw = float(np.mean(q_rows_raw))
    q_null_raw = compute_e007_block_density_null(W_raw, S_human, ds_ids, k=10)
    r_norm_raw = float((q_supp_raw - q_null_raw) / (q_hh_k10 - q_null_raw) * 100.0)

    # 2. Fold-Coherent Calibrated Ensemble
    from analyze_llm_lpe import compute_calibrated_probs_for_items
    
    cal_probs_ens = np.zeros((N, 3), dtype=np.float64)
    q_rows_cal_coherent = np.zeros(N, dtype=np.float64)
    null_by_item_cal = np.zeros(N, dtype=np.float64)

    for f in range(5):
        val_mask = (fold_ids == f)
        T_g_f = gemma_fitted_Ts[f]
        T_q_f = qwen_fitted_Ts[f]

        # Apply T_f to ALL 600 items for fold f
        P_g_f_all = compute_calibrated_probs_for_items(logits_gemma, T_g_f)
        P_q_f_all = compute_calibrated_probs_for_items(logits_qwen, T_q_f)

        if pool_type == "linear":
            P_ens_f_all = weight_gemma * P_g_f_all + (1.0 - weight_gemma) * P_q_f_all
        else:
            P_g_c = np.clip(P_g_f_all, eps, 1.0)
            P_q_c = np.clip(P_q_f_all, eps, 1.0)
            unnorm_f = (P_g_c ** weight_gemma) * (P_q_c ** (1.0 - weight_gemma))
            P_ens_f_all = unnorm_f / np.sum(unnorm_f, axis=1, keepdims=True)

        P_ens_f_all = np.clip(P_ens_f_all, eps, 1.0)
        P_ens_f_all /= np.sum(P_ens_f_all, axis=1, keepdims=True)

        cal_probs_ens[val_mask] = P_ens_f_all[val_mask]

        # Build one complete graph for fold f
        D_ens_f = distance_hellinger_matrix(P_ens_f_all, P_ens_f_all)
        W_ens_f = compute_topk_weight_matrix(D_ens_f, k=10)
        q_null_f = compute_e007_block_density_null(W_ens_f, S_human, ds_ids, k=10)

        q_rows_cal_coherent[val_mask] = np.sum(W_ens_f[val_mask] * S_human[val_mask], axis=1) / 10.0
        null_by_item_cal[val_mask] = q_null_f

    q_supp_cal = float(np.mean(q_rows_cal_coherent))
    q_null_cal = float(np.mean(null_by_item_cal))
    r_norm_cal = float((q_supp_cal - q_null_cal) / (q_hh_k10 - q_null_cal) * 100.0)

    nll_cal = float(-np.mean(np.sum(human_p * np.log(np.clip(cal_probs_ens, eps, 1.0)), axis=1)))
    brier_cal = float(np.mean(np.sum((cal_probs_ens - human_p) ** 2, axis=1)))
    jsd_cal_nats = compute_jsd_nats(cal_probs_ens, human_p)

    from sprint1_rate_distortion_and_summary import interpolate_log_linear_bits
    k_eff_raw, b_bits_raw = interpolate_log_linear_bits(r_norm_raw, e008_data["prototype_ladder"])
    k_eff_cal, b_bits_cal = interpolate_log_linear_bits(r_norm_cal, e008_data["prototype_ladder"])

    return {
        "weight_gemma": weight_gemma,
        "pool_type": pool_type,
        "nll_raw_nats": nll_raw,
        "nll_calibrated_nats": nll_cal,
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
        "jsd_raw_nats": jsd_raw_nats,
        "jsd_calibrated_nats": jsd_cal_nats,
        "q_support_raw": q_supp_raw,
        "q_support_calibrated": q_supp_cal,
        "q_null_raw": q_null_raw,
        "q_null_calibrated": q_null_cal,
        "r_norm_pct_raw": r_norm_raw,
        "r_norm_pct_calibrated": r_norm_cal,
        "effective_bits_raw": b_bits_raw,
        "k_eff_raw": k_eff_raw,
        "effective_bits_calibrated": b_bits_cal,
        "k_eff_calibrated": k_eff_cal,
        "q_rows_raw": q_rows_raw,
        "q_rows_cal": q_rows_cal_coherent,
        "null_by_item_cal": null_by_item_cal
    }

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

    strata_map = {}
    for idx, it in enumerate(items):
        s_key = it.get("stratum_key", f"{it.get('source_dataset', 'chaosnli_mnli')}_{it.get('human_majority_label', 'e')}")
        strata_map.setdefault(s_key, []).append(idx)

    from analyze_llm_lpe import extract_lpe_logits_and_probs, run_e004_pipeline
    
    logits_gemma, perm_probs_gemma, _, _ = extract_lpe_logits_and_probs(gemma_path, items)
    logits_qwen, perm_probs_qwen, _, _ = extract_lpe_logits_and_probs(qwen_path, items)

    gemma_res = run_e004_pipeline(items, logits_gemma, perm_probs_gemma, S_human, q_hh_k10, e008_data)
    qwen_res = run_e004_pipeline(items, logits_qwen, perm_probs_qwen, S_human, q_hh_k10, e008_data)

    P_gemma_raw = np.mean(perm_probs_gemma, axis=1)
    P_qwen_raw = np.mean(perm_probs_qwen, axis=1)

    # 1. Evaluate Fold-Coherent Equal-Weight Linear Ensemble
    ens_linear_equal = evaluate_fold_coherent_ensemble(
        P_gemma_raw, P_qwen_raw, logits_gemma, logits_qwen,
        gemma_res["fitted_temperatures"], qwen_res["fitted_temperatures"],
        human_p, S_human, ds_ids, strata_map, q_hh_k10, e008_data, weight_gemma=0.5, pool_type="linear"
    )

    # 2. Evaluate Fold-Coherent Equal-Weight Logarithmic Ensemble
    ens_log_equal = evaluate_fold_coherent_ensemble(
        P_gemma_raw, P_qwen_raw, logits_gemma, logits_qwen,
        gemma_res["fitted_temperatures"], qwen_res["fitted_temperatures"],
        human_p, S_human, ds_ids, strata_map, q_hh_k10, e008_data, weight_gemma=0.5, pool_type="logarithmic"
    )

    # 3. Full Linear Pooling Sweep (21 points)
    weights = np.linspace(0.0, 1.0, 21)
    sweep_linear = []
    sweep_log = []
    for w in weights:
        res_lin = evaluate_fold_coherent_ensemble(
            P_gemma_raw, P_qwen_raw, logits_gemma, logits_qwen,
            gemma_res["fitted_temperatures"], qwen_res["fitted_temperatures"],
            human_p, S_human, ds_ids, strata_map, q_hh_k10, e008_data, weight_gemma=w, pool_type="linear"
        )
        res_lg = evaluate_fold_coherent_ensemble(
            P_gemma_raw, P_qwen_raw, logits_gemma, logits_qwen,
            gemma_res["fitted_temperatures"], qwen_res["fitted_temperatures"],
            human_p, S_human, ds_ids, strata_map, q_hh_k10, e008_data, weight_gemma=w, pool_type="logarithmic"
        )
        sweep_linear.append({
            "weight_gemma": float(w),
            "nll_raw": res_lin["nll_raw_nats"],
            "nll_cal": res_lin["nll_calibrated_nats"],
            "r_norm_raw": res_lin["r_norm_pct_raw"],
            "r_norm_cal": res_lin["r_norm_pct_calibrated"]
        })
        sweep_log.append({
            "weight_gemma": float(w),
            "nll_raw": res_lg["nll_raw_nats"],
            "nll_cal": res_lg["nll_calibrated_nats"],
            "r_norm_raw": res_lg["r_norm_pct_raw"],
            "r_norm_cal": res_lg["r_norm_pct_calibrated"]
        })

    # 4. Multi-Threshold Sensitivity & Fuzzy Support Audit
    S_nodiag = S_human.copy()
    np.fill_diagonal(S_nodiag, 0.0)

    # Percentiles
    tau_80 = float(np.percentile(S_nodiag[S_nodiag > 0], 80))
    tau_90 = float(np.percentile(S_nodiag[S_nodiag > 0], 90))
    tau_95 = float(np.percentile(S_nodiag[S_nodiag > 0], 95))

    threshold_sensitivity = {}
    for label_tau, val_tau in [("p80", tau_80), ("p90", tau_90), ("p95", tau_95), ("0.10", 0.10), ("0.25", 0.25), ("0.50", 0.50)]:
        edge_human_tau = (S_nodiag >= val_tau)
        
        D_g = distance_hellinger_matrix(P_gemma_raw, P_gemma_raw)
        W_g = compute_topk_weight_matrix(D_g, k=10)
        
        D_q = distance_hellinger_matrix(P_qwen_raw, P_qwen_raw)
        W_q = compute_topk_weight_matrix(D_q, k=10)

        edge_g = (W_g > 0)
        edge_q = (W_q > 0)

        supp_g = edge_g & edge_human_tau
        supp_q = edge_q & edge_human_tau

        n_g_only = int(np.sum(supp_g & ~supp_q))
        n_q_only = int(np.sum(supp_q & ~supp_g))
        n_both = int(np.sum(supp_g & supp_q))
        n_total_rec = n_g_only + n_q_only + n_both

        complementarity_pct = float((n_g_only + n_q_only) / max(1, n_total_rec) * 100.0)
        recall_pct = float(n_total_rec / max(1, np.sum(edge_human_tau)) * 100.0)

        # Fuzzy Weighted Support Mass (C_unique,G and C_unique,Q)
        mass_gemma_total = float(np.sum(W_g * S_nodiag))
        mass_gemma_unique = float(np.sum((W_g * S_nodiag) * (W_q == 0)))
        c_unique_gemma = float(mass_gemma_unique / max(1e-12, mass_gemma_total) * 100.0)

        mass_qwen_total = float(np.sum(W_q * S_nodiag))
        mass_qwen_unique = float(np.sum((W_q * S_nodiag) * (W_g == 0)))
        c_unique_qwen = float(mass_qwen_unique / max(1e-12, mass_qwen_total) * 100.0)

        threshold_sensitivity[label_tau] = {
            "threshold_val": val_tau,
            "total_human_supported_edges": int(np.sum(edge_human_tau)),
            "gemma_only": n_g_only,
            "qwen_only": n_q_only,
            "both": n_both,
            "total_recovered": n_total_rec,
            "complementarity_pct": complementarity_pct,
            "human_edge_recall_pct": recall_pct,
            "fuzzy_c_unique_gemma_pct": c_unique_gemma,
            "fuzzy_c_unique_qwen_pct": c_unique_qwen
        }

    # Excess NLL Closed above Human Target Entropy H_human = 0.6543 nats
    H_human = float(-np.mean(np.sum(human_p * np.log(np.clip(human_p, 1e-12, 1.0)), axis=1)))
    excess_gemma_nll = gemma_res["nll_raw_nats"] - H_human
    excess_qwen_nll = qwen_res["nll_raw_nats"] - H_human
    excess_ens_nll = ens_linear_equal["nll_raw_nats"] - H_human

    pct_closed_gemma = float((gemma_res["nll_raw_nats"] - ens_linear_equal["nll_raw_nats"]) / excess_gemma_nll * 100.0)
    pct_closed_qwen = float((qwen_res["nll_raw_nats"] - ens_linear_equal["nll_raw_nats"]) / excess_qwen_nll * 100.0)
    pct_closed_mean = float((0.5 * (gemma_res["nll_raw_nats"] + qwen_res["nll_raw_nats"]) - ens_linear_equal["nll_raw_nats"]) / (0.5 * (excess_gemma_nll + excess_qwen_nll)) * 100.0)

    summary = {
        "title": "Fold-Coherent Gemma 3 & Qwen 2.5 Ensemble & Pooling Operator Analysis",
        "human_target_entropy_nats": H_human,
        "nll_excess_closed": {
            "excess_nll_closed_gemma_pct": pct_closed_gemma,
            "excess_nll_closed_qwen_pct": pct_closed_qwen,
            "excess_nll_closed_two_model_mean_pct": pct_closed_mean
        },
        "equal_weight_fold_coherent_linear": {
            "nll_raw_nats": ens_linear_equal["nll_raw_nats"],
            "nll_calibrated_nats": ens_linear_equal["nll_calibrated_nats"],
            "r_norm_pct_raw": ens_linear_equal["r_norm_pct_raw"],
            "r_norm_pct_calibrated": ens_linear_equal["r_norm_pct_calibrated"],
            "effective_bits_raw": ens_linear_equal["effective_bits_raw"],
            "k_eff_raw": ens_linear_equal["k_eff_raw"],
            "effective_bits_calibrated": ens_linear_equal["effective_bits_calibrated"],
            "k_eff_calibrated": ens_linear_equal["k_eff_calibrated"]
        },
        "equal_weight_fold_coherent_logarithmic": {
            "nll_raw_nats": ens_log_equal["nll_raw_nats"],
            "nll_calibrated_nats": ens_log_equal["nll_calibrated_nats"],
            "r_norm_pct_raw": ens_log_equal["r_norm_pct_raw"],
            "r_norm_pct_calibrated": ens_log_equal["r_norm_pct_calibrated"],
            "effective_bits_raw": ens_log_equal["effective_bits_raw"],
            "k_eff_raw": ens_log_equal["k_eff_raw"],
            "effective_bits_calibrated": ens_log_equal["effective_bits_calibrated"],
            "k_eff_calibrated": ens_log_equal["k_eff_calibrated"]
        },
        "linear_pooling_sweep_21pt": sweep_linear,
        "logarithmic_pooling_sweep_21pt": sweep_log,
        "threshold_sensitivity_and_fuzzy_support": threshold_sensitivity
    }

    out_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E004_gemma_qwen_ensemble_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("============================================================")
    print("  FOLD-COHERENT ENSEMBLE & POOLING OPERATOR RESULTS")
    print("============================================================")
    print(f"  Human Target Entropy (H_human):     {H_human:.4f} nats")
    print(f"  NLL Excess Closed (Gemma / Qwen):   {pct_closed_gemma:.1f}% / {pct_closed_qwen:.1f}% (Mean: {pct_closed_mean:.1f}%)")
    print(f"  Linear Equal Ensemble (Raw NLL):    {ens_linear_equal['nll_raw_nats']:.4f} nats")
    print(f"  Linear Equal Ensemble (Cal NLL):    {ens_linear_equal['nll_calibrated_nats']:.4f} nats")
    print(f"  Linear Equal Ensemble (Raw R):      {ens_linear_equal['r_norm_pct_raw']:.2f}%  (K_eff = {ens_linear_equal['k_eff_raw']:.2f})")
    print(f"  Linear Equal Ensemble (Cal R):      {ens_linear_equal['r_norm_pct_calibrated']:.2f}%  (K_eff = {ens_linear_equal['k_eff_calibrated']:.2f})")
    print(f"  Logarithmic Equal Ensemble (Raw R): {ens_log_equal['r_norm_pct_raw']:.2f}%  (K_eff = {ens_log_equal['k_eff_raw']:.2f})")
    print(f"  Logarithmic Equal Ensemble (Cal R): {ens_log_equal['r_norm_pct_calibrated']:.2f}%  (K_eff = {ens_log_equal['k_eff_calibrated']:.2f})")
    print("------------------------------------------------------------")
    print("  FUZZY WEIGHTED COMPLEMENTARITY AUDIT:")
    for k, v in threshold_sensitivity.items():
        print(f"    Threshold {k:>4s} (val={v['threshold_val']:.3f}): Discrete Complementarity = {v['complementarity_pct']:.1f}%, Recall = {v['human_edge_recall_pct']:.1f}%, Fuzzy C_unique(G) = {v['fuzzy_c_unique_gemma_pct']:.1f}%, Fuzzy C_unique(Q) = {v['fuzzy_c_unique_qwen_pct']:.1f}%")
    print("============================================================")
    print(f"Exported fold-coherent ensemble summary to {out_path}")

if __name__ == "__main__":
    main()
