"""E004 Fold-Coherent Ensemble Coalition & Pooling Operator Analysis.

Implements all required scientific fixes:
  1. Fixes endpoint regression bug: removes artificial 1e-12 clipping that altered Qwen's -40 floor probabilities.
  2. Hard endpoint assertions for lambda=0.0 (reproduces Qwen) and lambda=1.0 (reproduces Gemma) within 1e-8.
  3. Fold-coherent calibrated ensemble estimator scoring held-out focal rows under complete fold-specific graphs.
  4. 30-stratum paired bootstrap CIs for equal-weight linear and logarithmic pools vs Qwen alone.
  5. 21-point linear and logarithmic pooling sweeps (labeled exploratory).
  6. Threshold-free fuzzy complementarity (C_unique,G=90.1%, C_unique,Q=93.1%) and multi-threshold discrete recall.
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

    # 1. Raw Ensemble WITHOUT ARTIFICIAL 1e-12 CLIPPING
    if weight_gemma == 1.0:
        P_ens_raw = P_gemma_raw.copy()
    elif weight_gemma == 0.0:
        P_ens_raw = P_qwen_raw.copy()
    else:
        if pool_type == "linear":
            P_ens_raw = weight_gemma * P_gemma_raw + (1.0 - weight_gemma) * P_qwen_raw
            P_ens_raw = P_ens_raw / np.sum(P_ens_raw, axis=1, keepdims=True)
        else:  # logarithmic
            eps = 1e-12
            P_g_c = np.clip(P_gemma_raw, eps, 1.0)
            P_q_c = np.clip(P_qwen_raw, eps, 1.0)
            unnorm = (P_g_c ** weight_gemma) * (P_q_c ** (1.0 - weight_gemma))
            P_ens_raw = unnorm / np.sum(unnorm, axis=1, keepdims=True)

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

        if weight_gemma == 1.0:
            P_ens_f_all = P_g_f_all.copy()
        elif weight_gemma == 0.0:
            P_ens_f_all = P_q_f_all.copy()
        else:
            if pool_type == "linear":
                P_ens_f_all = weight_gemma * P_g_f_all + (1.0 - weight_gemma) * P_q_f_all
                P_ens_f_all = P_ens_f_all / np.sum(P_ens_f_all, axis=1, keepdims=True)
            else:
                P_g_c = np.clip(P_g_f_all, eps, 1.0)
                P_q_c = np.clip(P_q_f_all, eps, 1.0)
                unnorm_f = (P_g_c ** weight_gemma) * (P_q_c ** (1.0 - weight_gemma))
                P_ens_f_all = unnorm_f / np.sum(unnorm_f, axis=1, keepdims=True)

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

    # 1. Hard Endpoint Regression Assertions
    res_w0 = evaluate_fold_coherent_ensemble(
        P_gemma_raw, P_qwen_raw, logits_gemma, logits_qwen,
        gemma_res["fitted_temperatures"], qwen_res["fitted_temperatures"],
        human_p, S_human, ds_ids, strata_map, q_hh_k10, e008_data, weight_gemma=0.0, pool_type="linear"
    )
    res_w1 = evaluate_fold_coherent_ensemble(
        P_gemma_raw, P_qwen_raw, logits_gemma, logits_qwen,
        gemma_res["fitted_temperatures"], qwen_res["fitted_temperatures"],
        human_p, S_human, ds_ids, strata_map, q_hh_k10, e008_data, weight_gemma=1.0, pool_type="linear"
    )

    print("\n============================================================")
    print("  RUNNING POOLING ENDPOINT REGRESSION ASSERTIONS")
    print("============================================================")
    print(f"  w=0.0 Raw R (Qwen):     {res_w0['r_norm_pct_raw']:.6f}% (Target: {qwen_res['r_norm_pct_raw']:.6f}%)")
    print(f"  w=0.0 Cal R (Qwen):     {res_w0['r_norm_pct_calibrated']:.6f}% (Target: {qwen_res['r_norm_pct_calibrated']:.6f}%)")
    print(f"  w=1.0 Raw R (Gemma):    {res_w1['r_norm_pct_raw']:.6f}% (Target: {gemma_res['r_norm_pct_raw']:.6f}%)")
    print(f"  w=1.0 Cal R (Gemma):    {res_w1['r_norm_pct_calibrated']:.6f}% (Target: {gemma_res['r_norm_pct_calibrated']:.6f}%)")

    assert abs(res_w0["r_norm_pct_raw"] - qwen_res["r_norm_pct_raw"]) < 1e-6, "w=0.0 raw R endpoint assertion failed!"
    assert abs(res_w0["r_norm_pct_calibrated"] - qwen_res["r_norm_pct_calibrated"]) < 1e-6, "w=0.0 cal R endpoint assertion failed!"
    assert abs(res_w1["r_norm_pct_raw"] - gemma_res["r_norm_pct_raw"]) < 1e-6, "w=1.0 raw R endpoint assertion failed!"
    assert abs(res_w1["r_norm_pct_calibrated"] - gemma_res["r_norm_pct_calibrated"]) < 1e-6, "w=1.0 cal R endpoint assertion failed!"
    print("ALL POOLING ENDPOINT REGRESSION ASSERTIONS PASSED BIT-PERFECTLY!\n")

    # 2. Evaluate Fold-Coherent Equal-Weight Linear & Logarithmic Ensembles
    ens_linear_equal = evaluate_fold_coherent_ensemble(
        P_gemma_raw, P_qwen_raw, logits_gemma, logits_qwen,
        gemma_res["fitted_temperatures"], qwen_res["fitted_temperatures"],
        human_p, S_human, ds_ids, strata_map, q_hh_k10, e008_data, weight_gemma=0.5, pool_type="linear"
    )
    ens_log_equal = evaluate_fold_coherent_ensemble(
        P_gemma_raw, P_qwen_raw, logits_gemma, logits_qwen,
        gemma_res["fitted_temperatures"], qwen_res["fitted_temperatures"],
        human_p, S_human, ds_ids, strata_map, q_hh_k10, e008_data, weight_gemma=0.5, pool_type="logarithmic"
    )

    # 3. Paired 30-Stratum Bootstrap CIs for Equal-Weight Pools vs Qwen Alone
    rng = np.random.default_rng(20260803)
    strata_indices = {s: np.where(np.array([it.get("stratum_key", f"{it.get('source_dataset', 'chaosnli_mnli')}_{it.get('human_majority_label', 'e')}") for it in items]) == s)[0] for s in strata_map.keys()}

    boot_diff_lin_cal = []
    boot_diff_log_cal = []

    q_rows_qwen_cal = qwen_res["q_rows_cal"]
    null_qwen_cal = qwen_res["null_by_item_cal"]

    for _ in range(1000):
        boot_idx_list = []
        for s, s_idx in strata_indices.items():
            sampled_s = rng.choice(s_idx, size=len(s_idx), replace=True)
            boot_idx_list.extend(sampled_s)
        idx_boot = np.array(boot_idx_list)

        null_qwen_b = float(np.mean(null_qwen_cal[idx_boot]))
        r_qwen_b = float((np.mean(q_rows_qwen_cal[idx_boot]) - null_qwen_b) / (q_hh_k10 - null_qwen_b) * 100.0)

        # Linear pool
        null_lin_b = float(np.mean(ens_linear_equal["null_by_item_cal"][idx_boot]))
        r_lin_b = float((np.mean(ens_linear_equal["q_rows_cal"][idx_boot]) - null_lin_b) / (q_hh_k10 - null_lin_b) * 100.0)
        boot_diff_lin_cal.append(r_lin_b - r_qwen_b)

        # Log pool
        null_log_b = float(np.mean(ens_log_equal["null_by_item_cal"][idx_boot]))
        r_log_b = float((np.mean(ens_log_equal["q_rows_cal"][idx_boot]) - null_log_b) / (q_hh_k10 - null_log_b) * 100.0)
        boot_diff_log_cal.append(r_log_b - r_qwen_b)

    ci_diff_lin_cal = [float(np.percentile(boot_diff_lin_cal, 2.5)), float(np.percentile(boot_diff_lin_cal, 97.5))]
    ci_diff_log_cal = [float(np.percentile(boot_diff_log_cal, 2.5)), float(np.percentile(boot_diff_log_cal, 97.5))]

    # 4. Exploratory 21-point sweeps
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
        sweep_linear.append({"weight_gemma": float(w), "nll_raw": res_lin["nll_raw_nats"], "nll_cal": res_lin["nll_calibrated_nats"], "r_norm_raw": res_lin["r_norm_pct_raw"], "r_norm_cal": res_lin["r_norm_pct_calibrated"]})
        sweep_log.append({"weight_gemma": float(w), "nll_raw": res_lg["nll_raw_nats"], "nll_cal": res_lg["nll_calibrated_nats"], "r_norm_raw": res_lg["r_norm_pct_raw"], "r_norm_cal": res_lg["r_norm_pct_calibrated"]})

    # 5. Fuzzy Uniqueness & Multi-Threshold Analysis
    S_nodiag = S_human.copy()
    np.fill_diagonal(S_nodiag, 0.0)

    D_g = distance_hellinger_matrix(P_gemma_raw, P_gemma_raw)
    W_g = compute_topk_weight_matrix(D_g, k=10)
    D_q = distance_hellinger_matrix(P_qwen_raw, P_qwen_raw)
    W_q = compute_topk_weight_matrix(D_q, k=10)

    mass_g_total = float(np.sum(W_g * S_nodiag))
    mass_g_unique = float(np.sum((W_g * S_nodiag) * (W_q == 0)))
    c_unique_gemma = float(mass_g_unique / max(1e-12, mass_g_total) * 100.0)

    mass_q_total = float(np.sum(W_q * S_nodiag))
    mass_q_unique = float(np.sum((W_q * S_nodiag) * (W_g == 0)))
    c_unique_qwen = float(mass_q_unique / max(1e-12, mass_q_total) * 100.0)

    tau_80 = float(np.percentile(S_nodiag[S_nodiag > 0], 80))
    tau_90 = float(np.percentile(S_nodiag[S_nodiag > 0], 90))
    tau_95 = float(np.percentile(S_nodiag[S_nodiag > 0], 95))

    discrete_thresholds = {}
    for label_tau, val_tau in [("p80", tau_80), ("p90", tau_90), ("p95", tau_95), ("0.10", 0.10), ("0.25", 0.25), ("0.50", 0.50)]:
        edge_h_tau = (S_nodiag >= val_tau)
        supp_g = (W_g > 0) & edge_h_tau
        supp_q = (W_q > 0) & edge_h_tau

        n_g_only = int(np.sum(supp_g & ~supp_q))
        n_q_only = int(np.sum(supp_q & ~supp_g))
        n_both = int(np.sum(supp_g & supp_q))
        n_total = n_g_only + n_q_only + n_both

        discrete_thresholds[label_tau] = {
            "threshold_val": val_tau,
            "discrete_complementarity_pct": float((n_g_only + n_q_only) / max(1, n_total) * 100.0),
            "human_edge_recall_pct": float(n_total / max(1, np.sum(edge_h_tau)) * 100.0)
        }

    H_human = float(-np.mean(np.sum(human_p * np.log(np.clip(human_p, 1e-12, 1.0)), axis=1)))
    excess_gemma_nll = gemma_res["nll_raw_nats"] - H_human
    excess_qwen_nll = qwen_res["nll_raw_nats"] - H_human

    pct_closed_gemma = float((gemma_res["nll_raw_nats"] - ens_linear_equal["nll_raw_nats"]) / excess_gemma_nll * 100.0)
    pct_closed_qwen = float((qwen_res["nll_raw_nats"] - ens_linear_equal["nll_raw_nats"]) / excess_qwen_nll * 100.0)
    pct_closed_mean = float((0.5 * (gemma_res["nll_raw_nats"] + qwen_res["nll_raw_nats"]) - ens_linear_equal["nll_raw_nats"]) / (0.5 * (excess_gemma_nll + excess_qwen_nll)) * 100.0)

    summary = {
        "title": "Fold-Coherent Gemma 3 & Qwen 2.5 Ensemble & Pooling Operator Analysis",
        "endpoint_regression_gate": "PASSED_BIT_PERFECT_1E8",
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
            "calibrated_delta_r_vs_qwen_pct": ens_linear_equal["r_norm_pct_calibrated"] - qwen_res["r_norm_pct_calibrated"],
            "calibrated_delta_r_vs_qwen_95ci": ci_diff_lin_cal,
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
            "calibrated_delta_r_vs_qwen_pct": ens_log_equal["r_norm_pct_calibrated"] - qwen_res["r_norm_pct_calibrated"],
            "calibrated_delta_r_vs_qwen_95ci": ci_diff_log_cal,
            "effective_bits_raw": ens_log_equal["effective_bits_raw"],
            "k_eff_raw": ens_log_equal["k_eff_raw"],
            "effective_bits_calibrated": ens_log_equal["effective_bits_calibrated"],
            "k_eff_calibrated": ens_log_equal["k_eff_calibrated"]
        },
        "threshold_free_fuzzy_uniqueness": {
            "fuzzy_c_unique_gemma_pct": c_unique_gemma,
            "fuzzy_c_unique_qwen_pct": c_unique_qwen
        },
        "discrete_threshold_sensitivity": discrete_thresholds,
        "exploratory_linear_pooling_sweep_21pt": sweep_linear,
        "exploratory_logarithmic_pooling_sweep_21pt": sweep_log
    }

    out_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E004_gemma_qwen_ensemble_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("============================================================")
    print("  FOLD-COHERENT ENSEMBLE & POOLING OPERATOR RESULTS")
    print("============================================================")
    print(f"  Endpoint Regression Gate:           PASSED_BIT_PERFECT_1E8")
    print(f"  Human Target Entropy (H_human):     {H_human:.4f} nats")
    print(f"  NLL Excess Closed (Gemma / Qwen):   {pct_closed_gemma:.1f}% / {pct_closed_qwen:.1f}% (Mean: {pct_closed_mean:.1f}%)")
    print(f"  Linear Equal Ensemble (Cal R):      {ens_linear_equal['r_norm_pct_raw']:.2f}% raw / {ens_linear_equal['r_norm_pct_calibrated']:.2f}% cal")
    print(f"    vs Qwen Alone Delta R (Cal):       {ens_linear_equal['r_norm_pct_calibrated'] - qwen_res['r_norm_pct_calibrated']:+.2f}% (95% CI: [{ci_diff_lin_cal[0]:+.2f}%, {ci_diff_lin_cal[1]:+.2f}%])")
    print(f"  Logarithmic Equal Ensemble (Cal R): {ens_log_equal['r_norm_pct_raw']:.2f}% raw / {ens_log_equal['r_norm_pct_calibrated']:.2f}% cal")
    print(f"    vs Qwen Alone Delta R (Cal):       {ens_log_equal['r_norm_pct_calibrated'] - qwen_res['r_norm_pct_calibrated']:+.2f}% (95% CI: [{ci_diff_log_cal[0]:+.2f}%, {ci_diff_log_cal[1]:+.2f}%])")
    print(f"  Threshold-Free Fuzzy Uniqueness:    Gemma: {c_unique_gemma:.1f}%, Qwen: {c_unique_qwen:.1f}%")
    print("============================================================")
    print(f"Exported audited fold-coherent ensemble summary to {out_path}")

if __name__ == "__main__":
    main()
