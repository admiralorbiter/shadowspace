"""E004 Stage 1.5 Three-Model Panel Synthesis & Coalition Census.

Evaluates Gemma 3 12B, Gemma 4 12B, and Qwen 2.5 14B on pilot_600:
  1. Gemma 4 Fold-Coherent Orbit & Support-Band Decomposition.
  2. 7-Coalition Census (G3, G4, Qwen, G3+G4, G3+Qwen, G4+Qwen, G3+G4+Qwen)
     for linear and logarithmic pooling operators.
  3. Paired 30-stratum bootstrap CIs for all coalitions against Qwen 2.5 alone.
  4. 7-Bucket Three-Model Edge Atlas (generational refinement vs family complementarity).
  5. Pointwise-Relational Pareto coordinates (NLL vs R_norm).
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

def evaluate_coalition(
    probs_dict_raw: Dict[str, np.ndarray],
    logits_dict: Dict[str, np.ndarray],
    fitted_Ts_dict: Dict[str, List[float]],
    member_keys: List[str],
    pool_type: str,
    human_p: np.ndarray,
    S_human: np.ndarray,
    ds_ids: np.ndarray,
    strata_map: Dict,
    q_hh_k10: float,
    e008_data: Dict
) -> Dict:
    N = len(human_p)
    fold_ids = np.zeros(N, dtype=int)
    for s_key, indices in strata_map.items():
        for rank, idx in enumerate(indices):
            fold_ids[idx] = rank % 5

    # 1. Raw Pool
    if len(member_keys) == 1:
        P_pool_raw = probs_dict_raw[member_keys[0]].copy()
    else:
        if pool_type == "linear":
            P_sum = np.zeros((N, 3), dtype=np.float64)
            for k in member_keys:
                P_sum += probs_dict_raw[k]
            P_pool_raw = P_sum / float(len(member_keys))
        else:  # logarithmic
            eps = 1e-12
            prod = np.ones((N, 3), dtype=np.float64)
            for k in member_keys:
                prod *= np.clip(probs_dict_raw[k], eps, 1.0) ** (1.0 / float(len(member_keys)))
            P_pool_raw = prod / np.sum(prod, axis=1, keepdims=True)

    eps = 1e-12
    nll_raw = float(-np.mean(np.sum(human_p * np.log(np.clip(P_pool_raw, eps, 1.0)), axis=1)))
    D_raw = distance_hellinger_matrix(P_pool_raw, P_pool_raw)
    W_raw = compute_topk_weight_matrix(D_raw, k=10)
    q_rows_raw = np.sum(W_raw * S_human, axis=1) / 10.0
    q_supp_raw = float(np.mean(q_rows_raw))
    q_null_raw = compute_e007_block_density_null(W_raw, S_human, ds_ids, k=10)
    r_norm_raw = float((q_supp_raw - q_null_raw) / (q_hh_k10 - q_null_raw) * 100.0)

    # 2. Fold-Coherent Calibrated Pool
    from analyze_llm_lpe import compute_calibrated_probs_for_items
    cal_probs_pool = np.zeros((N, 3), dtype=np.float64)
    q_rows_cal_coherent = np.zeros(N, dtype=np.float64)
    null_by_item_cal = np.zeros(N, dtype=np.float64)

    for f in range(5):
        val_mask = (fold_ids == f)
        
        # Apply fold f temperature for each member model to ALL 600 items
        P_cal_f_members = {}
        for k in member_keys:
            T_k_f = fitted_Ts_dict[k][f]
            P_cal_f_members[k] = compute_calibrated_probs_for_items(logits_dict[k], T_k_f)

        if len(member_keys) == 1:
            P_pool_f_all = P_cal_f_members[member_keys[0]].copy()
        else:
            if pool_type == "linear":
                P_sum_f = np.zeros((N, 3), dtype=np.float64)
                for k in member_keys:
                    P_sum_f += P_cal_f_members[k]
                P_pool_f_all = P_sum_f / float(len(member_keys))
            else:
                prod_f = np.ones((N, 3), dtype=np.float64)
                for k in member_keys:
                    prod_f *= np.clip(P_cal_f_members[k], eps, 1.0) ** (1.0 / float(len(member_keys)))
                P_pool_f_all = prod_f / np.sum(prod_f, axis=1, keepdims=True)

        cal_probs_pool[val_mask] = P_pool_f_all[val_mask]

        D_pool_f = distance_hellinger_matrix(P_pool_f_all, P_pool_f_all)
        W_pool_f = compute_topk_weight_matrix(D_pool_f, k=10)
        q_null_f = compute_e007_block_density_null(W_pool_f, S_human, ds_ids, k=10)

        q_rows_cal_coherent[val_mask] = np.sum(W_pool_f[val_mask] * S_human[val_mask], axis=1) / 10.0
        null_by_item_cal[val_mask] = q_null_f

    q_supp_cal = float(np.mean(q_rows_cal_coherent))
    q_null_cal = float(np.mean(null_by_item_cal))
    r_norm_cal = float((q_supp_cal - q_null_cal) / (q_hh_k10 - q_null_cal) * 100.0)
    nll_cal = float(-np.mean(np.sum(human_p * np.log(np.clip(cal_probs_pool, eps, 1.0)), axis=1)))

    from sprint1_rate_distortion_and_summary import interpolate_log_linear_bits
    k_eff_raw, b_bits_raw = interpolate_log_linear_bits(r_norm_raw, e008_data["prototype_ladder"])
    k_eff_cal, b_bits_cal = interpolate_log_linear_bits(r_norm_cal, e008_data["prototype_ladder"])

    return {
        "coalition_name": "+".join(member_keys),
        "member_keys": member_keys,
        "pool_type": pool_type,
        "nll_raw_nats": nll_raw,
        "nll_calibrated_nats": nll_cal,
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
        "null_by_item_cal": null_by_item_cal,
        "W_raw": W_raw
    }

def main():
    manifest_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "manifests" / "pilot_600.jsonl"
    supp_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "pilot_support" / "S_hellinger_k010_pilot.bin"

    resp_g3 = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "raw_responses" / "pilot600_gemma3-12b_v2_abc_t10_lpe.jsonl"
    resp_g4 = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "raw_responses" / "pilot600_gemma4-12b_v2_abc_t10_lpe.jsonl"
    resp_qwen = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "raw_responses" / "pilot600_qwen2.5-14b_v2_abc_t10_lpe.jsonl"

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

    logits_g3, probs_g3, _, _ = extract_lpe_logits_and_probs(resp_g3, items)
    logits_g4, probs_g4, _, _ = extract_lpe_logits_and_probs(resp_g4, items)
    logits_qwen, probs_qwen, _, _ = extract_lpe_logits_and_probs(resp_qwen, items)

    g3_res = run_e004_pipeline(items, logits_g3, probs_g3, S_human, q_hh_k10, e008_data)
    g4_res = run_e004_pipeline(items, logits_g4, probs_g4, S_human, q_hh_k10, e008_data)
    qwen_res = run_e004_pipeline(items, logits_qwen, probs_qwen, S_human, q_hh_k10, e008_data)

    probs_dict_raw = {
        "G3": np.mean(probs_g3, axis=1),
        "G4": np.mean(probs_g4, axis=1),
        "Qwen": np.mean(probs_qwen, axis=1)
    }

    logits_dict = {
        "G3": logits_g3,
        "G4": logits_g4,
        "Qwen": logits_qwen
    }

    fitted_Ts_dict = {
        "G3": g3_res["fitted_temperatures"],
        "G4": g4_res["fitted_temperatures"],
        "Qwen": qwen_res["fitted_temperatures"]
    }

    # 1. Fold-Coherent Gemma 4 Support-Band Decomposition
    S_nodiag = S_human.copy()
    np.fill_diagonal(S_nodiag, 0.0)

    bands = {
        "b1_under_005": S_nodiag < 0.05,
        "b2_005_to_025": (S_nodiag >= 0.05) & (S_nodiag < 0.25),
        "b3_025_to_050": (S_nodiag >= 0.25) & (S_nodiag < 0.50),
        "b4_over_050": S_nodiag >= 0.50
    }

    W_g4_raw = compute_topk_weight_matrix(distance_hellinger_matrix(probs_dict_raw["G4"], probs_dict_raw["G4"]), k=10)
    
    from analyze_llm_lpe import compute_calibrated_probs_for_items
    W_g4_cal_folds = np.zeros((5, N, N), dtype=np.float64)
    for f in range(5):
        P_g4_f = compute_calibrated_probs_for_items(logits_g4, g4_res["fitted_temperatures"][f])
        D_g4_f = distance_hellinger_matrix(P_g4_f, P_g4_f)
        W_g4_cal_folds[f] = compute_topk_weight_matrix(D_g4_f, k=10)

    W_g4_cal_avg = np.mean(W_g4_cal_folds, axis=0)
    dW_g4 = W_g4_cal_avg - W_g4_raw

    g4_support_band_decomp = {}
    total_dq_supp_g4 = float(np.mean(np.sum(dW_g4 * S_nodiag, axis=1) / 10.0))

    for b_name, b_mask in bands.items():
        delta_q_band = float(np.mean(np.sum((dW_g4 * b_mask) * S_nodiag, axis=1) / 10.0))
        share = float(delta_q_band / max(1e-12, total_dq_supp_g4) * 100.0) if total_dq_supp_g4 > 0 else 0.0
        g4_support_band_decomp[b_name] = {
            "delta_q_support_band": delta_q_band,
            "share_of_total_delta_q_support_pct": share
        }

    # 2. Evaluate All 7 Coalitions for Linear and Logarithmic Pools
    coalition_specs = [
        ["G3"],
        ["G4"],
        ["Qwen"],
        ["G3", "G4"],
        ["G3", "Qwen"],
        ["G4", "Qwen"],
        ["G3", "G4", "Qwen"]
    ]

    coalition_results = {}
    serializable_coalition_results = {}
    rng = np.random.default_rng(20260803)
    strata_indices = {s: np.where(np.array([it.get("stratum_key", f"{it.get('source_dataset', 'chaosnli_mnli')}_{it.get('human_majority_label', 'e')}") for it in items]) == s)[0] for s in strata_map.keys()}

    qwen_q_rows_cal = qwen_res["q_rows_cal"]
    qwen_null_cal = qwen_res["null_by_item_cal"]

    for spec in coalition_specs:
        c_name = "+".join(spec)
        pool_types = ["linear"] if len(spec) == 1 else ["linear", "logarithmic"]

        for p_type in pool_types:
            key = f"{c_name}_{p_type}"
            c_res = evaluate_coalition(
                probs_dict_raw, logits_dict, fitted_Ts_dict, spec, p_type,
                human_p, S_human, ds_ids, strata_map, q_hh_k10, e008_data
            )

            # Paired 30-stratum bootstrap CI against Qwen alone
            boot_diff_cal = []
            for _ in range(1000):
                boot_idx_list = []
                for s, s_idx in strata_indices.items():
                    sampled_s = rng.choice(s_idx, size=len(s_idx), replace=True)
                    boot_idx_list.extend(sampled_s)
                idx_boot = np.array(boot_idx_list)

                null_qwen_b = float(np.mean(qwen_null_cal[idx_boot]))
                r_qwen_b = float((np.mean(qwen_q_rows_cal[idx_boot]) - null_qwen_b) / (q_hh_k10 - null_qwen_b) * 100.0)

                null_c_b = float(np.mean(c_res["null_by_item_cal"][idx_boot]))
                r_c_b = float((np.mean(c_res["q_rows_cal"][idx_boot]) - null_c_b) / (q_hh_k10 - null_c_b) * 100.0)

                boot_diff_cal.append(r_c_b - r_qwen_b)

            ci_diff_vs_qwen = [float(np.percentile(boot_diff_cal, 2.5)), float(np.percentile(boot_diff_cal, 97.5))]
            c_res["delta_r_vs_qwen_calibrated_pct"] = c_res["r_norm_pct_calibrated"] - qwen_res["r_norm_pct_calibrated"]
            c_res["delta_r_vs_qwen_calibrated_95ci"] = ci_diff_vs_qwen

            coalition_results[key] = c_res

            # Clean JSON-serializable version
            ser_c = {k_sub: v_sub for k_sub, v_sub in c_res.items() if not isinstance(v_sub, np.ndarray)}
            serializable_coalition_results[key] = ser_c

    # 3. Three-Model Edge Atlas (7 Mutually Exclusive Buckets)
    W_g3 = compute_topk_weight_matrix(distance_hellinger_matrix(probs_dict_raw["G3"], probs_dict_raw["G3"]), k=10)
    W_g4 = compute_topk_weight_matrix(distance_hellinger_matrix(probs_dict_raw["G4"], probs_dict_raw["G4"]), k=10)
    W_qwen = compute_topk_weight_matrix(distance_hellinger_matrix(probs_dict_raw["Qwen"], probs_dict_raw["Qwen"]), k=10)

    edge_human = (S_nodiag >= 0.05)
    e_g3 = (W_g3 > 0) & edge_human
    e_g4 = (W_g4 > 0) & edge_human
    e_qwen = (W_qwen > 0) & edge_human

    atlas_all_three = e_g3 & e_g4 & e_qwen
    atlas_gemma_family_shared = e_g3 & e_g4 & ~e_qwen
    atlas_gemma3_only = e_g3 & ~e_g4 & ~e_qwen
    atlas_gemma4_only = e_g4 & ~e_g3 & ~e_qwen
    atlas_qwen_only = e_qwen & ~e_g3 & ~e_g4
    atlas_gemma4_qwen_shared = e_g4 & e_qwen & ~e_g3
    atlas_gemma3_qwen_shared = e_g3 & e_qwen & ~e_g4

    total_captured_edges = int(np.sum(e_g3 | e_g4 | e_qwen))

    atlas_buckets = {
        "all_three": {
            "num_edges": int(np.sum(atlas_all_three)),
            "weighted_support_mass": float(np.sum(S_nodiag[atlas_all_three])),
            "share_of_captured_edges_pct": float(np.sum(atlas_all_three) / max(1, total_captured_edges) * 100.0)
        },
        "gemma_family_shared_only": {
            "num_edges": int(np.sum(atlas_gemma_family_shared)),
            "weighted_support_mass": float(np.sum(S_nodiag[atlas_gemma_family_shared])),
            "share_of_captured_edges_pct": float(np.sum(atlas_gemma_family_shared) / max(1, total_captured_edges) * 100.0)
        },
        "gemma3_only": {
            "num_edges": int(np.sum(atlas_gemma3_only)),
            "weighted_support_mass": float(np.sum(S_nodiag[atlas_gemma3_only])),
            "share_of_captured_edges_pct": float(np.sum(atlas_gemma3_only) / max(1, total_captured_edges) * 100.0)
        },
        "gemma4_only": {
            "num_edges": int(np.sum(atlas_gemma4_only)),
            "weighted_support_mass": float(np.sum(S_nodiag[atlas_gemma4_only])),
            "share_of_captured_edges_pct": float(np.sum(atlas_gemma4_only) / max(1, total_captured_edges) * 100.0)
        },
        "qwen_only": {
            "num_edges": int(np.sum(atlas_qwen_only)),
            "weighted_support_mass": float(np.sum(S_nodiag[atlas_qwen_only])),
            "share_of_captured_edges_pct": float(np.sum(atlas_qwen_only) / max(1, total_captured_edges) * 100.0)
        },
        "gemma4_qwen_shared_only": {
            "num_edges": int(np.sum(atlas_gemma4_qwen_shared)),
            "weighted_support_mass": float(np.sum(S_nodiag[atlas_gemma4_qwen_shared])),
            "share_of_captured_edges_pct": float(np.sum(atlas_gemma4_qwen_shared) / max(1, total_captured_edges) * 100.0)
        },
        "gemma3_qwen_shared_only": {
            "num_edges": int(np.sum(atlas_gemma3_qwen_shared)),
            "weighted_support_mass": float(np.sum(S_nodiag[atlas_gemma3_qwen_shared])),
            "share_of_captured_edges_pct": float(np.sum(atlas_gemma3_qwen_shared) / max(1, total_captured_edges) * 100.0)
        }
    }

    # 4. Pointwise-Relational Pareto Data
    pareto_points = []
    for k, v in coalition_results.items():
        pareto_points.append({
            "name": k,
            "nll_raw": v["nll_raw_nats"],
            "nll_cal": v["nll_calibrated_nats"],
            "r_norm_raw": v["r_norm_pct_raw"],
            "r_norm_cal": v["r_norm_pct_calibrated"]
        })

    summary = {
        "title": "E004 Stage 1.5 Three-Model Panel Synthesis & Coalition Census",
        "gemma4_support_band_decomposition": g4_support_band_decomp,
        "coalition_census": serializable_coalition_results,
        "three_model_edge_atlas": atlas_buckets,
        "pareto_pointwise_relational_coordinates": pareto_points
    }

    out_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E004_three_model_coalition_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n============================================================")
    print("  STAGE 1.5 THREE-MODEL COALITION CENSUS & EDGE ATLAS")
    print("============================================================")
    print(f"  Gemma 4 Orbit Support Decomp (+0.21% gain):")
    for b_k, b_v in g4_support_band_decomp.items():
        print(f"    {b_k}: Delta Q = {b_v['delta_q_support_band']:+.6f} ({b_v['share_of_total_delta_q_support_pct']:.1f}%)")
    print(f"------------------------------------------------------------")
    print("  Coalition Census Highlights (Calibrated R):")
    for k in ["G3_linear", "G4_linear", "Qwen_linear", "G3+G4_linear", "G3+Qwen_linear", "G4+Qwen_linear", "G3+G4+Qwen_linear", "G3+G4+Qwen_logarithmic"]:
        if k in coalition_results:
            c = coalition_results[k]
            print(f"    {k:22s}: NLL={c['nll_calibrated_nats']:.4f} nats | R_cal={c['r_norm_pct_calibrated']:.2f}% | vs Qwen: {c['delta_r_vs_qwen_calibrated_pct']:+.2f}% (95% CI: [{c['delta_r_vs_qwen_calibrated_95ci'][0]:+.2f}%, {c['delta_r_vs_qwen_calibrated_95ci'][1]:+.2f}%])")
    print(f"------------------------------------------------------------")
    print("  Three-Model Edge Atlas:")
    for b_k, b_v in atlas_buckets.items():
        print(f"    {b_k:26s}: {b_v['num_edges']} edges ({b_v['share_of_captured_edges_pct']:.1f}%) | Mass = {b_v['weighted_support_mass']:.4f}")
    print("============================================================")
    print(f"Exported Stage 1.5 synthesis summary to {out_path}")

if __name__ == "__main__":
    main()
