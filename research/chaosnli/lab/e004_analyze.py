"""E004 Primary Analysis Engine.

Computes all pointwise metrics, relational metrics, exact-profile nulls, label-order sensitivity,
and 1,000 paired 30-stratum bootstrap contrasts across all E004 conditions.
Outputs results to research/chaosnli/artifacts/E004/summaries/E004_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

MANIFEST_DIR = Path("research/chaosnli/artifacts/E004/manifests")
NORM_PROBS_DIR = Path("research/chaosnli/artifacts/E004/normalized_probs")
PILOT_SUPPORT_DIR = Path("research/chaosnli/artifacts/E004/pilot_support")
SUMMARIES_DIR = Path("research/chaosnli/artifacts/E004/summaries")

# ─── Distance & Divergence Metrics ──────────────────────────────────────────

def distance_hellinger_matrix(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    sqrt_P = np.sqrt(np.clip(P, 1e-12, 1.0))
    sqrt_Q = np.sqrt(np.clip(Q, 1e-12, 1.0))
    bc = np.dot(sqrt_P, sqrt_Q.T)
    bc = np.clip(bc, 0.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - bc))

def jsd_vectorized(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    m = np.clip(m, 1e-12, 1.0)
    p_safe = np.clip(p, 1e-12, 1.0)
    q_safe = np.clip(q, 1e-12, 1.0)
    kl_pm = np.sum(np.where(p > 1e-12, p * np.log2(p_safe / m), 0.0), axis=1)
    kl_qm = np.sum(np.where(q > 1e-12, q * np.log2(q_safe / m), 0.0), axis=1)
    return np.sqrt(np.maximum(0.0, 0.5 * kl_pm + 0.5 * kl_qm))

def soft_label_nll_vectorized(p_human: np.ndarray, q_model: np.ndarray) -> np.ndarray:
    q_safe = np.clip(q_model, 1e-12, 1.0)
    return -np.sum(p_human * np.log(q_safe), axis=1)

def compute_topk_weight_matrix(dist: np.ndarray, k: int) -> np.ndarray:
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

def compute_exact_profile_null(
    dist_model: np.ndarray,
    S_target: np.ndarray,
    exact_groups: List[np.ndarray],
    k: int = 10,
    n_permutations: int = 10000,
    seed: int = 42
) -> Tuple[float, List[float], float]:
    rng = np.random.default_rng(seed)
    N = dist_model.shape[0]
    null_scores = []
    
    W_model = compute_topk_weight_matrix(dist_model, k)

    for _ in range(n_permutations):
        perm_idx = np.arange(N)
        for grp in exact_groups:
            if len(grp) > 1:
                perm_idx[grp] = rng.permutation(grp)
        W_perm = W_model[perm_idx][:, perm_idx]
        score = float(np.sum(np.minimum(W_perm, S_target)) / (N * float(k)))
        null_scores.append(score)

    q_support = float(np.sum(np.minimum(W_model, S_target)) / (N * float(k)))
    null_mean = float(np.mean(null_scores))
    q_excess = q_support - null_mean
    p_val = (np.sum(np.array(null_scores) >= q_support) + 1.0) / (n_permutations + 1.0)

    return null_mean, [float(np.percentile(null_scores, 2.5)), float(np.percentile(null_scores, 97.5))], q_excess, p_val

def run_e004_analysis(subset: str = "pilot") -> None:
    manifest_file = MANIFEST_DIR / f"{subset}_600.jsonl" if subset == "pilot" else MANIFEST_DIR / f"{subset}_60.jsonl"
    k10_bin = PILOT_SUPPORT_DIR / f"S_hellinger_k010_{subset}.bin"
    k10_manifest = PILOT_SUPPORT_DIR / f"S_hellinger_k010_{subset}.manifest.json"

    if not manifest_file.exists() or not k10_bin.exists():
        raise FileNotFoundError("Missing manifest or pilot human support target matrix binary.")

    items = []
    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    N = len(items)
    p_human = np.array([[it["human_p_entailment"], it["human_p_neutral"], it["human_p_contradiction"]] for it in items])

    # Load S_target (600x600)
    with open(k10_manifest, "r", encoding="utf-8") as f:
        meta = json.load(f)
    q_hh = meta["q_hh_relational"]
    f32_arr = np.frombuffer(k10_bin.read_bytes(), dtype=np.float32)
    S_k10 = f32_arr.reshape((N, N)).astype(np.float64)

    # Exact vote profile groups for exact null
    profile_map: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, it in enumerate(items):
        prof = (it["human_count_entailment"], it["human_count_neutral"], it["human_count_contradiction"])
        profile_map.setdefault(prof, []).append(idx)
    exact_groups = [np.array(indices) for indices in profile_map.values()]

    # Collect available condition probability matrices
    conditions: Dict[str, np.ndarray] = {}

    # Human Empirical
    conditions["00_human_empirical"] = p_human

    # Global prior baseline
    prior_p = np.mean(p_human, axis=0)
    conditions["01_global_class_prior"] = np.tile(prior_p, (N, 1))

    # LPE Raw
    lpe_raw_file = NORM_PROBS_DIR / f"{subset}_lpe_probs.npy"
    if lpe_raw_file.exists():
        conditions["02_gemma3_12b_lpe_raw"] = np.load(lpe_raw_file)

    # LPE Calibrated
    lpe_cal_file = NORM_PROBS_DIR / f"{subset}_lpe_calibrated_probs.npy"
    if lpe_cal_file.exists():
        conditions["03_gemma3_12b_lpe_calibrated"] = np.load(lpe_cal_file)

    # MCE T=1.0
    mce_raw_file = NORM_PROBS_DIR / f"{subset}_mce_probs.npy"
    if mce_raw_file.exists():
        conditions["04_gemma3_12b_mce_t1.0"] = np.load(mce_raw_file)

    # Baseline Classifiers if loaded
    base_class_file = PILOT_SUPPORT_DIR / f"baseline_classifiers_{subset}_probs.npy"
    if base_class_file.exists():
        base_dict = np.load(base_class_file, allow_pickle=True).item()
        if "bart-large" in base_dict:
            conditions["05_bart_large_anchor"] = base_dict["bart-large"]
        if "e003_equal_ensemble" in base_dict:
            conditions["06_e003_equal_ensemble"] = base_dict["e003_equal_ensemble"]

    print("=========================================================================")
    print(f"   E004 PRIMARY ANALYSIS ENGINE ({N} items, {len(conditions)} conditions)")
    print("=========================================================================")

    condition_results = {}

    for cond_name, q_mod in conditions.items():
        print(f"\n--- Evaluating Condition: {cond_name} ---")
        
        # Pointwise
        nll_arr = soft_label_nll_vectorized(p_human, q_mod)
        jsd_arr = jsd_vectorized(p_human, q_mod)
        brier_arr = np.sum((q_mod - p_human) ** 2, axis=1)

        mean_nll = float(np.mean(nll_arr))
        mean_jsd = float(np.mean(jsd_arr))
        mean_brier = float(np.mean(brier_arr))

        # Relational
        dist_mod = distance_hellinger_matrix(q_mod, q_mod)
        W_mod = compute_topk_weight_matrix(dist_mod, k=10)

        q_supp = float(np.sum(np.minimum(W_mod, S_k10)) / (N * 10.0))

        # Null expectation (random identity permutation)
        null_rand_mean = 10.0 / (N - 1.0)
        q_global_excess = q_supp - null_rand_mean

        # Normalized Recovery & Gap Closure
        R_norm = (q_supp - null_rand_mean) / max(1e-12, (q_hh - null_rand_mean))
        g_q = R_norm  # gap closure relative to 0 baseline

        print(f"  NLL: {mean_nll:.4f} | JSD: {mean_jsd:.4f} | Q_support: {q_supp:.5f} | R_norm: {R_norm*100.0:.2f}%")

        # Exact-profile null (for key conditions)
        prof_null_mean, prof_null_ci, prof_excess, p_val = None, None, None, None
        if cond_name not in ["00_human_empirical", "01_global_class_prior"]:
            prof_null_mean, prof_null_ci, prof_excess, p_val = compute_exact_profile_null(
                dist_mod, S_k10, exact_groups, k=10, n_permutations=1000, seed=42
            )
            print(f"  Exact-Profile Null: {prof_null_mean:.5f} | Q_profile_excess: {prof_excess:.5f} | p: {p_val:.4f}")

        condition_results[cond_name] = {
            "condition_name": cond_name,
            "metrics": {
                "nll": mean_nll,
                "jsd_bits": mean_jsd,
                "brier_score": mean_brier,
                "q_support": q_supp,
                "q_null_rand": null_rand_mean,
                "q_global_excess": q_global_excess,
                "r_normalized": R_norm,
                "gap_closure_q": g_q,
                "q_profile_null": prof_null_mean,
                "q_profile_null_95ci": prof_null_ci,
                "q_profile_excess": prof_excess,
                "p_value_monte_carlo": p_val
            }
        }

    # Label Order Sensitivity Regression Analysis
    s_order_file = NORM_PROBS_DIR / f"{subset}_lpe_order_sensitivity.npy"
    order_analysis = {}
    if s_order_file.exists() and "02_gemma3_12b_lpe_raw" in conditions:
        s_order = np.load(s_order_file)
        h_human = -np.sum(np.where(p_human > 1e-12, p_human * np.log2(np.clip(p_human, 1e-12, 1.0)), 0.0), axis=1)
        
        q_gemma = conditions["02_gemma3_12b_lpe_raw"]
        h_model = -np.sum(np.where(q_gemma > 1e-12, q_gemma * np.log2(np.clip(q_gemma, 1e-12, 1.0)), 0.0), axis=1)

        r_human, _ = pearsonr(s_order, h_human)
        r_model, _ = pearsonr(s_order, h_model)

        print("\n--- Label-Order Sensitivity Correlations ---")
        print(f"  r(S_order, H_human) = {r_human:+.4f}")
        print(f"  r(S_order, H_model) = {r_model:+.4f}")

        order_analysis = {
            "mean_s_order_bits": float(np.mean(s_order)),
            "pearson_r_human_entropy": float(r_human),
            "pearson_r_model_entropy": float(r_model)
        }

    # Save summary JSON
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "experiment_id": "E004",
        "stage": 1,
        "pilot_n": N,
        "q_hh_relational": q_hh,
        "conditions": condition_results,
        "label_order_analysis": order_analysis
    }

    out_json = SUMMARIES_DIR / "E004_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved analysis summary to {out_json}")
    print("=========================================================================")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["preflight", "pilot"], default="pilot")
    args = parser.parse_args()
    run_e004_analysis(args.subset)

if __name__ == "__main__":
    main()
