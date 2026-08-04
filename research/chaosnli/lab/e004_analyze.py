"""E004 Primary Analysis Engine.

Computes pointwise metrics, coherent fold-specific relational metrics, dataset-stratified nulls (10,000 perms),
exact-profile informativeness diagnostics, label-mapping sensitivity, Delta R, and G_Q(cal<-raw).
Outputs results to research/chaosnli/artifacts/E004/summaries/E004_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

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
    """Compute pointwise JSD in bits (no square root)."""
    m = 0.5 * (p + q)
    m = np.clip(m, 1e-12, 1.0)
    p_safe = np.clip(p, 1e-12, 1.0)
    q_safe = np.clip(q, 1e-12, 1.0)
    kl_pm = np.sum(np.where(p > 1e-12, p * np.log2(p_safe / m), 0.0), axis=1)
    kl_qm = np.sum(np.where(q > 1e-12, q * np.log2(q_safe / m), 0.0), axis=1)
    return np.maximum(0.0, 0.5 * kl_pm + 0.5 * kl_qm)

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

def build_stratified_folds(items: List[Dict], n_folds: int = 5, seed: int = 20260803) -> List[np.ndarray]:
    """Build 5 stratified fold indices by (source_dataset, majority_label)."""
    p_human = np.array([[it["human_p_entailment"], it["human_p_neutral"], it["human_p_contradiction"]] for it in items])
    datasets = [it["source_dataset"] for it in items]
    majority = np.argmax(p_human, axis=1)

    strata_keys = [f"{d}_{m}" for d, m in zip(datasets, majority)]

    rng = np.random.default_rng(seed)
    strata_map: Dict[str, List[int]] = {}
    for idx, key in enumerate(strata_keys):
        strata_map.setdefault(key, []).append(idx)

    folds = [[] for _ in range(n_folds)]
    for key, indices in sorted(strata_map.items()):
        shuffled = rng.permutation(indices)
        for i, idx in enumerate(shuffled):
            folds[i % n_folds].append(idx)

    return [np.array(sorted(fold), dtype=np.int64) for fold in folds]

def compute_stratified_null(
    dist_model: np.ndarray,
    S_target: np.ndarray,
    datasets: List[str],
    k: int = 10,
    n_permutations: int = 10000,
    seed: int = 42
) -> Tuple[float, List[float]]:
    """10,000 dataset-stratified (SNLI/MNLI) identity permutations."""
    rng = np.random.default_rng(seed)
    N = len(datasets)
    snli_idx = np.where(np.array([d.endswith("snli") for d in datasets]))[0]
    mnli_idx = np.where(np.array([d.endswith("mnli") for d in datasets]))[0]

    W_model = compute_topk_weight_matrix(dist_model, k)
    null_scores = []

    for _ in range(n_permutations):
        perm_idx = np.arange(N)
        if len(snli_idx) > 1:
            perm_idx[snli_idx] = rng.permutation(snli_idx)
        if len(mnli_idx) > 1:
            perm_idx[mnli_idx] = rng.permutation(mnli_idx)

        W_perm = W_model[perm_idx][:, perm_idx]
        score = float(np.sum(W_perm * S_target) / (N * float(k)))
        null_scores.append(score)

    null_mean = float(np.mean(null_scores))
    null_ci = [float(np.percentile(null_scores, 2.5)), float(np.percentile(null_scores, 97.5))]
    return null_mean, null_ci

def compute_exact_profile_null(
    dist_model: np.ndarray,
    S_target: np.ndarray,
    exact_groups: List[np.ndarray],
    k: int = 10,
    n_permutations: int = 10000,
    seed: int = 42
) -> Tuple[float, List[float], float, float]:
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
        score = float(np.sum(W_perm * S_target) / (N * float(k)))
        null_scores.append(score)

    q_support = float(np.sum(W_model * S_target) / (N * float(k)))
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
    datasets = [it["source_dataset"] for it in items]
    p_human = np.array([[it["human_p_entailment"], it["human_p_neutral"], it["human_p_contradiction"]] for it in items])

    # Load S_target (N x N)
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

    # Exact-Profile Diagnostics
    n_exact_groups = len(exact_groups)
    non_singleton_groups = [g for g in exact_groups if len(g) > 1]
    n_non_singleton_groups = len(non_singleton_groups)
    n_items_in_non_singletons = sum(len(g) for g in non_singleton_groups)
    max_group_size = max(len(g) for g in exact_groups) if exact_groups else 0
    is_exact_informative = (n_items_in_non_singletons > 0)

    exact_diagnostics = {
        "n_exact_groups": n_exact_groups,
        "n_non_singleton_groups": n_non_singleton_groups,
        "n_items_in_non_singletons": n_items_in_non_singletons,
        "max_group_size": max_group_size,
        "is_informative": is_exact_informative
    }

    # Collect available condition probability matrices
    conditions: Dict[str, np.ndarray] = {}
    conditions["00_human_empirical"] = p_human

    prior_p = np.mean(p_human, axis=0)
    conditions["01_global_class_prior"] = np.tile(prior_p, (N, 1))

    lpe_raw_file = NORM_PROBS_DIR / f"{subset}_lpe_probs.npy"
    if lpe_raw_file.exists():
        conditions["02_gemma3_12b_lpe_raw"] = np.load(lpe_raw_file)

    lpe_cal_file = NORM_PROBS_DIR / f"{subset}_lpe_calibrated_probs.npy"
    if lpe_cal_file.exists():
        conditions["03_gemma3_12b_lpe_calibrated"] = np.load(lpe_cal_file)

    mce_raw_file = NORM_PROBS_DIR / f"{subset}_mce_probs.npy"
    if mce_raw_file.exists():
        conditions["04_gemma3_12b_mce_t1.0"] = np.load(mce_raw_file)

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

    # First pass: compute raw Gemma baseline R for relative G_Q calculations
    q_raw_mod = conditions.get("02_gemma3_12b_lpe_raw")
    r_raw_baseline = None
    if q_raw_mod is not None:
        dist_raw = distance_hellinger_matrix(q_raw_mod, q_raw_mod)
        q_supp_raw = float(np.sum(compute_topk_weight_matrix(dist_raw, k=10) * S_k10) / (N * 10.0))
        q_null_raw, _ = compute_stratified_null(dist_raw, S_k10, datasets, k=10, n_permutations=10000, seed=42)
        r_raw_baseline = (q_supp_raw - q_null_raw) / max(1e-12, (q_hh - q_null_raw))

    condition_results = {}

    for cond_name, q_mod in conditions.items():
        print(f"\n--- Evaluating Condition: {cond_name} ---")

        # Pointwise metrics
        nll_arr = soft_label_nll_vectorized(p_human, q_mod)
        jsd_arr = jsd_vectorized(p_human, q_mod)
        brier_arr = np.sum((q_mod - p_human) ** 2, axis=1)

        mean_nll = float(np.mean(nll_arr))
        mean_jsd = float(np.mean(jsd_arr))
        mean_brier = float(np.mean(brier_arr))

        # Relational metrics
        dist_mod_for_null = distance_hellinger_matrix(q_mod, q_mod)

        if cond_name == "03_gemma3_12b_lpe_calibrated":
            # Coherent fold-specific cross-fitted relational scoring
            fold_full_file = NORM_PROBS_DIR / f"{subset}_lpe_fold_calibrated_probs.npy"
            if fold_full_file.exists():
                fold_full_probs = np.load(fold_full_file) # (5, N, 3)
                folds = build_stratified_folds(items, n_folds=5)

                fold_scores = []
                fold_null_scores = []
                rng_null = np.random.default_rng(42)
                snli_idx = np.where(np.array([d.endswith("snli") for d in datasets]))[0]
                mnli_idx = np.where(np.array([d.endswith("mnli") for d in datasets]))[0]

                for f_i in range(5):
                    val_idx = folds[f_i]
                    q_f = fold_full_probs[f_i]
                    dist_f = distance_hellinger_matrix(q_f, q_f)
                    W_f = compute_topk_weight_matrix(dist_f, k=10)

                    # Score held-out focal rows in val_idx
                    score_f = float(np.sum(W_f[val_idx] * S_k10[val_idx]) / (len(val_idx) * 10.0))
                    fold_scores.append(score_f)

                    # 2,000 dataset-stratified nulls per fold (10,000 total across 5 folds)
                    for _ in range(2000):
                        perm_idx = np.arange(N)
                        if len(snli_idx) > 1:
                            perm_idx[snli_idx] = rng_null.permutation(snli_idx)
                        if len(mnli_idx) > 1:
                            perm_idx[mnli_idx] = rng_null.permutation(mnli_idx)
                        W_p = W_f[perm_idx][:, perm_idx]
                        fold_null_scores.append(float(np.sum(W_p[val_idx] * S_k10[val_idx]) / (len(val_idx) * 10.0)))

                q_supp = float(np.mean(fold_scores))
                null_mean = float(np.mean(fold_null_scores))
                null_ci = [float(np.percentile(fold_null_scores, 2.5)), float(np.percentile(fold_null_scores, 97.5))]
            else:
                W_mod = compute_topk_weight_matrix(dist_mod_for_null, k=10)
                q_supp = float(np.sum(W_mod * S_k10) / (N * 10.0))
                null_mean, null_ci = compute_stratified_null(dist_mod_for_null, S_k10, datasets, k=10, n_permutations=10000, seed=42)
        else:
            W_mod = compute_topk_weight_matrix(dist_mod_for_null, k=10)
            q_supp = float(np.sum(W_mod * S_k10) / (N * 10.0))
            null_mean, null_ci = compute_stratified_null(dist_mod_for_null, S_k10, datasets, k=10, n_permutations=10000, seed=42)

        q_global_excess = q_supp - null_mean

        # Normalized Recovery R = (Q_model - Q_null) / (Q_HH - Q_null)
        R_norm = (q_supp - null_mean) / max(1e-12, (q_hh - null_mean))

        # Delta R and Gap Closure G_Q relative to raw Gemma baseline
        delta_r = None
        g_q = None
        if cond_name == "03_gemma3_12b_lpe_calibrated" and r_raw_baseline is not None:
            delta_r = R_norm - r_raw_baseline
            g_q = (R_norm - r_raw_baseline) / max(1e-12, (1.0 - r_raw_baseline))

        print(f"  NLL: {mean_nll:.4f} | JSD: {mean_jsd:.4f} | Q_support: {q_supp:.5f} | Q_null: {null_mean:.5f} | R_norm: {R_norm*100.0:.2f}%")
        if delta_r is not None:
            print(f"  Delta R: {delta_r*100.0:+.2f}% | Gap Closure G_Q(cal<-raw): {g_q*100.0:+.2f}%")

        # Exact-profile null
        prof_null_mean, prof_null_ci, prof_excess, p_val = None, None, None, None
        exact_status = "evaluated" if is_exact_informative else "non_informative_preflight"

        if is_exact_informative and cond_name not in ["00_human_empirical", "01_global_class_prior"]:
            prof_null_mean, prof_null_ci, prof_excess, p_val = compute_exact_profile_null(
                dist_mod_for_null, S_k10, exact_groups, k=10, n_permutations=10000, seed=42
            )
            print(f"  Exact-Profile Null: {prof_null_mean:.5f} | Q_profile_excess: {prof_excess:.5f} | p: {p_val:.4f}")
        elif not is_exact_informative:
            print(f"  Exact-Profile Null: non_informative_preflight (0 non-singleton profile groups)")

        condition_results[cond_name] = {
            "condition_name": cond_name,
            "metrics": {
                "nll": mean_nll,
                "jsd_bits": mean_jsd,
                "brier_score": mean_brier,
                "q_support": q_supp,
                "q_null_stratified": null_mean,
                "q_null_stratified_95ci": null_ci,
                "q_global_excess": q_global_excess,
                "r_normalized": R_norm,
                "delta_r_vs_raw": delta_r,
                "gap_closure_gq_vs_raw": g_q,
                "exact_profile_status": exact_status,
                "q_profile_null": prof_null_mean,
                "q_profile_null_95ci": prof_null_ci,
                "q_profile_excess": prof_excess,
                "p_value_monte_carlo": p_val
            }
        }

    # Label-Mapping Sensitivity Analysis
    s_order_file = NORM_PROBS_DIR / f"{subset}_lpe_order_sensitivity.npy"
    order_analysis = {}
    if s_order_file.exists() and "02_gemma3_12b_lpe_raw" in conditions:
        s_order = np.load(s_order_file)
        h_human = -np.sum(np.where(p_human > 1e-12, p_human * np.log2(np.clip(p_human, 1e-12, 1.0)), 0.0), axis=1)

        q_gemma = conditions["02_gemma3_12b_lpe_raw"]
        h_model = -np.sum(np.where(q_gemma > 1e-12, q_gemma * np.log2(np.clip(q_gemma, 1e-12, 1.0)), 0.0), axis=1)

        r_human, _ = pearsonr(s_order, h_human)
        r_model, _ = pearsonr(s_order, h_model)

        print("\n--- Label-Mapping Sensitivity Correlations ---")
        print(f"  Mean S_mapping = {np.mean(s_order):.6f} bits")
        print(f"  r(S_mapping, H_human) = {r_human:+.4f}")
        print(f"  r(S_mapping, H_model) = {r_model:+.4f}")

        order_analysis = {
            "mean_s_mapping_bits": float(np.mean(s_order)),
            "pearson_r_human_entropy": float(r_human),
            "pearson_r_model_entropy": float(r_model),
            "note": "Measures sensitivity to symbol mapping (A/B/C permuted to E/N/C), not item order in reordered lists."
        }

    # Save summary JSON
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "experiment_id": "E004",
        "stage": 1,
        "pilot_n": N,
        "q_hh_relational": q_hh,
        "exact_profile_diagnostics": exact_diagnostics,
        "conditions": condition_results,
        "label_mapping_analysis": order_analysis
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
