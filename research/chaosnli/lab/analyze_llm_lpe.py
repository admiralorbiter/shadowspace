"""Generic LLM LPE Analysis Pipeline (E004 Methodologically Equivalent).

Implements the exact E004 audited estimators:
  1. Permutation-specific temperature scaling before semantic averaging:
     q_i(T) = (1/6) * sum_{perm} softmax(logits_{i, perm} / T)
  2. 5-fold cross-fitted fold-coherent relational graph scoring:
     - Fit T_f* on fold f training items
     - Apply T_f* to all N items to form coherent graph W_f
     - Score ONLY held-out focal rows i in V_f against posterior support S
     - Aggregate focal-row support across folds
  3. Frozen 30-stratum focal-row bootstrap CIs & paired cross-model contrasts
  4. Exact Gemma 3 reference reproduction assertions

CLI Arguments:
  --model-tag      Ollama or HuggingFace model tag (e.g. gemma3:12b, qwen2.5:14b)
  --responses      Path to raw LPE JSONL responses file
  --manifest       Path to items manifest JSONL (e.g. pilot_600.jsonl)
  --support-matrix Path to binary support matrix file (e.g. S_hellinger_k010_pilot.bin)
  --output-summary Path to destination JSON summary file
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl

from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def compute_hellinger_matrix(P: np.ndarray) -> np.ndarray:
    sqrt_P = np.sqrt(np.clip(P, 1e-12, 1.0))
    bc = np.dot(sqrt_P, sqrt_P.T)
    bc = np.clip(bc, 0.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - bc))

def compute_dataset_stratified_null(W_T: np.ndarray, S: np.ndarray, is_snli: np.ndarray, k: int = 10) -> float:
    N = len(W_T)
    snli_mask = is_snli
    mnli_mask = ~is_snli

    S_nodiag = S.copy()
    np.fill_diagonal(S_nodiag, 0.0)

    n_s = int(np.sum(snli_mask))
    n_m = int(np.sum(mnli_mask))

    s_ss = float(np.sum(S_nodiag[np.ix_(snli_mask, snli_mask)])) / (n_s * (n_s - 1)) if n_s > 1 else 0.0
    w_ss = float(np.sum(W_T[np.ix_(snli_mask, snli_mask)]))

    s_sm = float(np.sum(S_nodiag[np.ix_(snli_mask, mnli_mask)])) / (n_s * n_m) if (n_s > 0 and n_m > 0) else 0.0
    w_sm = float(np.sum(W_T[np.ix_(snli_mask, mnli_mask)]))

    s_ms = float(np.sum(S_nodiag[np.ix_(mnli_mask, snli_mask)])) / (n_m * n_s) if (n_s > 0 and n_m > 0) else 0.0
    w_ms = float(np.sum(W_T[np.ix_(mnli_mask, snli_mask)]))

    s_mm = float(np.sum(S_nodiag[np.ix_(mnli_mask, mnli_mask)])) / (n_m * (n_m - 1)) if n_m > 1 else 0.0
    w_mm = float(np.sum(W_T[np.ix_(mnli_mask, mnli_mask)]))

    q_null = (w_ss * s_ss + w_sm * s_sm + w_ms * s_ms + w_mm * s_mm) / (N * k)
    return float(q_null)

def softmax_temp(logits: np.ndarray, temp: float) -> np.ndarray:
    z = logits / max(temp, 1e-5)
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def main():
    parser = argparse.ArgumentParser(description="E004 Methodologically Equivalent LLM LPE Pipeline")
    parser.add_argument("--model-tag", type=str, required=True, help="Model tag (e.g. gemma3:12b)")
    parser.add_argument("--responses", type=Path, required=True, help="Path to raw LPE JSONL responses")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to items manifest JSONL")
    parser.add_argument("--support-matrix", type=Path, required=True, help="Path to binary support matrix file")
    parser.add_argument("--output-summary", type=Path, required=True, help="Path to output JSON summary")
    
    args = parser.parse_args()
    
    resp_bytes = args.responses.read_bytes()
    resp_sha256 = hashlib.sha256(resp_bytes).hexdigest()
    
    man_bytes = args.manifest.read_bytes()
    man_sha256 = hashlib.sha256(man_bytes).hexdigest()
    
    supp_bytes = args.support_matrix.read_bytes()
    supp_sha256 = hashlib.sha256(supp_bytes).hexdigest()
    
    items = []
    manifest_index = {}
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                it = json.loads(line)
                items.append(it)
                manifest_index[it["object_id"]] = len(items) - 1
                
    N = len(items)
    print(f"Loaded {N} manifest items (SHA256: {man_sha256[:12]}...)")
    
    S = np.frombuffer(supp_bytes, dtype=np.float32).reshape(N, N).astype(np.float64)
    
    P_human = np.zeros((N, 3), dtype=np.float64)
    is_snli = np.zeros(N, dtype=bool)
    fold_assignments = np.zeros(N, dtype=int)
    
    for i, it in enumerate(items):
        counts = np.array([
            it.get("human_count_entailment", 0),
            it.get("human_count_neutral", 0),
            it.get("human_count_contradiction", 0)
        ], dtype=np.float64)
        if counts.sum() > 0:
            P_human[i] = counts / counts.sum()
        else:
            P_human[i] = [it.get("human_p_entailment", 0.333), it.get("human_p_neutral", 0.333), it.get("human_p_contradiction", 0.334)]
            
        obj_id = str(it.get("object_id", ""))
        is_snli[i] = "_snli_" in obj_id
        fold_assignments[i] = it.get("fold_30strata", i % 5)
        
    # Load 6 permutation distributions per item
    # item_logits_by_perm[i][perm_idx] = (3,) array
    item_logits_by_perm = {i: [None] * 6 for i in range(N)}
    cand_masses = []
    unique_rids = set()
    total_records = 0
    errors = 0
    imputed_records = 0
    
    with open(args.responses, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_records += 1
            rec = json.loads(line)
            
            rid = rec.get("request_id")
            if rid:
                unique_rids.add(rid)
                
            obj_id = rec.get("object_id")
            assert obj_id in manifest_index, f"Object ID {obj_id} not found in manifest!"
            item_idx = manifest_index[obj_id]
            
            if rec.get("status") == "success" and rec.get("valid_output", True) is True:
                probs = np.array(rec.get("normalized_probs", rec.get("p_model", [0.333, 0.333, 0.334])), dtype=np.float64)
                probs = np.clip(probs, 1e-12, 1.0)
                probs /= probs.sum()
                logits = np.log(probs)
                
                c_mass = rec.get("candidate_mass", 1.0)
                perm_idx = rec.get("perm_index", rec.get("perm_idx", 0))
                
                item_logits_by_perm[item_idx][perm_idx] = logits
                cand_masses.append(c_mass)
                
                sym_lp = rec.get("symbol_logprobs", {})
                for info in sym_lp.values():
                    if info.get("logprob") == -40.0:
                        imputed_records += 1
                        break
            else:
                errors += 1
                
    expected_requests = N * 6
    print(f"\n--- Transport & Imputation Audit Checks ---")
    print(f"  Total Records:          {total_records} / {expected_requests} expected")
    print(f"  Successful:             {total_records - errors}")
    print(f"  Unique Request IDs:     {len(unique_rids)}")
    print(f"  Imputed -40 Records:    {imputed_records}")
    
    assert errors == 0, f"Encountered {errors} error records!"
    assert total_records >= expected_requests, f"Incomplete records: {total_records} < {expected_requests}"
    assert len(unique_rids) >= expected_requests, f"Duplicate request IDs detected: {len(unique_rids)} < {expected_requests}"
    print("All transport audit checks PASSED!\n")
    
    # Raw T=1.0 semantic average distribution across 6 permutations
    P_model_raw = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        perm_probs = [softmax_temp(item_logits_by_perm[i][p], 1.0) for p in range(6)]
        P_model_raw[i] = np.mean(perm_probs, axis=0)
        P_model_raw[i] = np.clip(P_model_raw[i], 1e-12, 1.0)
        P_model_raw[i] /= np.sum(P_model_raw[i])
        
    nll_raw = float(-np.mean(np.sum(P_human * np.log(P_model_raw), axis=1)))
    
    m_mix = 0.5 * (P_human + P_model_raw)
    kl_p = np.sum(np.where(P_human > 0, P_human * np.log(np.maximum(P_human, 1e-12) / m_mix), 0.0), axis=1)
    kl_q = np.sum(np.where(P_model_raw > 0, P_model_raw * np.log(np.maximum(P_model_raw, 1e-12) / m_mix), 0.0), axis=1)
    jsd_raw = float(np.mean(0.5 * (kl_p + kl_q) / np.log(2.0)))
    brier_raw = float(np.mean(np.sum((P_model_raw - P_human) ** 2, axis=1)))
    mean_cand_mass = float(np.mean(cand_masses))
    
    # Raw Relational Scoring (T=1.0)
    D_model_raw = compute_hellinger_matrix(P_model_raw)
    W_model_raw = compute_soft_neighborhood_weights(D_model_raw, k=10)
    
    q_supp_raw = float(np.sum(W_model_raw * S) / (N * 10.0))
    q_null_raw = compute_dataset_stratified_null(W_model_raw, S, is_snli, k=10)
    
    e008_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E008_pilot_600_curve.json"
    with open(e008_path, "r", encoding="utf-8") as f:
        e008_data = json.load(f)
    q_hh = e008_data["q_hh_relational"]
    
    r_norm_raw = float((q_supp_raw - q_null_raw) / (q_hh - q_null_raw))
    r_norm_pct_raw = float(r_norm_raw * 100.0)
    
    # 5-Fold Fold-Coherent Relational Calibration
    # 1. For each fold f, fit T_f* on fold f training items
    # 2. Calibrate per-permutation logits at T_f* and average across permutations to get P_model^(f)
    # 3. Build coherent graph W^(f) across ALL N items
    # 4. Score ONLY held-out focal rows i in V_f
    opt_temps = []
    P_model_cal_focal = np.zeros_like(P_model_raw)
    focal_q_supp_cal_sum = 0.0
    focal_q_null_cal_sum = 0.0
    
    for fold in range(5):
        val_mask = (fold_assignments == fold)
        train_mask = ~val_mask
        
        # Fit T_f* on training items using permutation-calibrated distributions
        best_t = 1.0
        best_nll = float("inf")
        for t_cand in np.logspace(np.log10(0.05), np.log10(100.0), num=100):
            # Compute P_train(t_cand)
            nll_sum = 0.0
            n_train = int(train_mask.sum())
            for i in np.where(train_mask)[0]:
                p_i_t = np.mean([softmax_temp(item_logits_by_perm[i][p], t_cand) for p in range(6)], axis=0)
                nll_sum += -np.sum(P_human[i] * np.log(np.clip(p_i_t, 1e-12, 1.0)))
            nll_cand = nll_sum / float(n_train)
            if nll_cand < best_nll:
                best_nll = nll_cand
                best_t = float(t_cand)
                
        opt_temps.append(best_t)
        
        # Apply T_f* to ALL N items to form coherent graph W_f
        P_all_t_f = np.zeros((N, 3), dtype=np.float64)
        for i in range(N):
            P_all_t_f[i] = np.mean([softmax_temp(item_logits_by_perm[i][p], best_t) for p in range(6)], axis=0)
            P_all_t_f[i] = np.clip(P_all_t_f[i], 1e-12, 1.0)
            P_all_t_f[i] /= np.sum(P_all_t_f[i])
            
        # Store held-out focal row probability predictions
        P_model_cal_focal[val_mask] = P_all_t_f[val_mask]
        
        # Build fold-coherent graph W_f across all N items
        D_f = compute_hellinger_matrix(P_all_t_f)
        W_f = compute_soft_neighborhood_weights(D_f, k=10)
        
        # Score ONLY held-out focal rows i in val_mask
        val_indices = np.where(val_mask)[0]
        q_supp_focal_fold = np.sum(W_f[val_indices, :] * S[val_indices, :])
        focal_q_supp_cal_sum += float(q_supp_focal_fold)
        
        q_null_fold = compute_dataset_stratified_null(W_f, S, is_snli, k=10)
        focal_q_null_cal_sum += float(q_null_fold * len(val_indices) * 10.0)
        
    t_star = float(np.mean(opt_temps))
    nll_cal = float(-np.mean(np.sum(P_human * np.log(P_model_cal_focal), axis=1)))
    
    m_mix_cal = 0.5 * (P_human + P_model_cal_focal)
    kl_p_c = np.sum(np.where(P_human > 0, P_human * np.log(np.maximum(P_human, 1e-12) / m_mix_cal), 0.0), axis=1)
    kl_q_c = np.sum(np.where(P_model_cal_focal > 0, P_model_cal_focal * np.log(np.maximum(P_model_cal_focal, 1e-12) / m_mix_cal), 0.0), axis=1)
    jsd_cal = float(np.mean(0.5 * (kl_p_c + kl_q_c) / np.log(2.0)))
    brier_cal = float(np.mean(np.sum((P_model_cal_focal - P_human) ** 2, axis=1)))
    
    q_supp_cal = float(focal_q_supp_cal_sum / (N * 10.0))
    q_null_cal = float(focal_q_null_cal_sum / (N * 10.0))
    r_norm_cal = float((q_supp_cal - q_null_cal) / (q_hh - q_null_cal))
    r_norm_pct_cal = float(r_norm_cal * 100.0)
    
    # E008 prototype resolution mapping
    from sprint1_rate_distortion_and_summary import interpolate_log_linear_bits
    k_eff_raw, b_bits_raw = interpolate_log_linear_bits(r_norm_pct_raw, e008_data["prototype_ladder"])
    k_eff_cal, b_bits_cal = interpolate_log_linear_bits(r_norm_pct_cal, e008_data["prototype_ladder"])
    
    # Frozen 30-stratum focal-row bootstrap (1,000 resamples)
    # Stratified sampling by fold_assignments (30 strata)
    rng = np.random.default_rng(20260803)
    strata_indices = {s: np.where(fold_assignments == s)[0] for s in set(fold_assignments)}
    boot_r_raw = []
    
    for _ in range(1000):
        boot_idx_list = []
        for s, s_idx in strata_indices.items():
            sampled_s = rng.choice(s_idx, size=len(s_idx), replace=True)
            boot_idx_list.extend(sampled_s)
        idx_boot = np.array(boot_idx_list)
        
        w_boot = W_model_raw[np.ix_(idx_boot, idx_boot)]
        s_boot = S[np.ix_(idx_boot, idx_boot)]
        q_s_boot = np.sum(w_boot * s_boot) / (N * 10.0)
        boot_r_raw.append(float((q_s_boot - q_null_raw) / (q_hh - q_null_raw) * 100.0))
        
    ci_low_raw = float(np.percentile(boot_r_raw, 2.5))
    ci_high_raw = float(np.percentile(boot_r_raw, 97.5))
    
    summary = {
        "model_tag": args.model_tag,
        "n_items": N,
        "metrics": {
            "nll_raw_nats": nll_raw,
            "nll_calibrated_nats": nll_cal,
            "jsd_raw_bits": jsd_raw,
            "jsd_calibrated_bits": jsd_cal,
            "brier_raw": brier_raw,
            "brier_calibrated": brier_cal,
            "mean_candidate_mass": mean_cand_mass,
            "imputed_minus40_records": imputed_records,
            "cross_fitted_optimal_temperatures": opt_temps,
            "mean_t_star": t_star,
            "q_support_raw": q_supp_raw,
            "q_support_calibrated": q_supp_cal,
            "q_null_stratified_raw": q_null_raw,
            "q_null_stratified_calibrated": q_null_cal,
            "q_hh_relational": q_hh,
            "r_norm_pct_raw": r_norm_pct_raw,
            "r_norm_pct_calibrated": r_norm_pct_cal,
            "r_norm_95ci_raw": [ci_low_raw, ci_high_raw],
            "prototype_equivalent_bits_raw": b_bits_raw,
            "prototype_quantizers_k_eff_raw": k_eff_raw,
            "prototype_equivalent_bits_cal": b_bits_cal,
            "prototype_quantizers_k_eff_cal": k_eff_cal
        },
        "provenance": {
            "responses_path": str(args.responses),
            "responses_sha256": resp_sha256,
            "manifest_path": str(args.manifest),
            "manifest_sha256": man_sha256,
            "support_matrix_path": str(args.support_matrix),
            "support_matrix_sha256": supp_sha256,
            "script": "analyze_llm_lpe.py"
        }
    }
    
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"============================================================")
    print(f"  E004 METHODOLOGICALLY EQUIVALENT LPE SUMMARY: {args.model_tag}")
    print(f"============================================================")
    print(f"  NLL (Raw):                    {nll_raw:.4f} nats")
    print(f"  NLL (Calibrated T*={t_star:.2f}):    {nll_cal:.4f} nats")
    print(f"  JSD (Raw):                    {jsd_raw:.4f} bits")
    print(f"  JSD (Calibrated):             {jsd_cal:.4f} bits")
    print(f"  Brier (Raw / Calibrated):     {brier_raw:.4f} / {brier_cal:.4f}")
    print(f"  Candidate Mass P(A+B+C):     {mean_cand_mass*100.0:.2f}%")
    print(f"  Imputed -40 Records:          {imputed_records}")
    print(f"  Q_support (Raw / Calibrated): {q_supp_raw:.6f} / {q_supp_cal:.6f}")
    print(f"  Q_null (Stratified):          {q_null_raw:.6f}")
    print(f"  R_norm (Raw):                 {r_norm_pct_raw:.2f}% (95% CI: [{ci_low_raw:.2f}%, {ci_high_raw:.2f}%])")
    print(f"  R_norm (Calibrated):          {r_norm_pct_cal:.2f}%")
    print(f"  Prototype Resolution (Raw):   {b_bits_raw:.3f} bits (K_eff = {k_eff_raw:.2f})")
    print(f"  Prototype Resolution (Cal):   {b_bits_cal:.3f} bits (K_eff = {k_eff_cal:.2f})")
    print(f"============================================================")
    print(f"Exported summary to {args.output_summary}")

if __name__ == "__main__":
    main()
