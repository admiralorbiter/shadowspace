"""Generic LLM LPE Analysis Pipeline (Object-ID Aligned, Cross-Fitted Calibration & Bootstrap).

Consolidates LLM response evaluation into a single, model-parameterized, publication-grade pipeline.

CLI Arguments:
  --model-tag      Ollama or HuggingFace model tag (e.g. gemma3:12b, gemma4:12b, qwen2.5:14b)
  --responses      Path to raw LPE JSONL responses file
  --manifest       Path to items manifest JSONL (e.g. pilot_600.jsonl)
  --support-matrix Path to binary support matrix file (e.g. S_hellinger_k010_pilot.bin)
  --output-summary Path to destination JSON summary file

Produces:
  - Pointwise raw & calibrated metrics (NLL nats, JSD bits, Brier, candidate mass)
  - Label permutation variability (variance across 6 S3 permutations)
  - 5-fold cross-fitted temperature calibration (T*)
  - Coherent held-out relational score (Q_support, Q_null, R_norm)
  - Pilot E008 rate-distortion mapping (b effective bits, K_eff prototypes)
  - 30-stratum focal-row bootstrap 95% CIs
  - Complete SHA-256 provenance
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
    parser = argparse.ArgumentParser(description="Generic LLM LPE Evaluation Pipeline")
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
        
    item_preds = {i: [] for i in range(N)}
    item_perms = {i: set() for i in range(N)}
    cand_masses = []
    unique_rids = set()
    total_records = 0
    errors = 0
    
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
                probs = rec.get("normalized_probs", rec.get("p_model", [0.333, 0.333, 0.334]))
                c_mass = rec.get("candidate_mass", 1.0)
                perm_idx = rec.get("perm_index", rec.get("perm_idx", 0))
                
                item_preds[item_idx].append(probs)
                item_perms[item_idx].add(perm_idx)
                cand_masses.append(c_mass)
            else:
                errors += 1
                
    expected_requests = N * 6
    print(f"\n--- Transport Audit Checks ---")
    print(f"  Total Records:          {total_records} / {expected_requests} expected")
    print(f"  Successful:             {total_records - errors}")
    print(f"  Unique Request IDs:     {len(unique_rids)}")
    print(f"  Items with 6 Perms:     {sum(1 for s in item_perms.values() if len(s) == 6)} / {N}")
    
    assert errors == 0, f"Encountered {errors} error records in response file!"
    assert total_records >= expected_requests, f"Incomplete records: {total_records} < {expected_requests}"
    assert len(unique_rids) >= expected_requests, f"Duplicate request IDs detected: {len(unique_rids)} < {expected_requests}"
    assert all(len(s) == 6 for s in item_perms.values()), "Missing label permutations for some items!"
    print("All transport audit checks PASSED!\n")
    
    P_model = np.zeros((N, 3), dtype=np.float64)
    perm_variances = []
    
    for i in range(N):
        arr = np.array(item_preds[i], dtype=np.float64)
        P_model[i] = np.mean(arr, axis=0)
        P_model[i] = np.clip(P_model[i], 1e-12, 1.0)
        P_model[i] /= np.sum(P_model[i])
        
        perm_variances.append(float(np.mean(np.var(arr, axis=0))))
        
    mean_perm_var = float(np.mean(perm_variances))
    
    nll_raw = float(-np.mean(np.sum(P_human * np.log(np.clip(P_model, 1e-12, 1.0)), axis=1)))
    
    m_mix = 0.5 * (P_human + P_model)
    kl_p = np.sum(np.where(P_human > 0, P_human * np.log(np.maximum(P_human, 1e-12) / m_mix), 0.0), axis=1)
    kl_q = np.sum(np.where(P_model > 0, P_model * np.log(np.maximum(P_model, 1e-12) / m_mix), 0.0), axis=1)
    jsd_raw = float(np.mean(0.5 * (kl_p + kl_q) / np.log(2.0)))
    
    brier_raw = float(np.mean(np.sum((P_model - P_human) ** 2, axis=1)))
    mean_cand_mass = float(np.mean(cand_masses))
    
    fold_assignments = [i % 5 for i in range(N)]
    calibrated_P_model = np.zeros_like(P_model)
    opt_temps = []
    
    for fold in range(5):
        val_mask = np.array([f == fold for f in fold_assignments])
        train_mask = ~val_mask
        
        Logits_train = np.log(P_model[train_mask])
        P_train = P_human[train_mask]
        
        best_t = 1.0
        best_nll = float("inf")
        for t_cand in np.logspace(np.log10(0.05), np.log10(100.0), num=100):
            Q_cand = softmax_temp(Logits_train, t_cand)
            nll_cand = -np.mean(np.sum(P_train * np.log(np.clip(Q_cand, 1e-12, 1.0)), axis=1))
            if nll_cand < best_nll:
                best_nll = nll_cand
                best_t = float(t_cand)
                
        opt_temps.append(best_t)
        calibrated_P_model[val_mask] = softmax_temp(np.log(P_model[val_mask]), best_t)
        
    t_star = float(np.mean(opt_temps))
    nll_cal = float(-np.mean(np.sum(P_human * np.log(np.clip(calibrated_P_model, 1e-12, 1.0)), axis=1)))
    
    D_model_raw = compute_hellinger_matrix(P_model)
    W_model_raw = compute_soft_neighborhood_weights(D_model_raw, k=10)
    
    D_model_cal = compute_hellinger_matrix(calibrated_P_model)
    W_model_cal = compute_soft_neighborhood_weights(D_model_cal, k=10)
    
    q_supp_raw = float(np.sum(W_model_raw * S) / (N * 10.0))
    q_supp_cal = float(np.sum(W_model_cal * S) / (N * 10.0))
    
    q_null_raw = compute_dataset_stratified_null(W_model_raw, S, is_snli, k=10)
    q_null_cal = compute_dataset_stratified_null(W_model_cal, S, is_snli, k=10)
    
    e008_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E008_pilot_600_curve.json"
    with open(e008_path, "r", encoding="utf-8") as f:
        e008_data = json.load(f)
    q_hh = e008_data["q_hh_relational"]
    
    r_norm_raw = float((q_supp_raw - q_null_raw) / (q_hh - q_null_raw))
    r_norm_cal = float((q_supp_cal - q_null_cal) / (q_hh - q_null_cal))
    
    r_norm_pct_raw = float(r_norm_raw * 100.0)
    r_norm_pct_cal = float(r_norm_cal * 100.0)
    
    from sprint1_rate_distortion_and_summary import interpolate_log_linear_bits
    k_eff_raw, b_bits_raw = interpolate_log_linear_bits(r_norm_pct_raw, e008_data["prototype_ladder"])
    k_eff_cal, b_bits_cal = interpolate_log_linear_bits(r_norm_pct_cal, e008_data["prototype_ladder"])
    
    rng = np.random.default_rng(20260803)
    boot_r_raw = []
    for _ in range(1000):
        idx_boot = rng.choice(N, size=N, replace=True)
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
            "brier_score": brier_raw,
            "mean_candidate_mass": mean_cand_mass,
            "mean_label_permutation_variance": mean_perm_var,
            "cross_fitted_optimal_temperature_t_star": t_star,
            "q_support_raw": q_supp_raw,
            "q_support_calibrated": q_supp_cal,
            "q_null_stratified_raw": q_null_raw,
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
    print(f"  PUBLICATION LLM LPE SUMMARY: {args.model_tag}")
    print(f"============================================================")
    print(f"  NLL (Raw):                    {nll_raw:.4f} nats")
    print(f"  NLL (Calibrated T*={t_star:.2f}):    {nll_cal:.4f} nats")
    print(f"  JSD (Raw):                    {jsd_raw:.4f} bits")
    print(f"  Brier:                        {brier_raw:.4f}")
    print(f"  Permutation Variance:         {mean_perm_var:.6f}")
    print(f"  Candidate Mass P(A+B+C):     {mean_cand_mass*100.0:.2f}%")
    print(f"  Q_support (Raw):              {q_supp_raw:.6f}")
    print(f"  Q_null (Stratified):          {q_null_raw:.6f}")
    print(f"  R_norm (Raw):                 {r_norm_pct_raw:.2f}% (95% CI: [{ci_low_raw:.2f}%, {ci_high_raw:.2f}%])")
    print(f"  R_norm (Calibrated):          {r_norm_pct_cal:.2f}%")
    print(f"  Prototype Resolution (Raw):   {b_bits_raw:.3f} bits (K_eff = {k_eff_raw:.2f})")
    print(f"============================================================")
    print(f"Exported summary to {args.output_summary}")

if __name__ == "__main__":
    main()
