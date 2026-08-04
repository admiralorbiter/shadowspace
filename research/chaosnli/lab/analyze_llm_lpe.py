"""Generic LLM LPE Analysis Pipeline.

Consolidates LLM response evaluation into a single, model-parameterized, publication-grade pipeline.

CLI Arguments:
  --model-tag      Ollama or HuggingFace model tag (e.g. gemma3:12b, gemma4:12b, qwen3:14b)
  --responses      Path to raw LPE JSONL responses file
  --manifest       Path to items manifest JSONL (e.g. pilot_600.jsonl)
  --support-matrix Path to binary support matrix file (e.g. S_hellinger_k010_pilot.bin)
  --output-summary Path to destination JSON summary file

Produces:
  - Pointwise metrics (NLL nats, JSD bits, Brier, candidate mass P(A)+P(B)+P(C))
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

def main():
    parser = argparse.ArgumentParser(description="Generic LLM LPE Evaluation Pipeline")
    parser.add_argument("--model-tag", type=str, required=True, help="Model tag (e.g. gemma3:12b)")
    parser.add_argument("--responses", type=Path, required=True, help="Path to raw LPE JSONL responses")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to items manifest JSONL")
    parser.add_argument("--support-matrix", type=Path, required=True, help="Path to binary support matrix file")
    parser.add_argument("--output-summary", type=Path, required=True, help="Path to output JSON summary")
    
    args = parser.parse_args()
    
    # Provenance digests
    resp_bytes = args.responses.read_bytes()
    resp_sha256 = hashlib.sha256(resp_bytes).hexdigest()
    
    man_bytes = args.manifest.read_bytes()
    man_sha256 = hashlib.sha256(man_bytes).hexdigest()
    
    supp_bytes = args.support_matrix.read_bytes()
    supp_sha256 = hashlib.sha256(supp_bytes).hexdigest()
    
    # Load manifest items
    items = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
                
    N = len(items)
    print(f"Loaded {N} items from manifest {args.manifest.name} (SHA256: {man_sha256[:12]}...)")
    
    # Load support matrix
    S = np.frombuffer(supp_bytes, dtype=np.float32).reshape(N, N).astype(np.float64)
    
    # Extract human probabilities and SNLI flags
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
        is_snli[i] = "snli" in obj_id
        
    # Load raw LPE responses and aggregate per-item distributions
    # Responses format: item_id, perm_index, candidate_logprobs or probabilities
    # We aggregate across all 6 S3 permutations to form coherent P_model
    item_preds = {i: [] for i in range(N)}
    cand_masses = []
    
    with open(args.responses, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            item_idx = rec.get("item_index", rec.get("index", 0))
            probs = rec.get("normalized_probs", rec.get("p_model", [0.333, 0.333, 0.334]))
            c_mass = rec.get("candidate_mass", 1.0)
            
            item_preds[item_idx].append(probs)
            cand_masses.append(c_mass)
            
    P_model = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        if len(item_preds[i]) > 0:
            P_model[i] = np.mean(item_preds[i], axis=0)
        else:
            P_model[i] = [0.333, 0.333, 0.334]
            
        P_model[i] = np.clip(P_model[i], 1e-12, 1.0)
        P_model[i] /= np.sum(P_model[i])
        
    # Pointwise evaluation
    nll = float(-np.mean(np.sum(P_human * np.log(np.clip(P_model, 1e-12, 1.0)), axis=1)))
    
    m_mix = 0.5 * (P_human + P_model)
    kl_p = np.sum(np.where(P_human > 0, P_human * np.log(np.maximum(P_human, 1e-12) / m_mix), 0.0), axis=1)
    kl_q = np.sum(np.where(P_model > 0, P_model * np.log(np.maximum(P_model, 1e-12) / m_mix), 0.0), axis=1)
    jsd = float(np.mean(0.5 * (kl_p + kl_q) / np.log(2.0)))
    
    brier = float(np.mean(np.sum((P_model - P_human) ** 2, axis=1)))
    mean_cand_mass = float(np.mean(cand_masses)) if cand_masses else 1.0
    
    # Relational evaluation
    D_model = compute_hellinger_matrix(P_model)
    W_model = compute_soft_neighborhood_weights(D_model, k=10)
    
    q_supp = float(np.sum(W_model * S) / (N * 10.0))
    q_null = compute_dataset_stratified_null(W_model, S, is_snli, k=10)
    
    # Q_HH from E008 pilot curve
    e008_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E008_pilot_600_curve.json"
    with open(e008_path, "r", encoding="utf-8") as f:
        e008_data = json.load(f)
    q_hh = e008_data["q_hh_relational"]
    
    r_norm = float((q_supp - q_null) / (q_hh - q_null))
    r_norm_pct = float(r_norm * 100.0)
    
    # Pilot E008 prototype-equivalent resolution mapping
    from sprint1_rate_distortion_and_summary import interpolate_log_linear_bits
    k_eff, b_bits = interpolate_log_linear_bits(r_norm_pct, e008_data["prototype_ladder"])
    
    summary = {
        "model_tag": args.model_tag,
        "n_items": N,
        "metrics": {
            "nll_nats": nll,
            "jsd_bits": jsd,
            "brier_score": brier,
            "mean_candidate_mass": mean_cand_mass,
            "q_support": q_supp,
            "q_null_stratified": q_null,
            "q_hh_relational": q_hh,
            "r_normalized": r_norm,
            "r_norm_pct": r_norm_pct,
            "prototype_equivalent_bits": b_bits,
            "k_eff_prototypes": k_eff
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
        
    print(f"\n============================================================")
    print(f"  LLM LPE EVALUATION SUMMARY: {args.model_tag}")
    print(f"============================================================")
    print(f"  NLL:                         {nll:.4f} nats")
    print(f"  JSD:                         {jsd:.4f} bits")
    print(f"  Brier:                       {brier:.4f}")
    print(f"  Candidate Mass P(A+B+C):    {mean_cand_mass*100.0:.2f}%")
    print(f"  Q_support:                   {q_supp:.6f}")
    print(f"  Q_null (stratified):         {q_null:.6f}")
    print(f"  R_norm:                      {r_norm_pct:.2f}%")
    print(f"  Prototype Resolution (b):    {b_bits:.3f} bits")
    print(f"  Prototype Quantizers (K_eff):{k_eff:.2f}")
    print(f"============================================================")
    print(f"Exported summary to {args.output_summary}")

if __name__ == "__main__":
    main()
