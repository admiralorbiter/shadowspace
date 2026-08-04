"""Audited E009: Temperature-Topology Phase Diagram Engine (Canonical Frozen Target).

Loads the frozen 500-draw Dirichlet posterior support matrix S directly from S_hellinger_k010.bin,
asserts exact SHA-256 integrity, uses the canonical tie-aware soft neighborhood engine,
and recomputes exact dataset-stratified block null Q_null(T) at every temperature T.

Asserts exact reproduction of T=1.0 model scores before running grid.
Exports tracked results to research/chaosnli/results/E009_full_summary.json.
"""

import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import polars as pl

from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "chaosnli" / "lab"))

EXPECTED_SUPPORT_SHA256 = "94e483e714d92f039f817389d948cbf41b7970077b56f852491832605dccc96f"

# Model scores at T=1.0 (R_norm pct) for exact T=1 reproduction assertion
EXPECTED_RAW_R_NORM = {
    "bart-large": 0.38002,
    "roberta-large": 0.32718,
    "xlnet-large": 0.28314,
    "albert-xxlarge": 0.23251,
    "bert-large": 0.20461,
    "roberta-base": 0.19889,
    "xlnet-base": 0.17136,
    "distilbert": 0.15185,
    "bert-base": 0.12986,
}

def softmax_temp(logits: np.ndarray, temp: float) -> np.ndarray:
    z = logits / max(temp, 1e-5)
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

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

def run_audited_e009():
    items_path = PROJECT_ROOT / "data" / "chaosnli" / "processed" / "canonical_items.parquet"
    models_path = PROJECT_ROOT / "data" / "chaosnli" / "processed" / "canonical_models.parquet"
    support_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E001" / "S_hellinger_k010.bin"
    if not support_path.exists():
        support_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "pilot_support" / "S_hellinger_k010_full.bin"

    with open(support_path, "rb") as f:
        support_bytes = f.read()
        sha256_support = hashlib.sha256(support_bytes).hexdigest()

    print(f"Loading frozen support matrix from {support_path}...")
    print(f"  SHA-256 Digest: {sha256_support}")
    assert sha256_support == EXPECTED_SUPPORT_SHA256, f"Support matrix SHA256 mismatch: {sha256_support} != {EXPECTED_SUPPORT_SHA256}"
    
    df_items = pl.read_parquet(items_path).sort("object_id")
    df_models = pl.read_parquet(models_path).sort(["model_name", "object_id"])
    
    P_human = df_items.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    is_snli = df_items["object_id"].str.contains("snli").to_numpy()
    N = len(P_human)
    
    assert len(support_bytes) == N * N * 4, f"Support matrix byte length mismatch: {len(support_bytes)} != {N*N*4}"
    S = np.frombuffer(support_bytes, dtype=np.float32).reshape(N, N).astype(np.float64)

    e007_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E007_full_census_summary.json"
    with open(e007_path, "r", encoding="utf-8") as f:
        e007_data = json.load(f)
    q_hh = e007_data["q_hh_relational"]
    
    temps = np.logspace(np.log10(0.05), np.log10(100.0), num=50)
    model_names = sorted(df_models["model_name"].unique().to_list())
    
    results_by_model = {}

    # Exact T=1 reproduction assertions before running grid
    print("\n--- Auditing T=1 Exact Reproduction Assertions ---")
    for mname in model_names:
        sub_m = df_models.filter(pl.col("model_name") == mname).sort("object_id")
        Logits = sub_m.select(["logit_entailment", "logit_neutral", "logit_contradiction"]).to_numpy()
        
        Q_raw = softmax_temp(Logits, 1.0)
        D_raw = compute_hellinger_matrix(Q_raw)
        W_raw = compute_soft_neighborhood_weights(D_raw, k=10)
        
        q_supp_raw = float(np.sum(W_raw * S) / (N * 10.0))
        q_null_raw = compute_dataset_stratified_null(W_raw, S, is_snli, k=10)
        r_norm_raw = float((q_supp_raw - q_null_raw) / (q_hh - q_null_raw))
        
        expected_r = EXPECTED_RAW_R_NORM[mname]
        diff_r = abs(r_norm_raw - expected_r)
        print(f"Model {mname:15s} | Calc R: {r_norm_raw*100.0:6.3f}% | Expected R: {expected_r*100.0:6.3f}% | Diff: {diff_r:.6e}")
        assert diff_r < 1e-3, f"Exact T=1 reproduction assertion failed for {mname}: calc={r_norm_raw}, expected={expected_r}"
    print("All T=1 reproduction assertions PASSED!\n")

    print("--- Running 50-Point Temperature Phase Sweep ---")
    for mname in model_names:
        sub_m = df_models.filter(pl.col("model_name") == mname).sort("object_id")
        Logits = sub_m.select(["logit_entailment", "logit_neutral", "logit_contradiction"]).to_numpy()
        
        Q_raw = softmax_temp(Logits, 1.0)
        D_raw = compute_hellinger_matrix(Q_raw)
        W_raw = compute_soft_neighborhood_weights(D_raw, k=10)
        
        q_supp_raw = float(np.sum(W_raw * S) / (N * 10.0))
        q_null_raw = compute_dataset_stratified_null(W_raw, S, is_snli, k=10)
        r_norm_raw = float((q_supp_raw - q_null_raw) / (q_hh - q_null_raw))
        nll_raw = float(-np.mean(np.sum(P_human * np.log(np.clip(Q_raw, 1e-12, 1.0)), axis=1)))
        
        grid_points = []
        best_nll = float("inf")
        best_nll_temp = 1.0
        best_nll_r_norm = r_norm_raw
        
        best_q_supp = -float("inf")
        best_q_temp = 1.0
        best_q_r_norm = r_norm_raw
        
        for T in temps:
            Q_T = softmax_temp(Logits, T)
            nll_T = float(-np.mean(np.sum(P_human * np.log(np.clip(Q_T, 1e-12, 1.0)), axis=1)))
            
            m_mix = 0.5 * (P_human + Q_T)
            kl_p = np.sum(np.where(P_human > 0, P_human * np.log(np.maximum(P_human, 1e-12) / m_mix), 0.0), axis=1)
            kl_q = np.sum(np.where(Q_T > 0, Q_T * np.log(np.maximum(Q_T, 1e-12) / m_mix), 0.0), axis=1)
            jsd_T = float(np.mean(0.5 * (kl_p + kl_q) / np.log(2.0)))
            
            D_T = compute_hellinger_matrix(Q_T)
            W_T = compute_soft_neighborhood_weights(D_T, k=10)
            
            q_supp_T = float(np.sum(W_T * S) / (N * 10.0))
            q_null_T = compute_dataset_stratified_null(W_T, S, is_snli, k=10)
            r_norm_T = float((q_supp_T - q_null_T) / (q_hh - q_null_T))
            
            min_W = np.minimum(W_raw, W_T)
            turnover = float(1.0 - np.mean(np.sum(min_W, axis=1) / 10.0))
            
            if nll_T < best_nll:
                best_nll = nll_T
                best_nll_temp = float(T)
                best_nll_r_norm = r_norm_T
                
            if r_norm_T > best_q_supp:
                best_q_supp = r_norm_T
                best_q_temp = float(T)
                best_q_r_norm = r_norm_T
                
            grid_points.append({
                "temperature": float(T),
                "nll": nll_T,
                "jsd_bits": jsd_T,
                "q_support": q_supp_T,
                "q_null_stratified": q_null_T,
                "r_normalized": r_norm_T,
                "graph_turnover_frac": turnover
            })
            
        nll_opt_gain = best_nll_r_norm - r_norm_raw
        max_r_gain = best_q_r_norm - r_norm_raw
        
        results_by_model[mname] = {
            "model_name": mname,
            "raw_nll": nll_raw,
            "raw_r_norm": r_norm_raw,
            "opt_nll_temp": best_nll_temp,
            "opt_nll_val": best_nll,
            "opt_nll_r_norm": best_nll_r_norm,
            "nll_opt_r_gain": nll_opt_gain,
            "opt_q_temp": best_q_temp,
            "opt_q_r_norm": best_q_r_norm,
            "max_r_gain": max_r_gain,
            "grid_points": grid_points
        }
        print(f"Model {mname:15s} | Raw R: {r_norm_raw*100.0:5.2f}% | Opt NLL T: {best_nll_temp:5.2f} (R: {best_nll_r_norm*100.0:5.2f}%, NLL-Gain: {nll_opt_gain*100.0:+5.2f}%) | Max R (T={best_q_temp:5.2f}): {best_q_r_norm*100.0:5.2f}% (Max-Gain: {max_r_gain*100.0:+5.2f}%)")

    summary = {
        "experiment_id": "E009",
        "title": "Audited Temperature-Topology Phase Diagram Engine",
        "subset": "full",
        "object_count": N,
        "q_hh_relational": q_hh,
        "support_matrix_sha256": sha256_support,
        "temperature_grid": temps.tolist(),
        "models": results_by_model
    }
    
    tracked_out_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E009_full_summary.json"
    tracked_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tracked_out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    provenance = {
        "experiment_id": "E009",
        "status": "complete",
        "support_matrix_path": str(support_path),
        "support_matrix_sha256": sha256_support,
        "n_items": N,
        "n_models": len(model_names),
        "q_hh_relational": q_hh,
        "t1_reproduction_assertions": "PASSED"
    }
    prov_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E009_PROVENANCE.json"
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    # Update E009 registry to complete
    e009_toml_path = PROJECT_ROOT / "research" / "chaosnli" / "lab" / "registry" / "E009.toml"
    if e009_toml_path.exists():
        content = e009_toml_path.read_text(encoding="utf-8")
        content = content.replace('status = "exploratory"', 'status = "complete"')
        content = content.replace('analysis_status = "not_started"', 'analysis_status = "complete"')
        e009_toml_path.write_text(content, encoding="utf-8")

    from e009_postprocess import generate_e009_markdown
    generate_e009_markdown()
    
    print(f"\nAudited E009 full-data execution complete!")
    print(f"  Tracked JSON: {tracked_out_path}")
    print(f"  Provenance:   {prov_path}")

if __name__ == "__main__":
    run_audited_e009()
