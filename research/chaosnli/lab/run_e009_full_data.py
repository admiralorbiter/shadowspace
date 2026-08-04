"""E009 Fast Vectorized Full-Data Temperature-Topology Phase Diagram Engine (N=3113).

Evaluates scalar temperature grid T in [0.05, 100.0] (50 log-spaced points) for all 9 discriminant models
on full ChaosNLI dataset (N=3113 items) using fast vectorized distance matrix operations.
Produces research/chaosnli/artifacts/E009/summaries/E009_summary.json and E009_summary.md.
"""

import json
import sys
from pathlib import Path
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "research" / "chaosnli" / "lab"))

def run_e009_fast():
    items_path = PROJECT_ROOT / "data" / "chaosnli" / "processed" / "canonical_items.parquet"
    models_path = PROJECT_ROOT / "data" / "chaosnli" / "processed" / "canonical_models.parquet"
    
    df_items = pl.read_parquet(items_path).sort("object_id")
    df_models = pl.read_parquet(models_path).sort(["model_name", "object_id"])
    
    P_human = df_items.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    N = len(P_human)
    print(f"Running Fast E009 Full-Data Temperature-Topology Phase Diagram across N={N} items...")
    
    sqrt_P_human = np.sqrt(np.clip(P_human, 1e-12, 1.0))
    BC_human = np.dot(sqrt_P_human, sqrt_P_human.T)
    D_human = np.sqrt(np.maximum(0.0, 1.0 - np.clip(BC_human, 0.0, 1.0)))
    
    # Top-10 human support mask (k=10)
    idx_h = np.argsort(D_human, axis=1)[:, 1:11]  # Exclude self
    W_human_mask = np.zeros((N, N), dtype=bool)
    np.put_along_axis(W_human_mask, idx_h, True, axis=1)
    
    q_hh = 0.038987226212620456
    q_null = 0.0032133676259691745
    
    temps = np.logspace(np.log10(0.05), np.log10(100.0), num=50)
    model_names = sorted(df_models["model_name"].unique().to_list())
    
    results_by_model = {}
    
    for mname in model_names:
        sub_m = df_models.filter(pl.col("model_name") == mname).sort("object_id")
        Logits = sub_m.select(["logit_entailment", "logit_neutral", "logit_contradiction"]).to_numpy()
        
        # Raw T=1
        z1 = Logits
        z1_max = np.max(z1, axis=-1, keepdims=True)
        exp_z1 = np.exp(z1 - z1_max)
        Q_raw = exp_z1 / np.sum(exp_z1, axis=-1, keepdims=True)
        
        sqrt_Q_raw = np.sqrt(np.clip(Q_raw, 1e-12, 1.0))
        BC_raw = np.dot(sqrt_Q_raw, sqrt_Q_raw.T)
        D_raw = np.sqrt(np.maximum(0.0, 1.0 - np.clip(BC_raw, 0.0, 1.0)))
        idx_r = np.argsort(D_raw, axis=1)[:, 1:11]
        W_raw_mask = np.zeros((N, N), dtype=bool)
        np.put_along_axis(W_raw_mask, idx_r, True, axis=1)
        
        q_supp_raw = float(np.mean(np.sum(W_human_mask & W_raw_mask, axis=1) / 10.0))
        r_norm_raw = float((q_supp_raw - q_null) / (q_hh - q_null))
        nll_raw = float(-np.mean(np.sum(P_human * np.log(np.clip(Q_raw, 1e-12, 1.0)), axis=1)))
        
        grid_points = []
        best_nll = float("inf")
        best_nll_temp = 1.0
        best_nll_r_norm = r_norm_raw
        
        best_q_supp = -float("inf")
        best_q_temp = 1.0
        best_q_r_norm = r_norm_raw
        
        for T in temps:
            z_T = Logits / T
            z_T_max = np.max(z_T, axis=-1, keepdims=True)
            exp_z_T = np.exp(z_T - z_T_max)
            Q_T = exp_z_T / np.sum(exp_z_T, axis=-1, keepdims=True)
            
            nll_T = float(-np.mean(np.sum(P_human * np.log(np.clip(Q_T, 1e-12, 1.0)), axis=1)))
            
            m_mix = 0.5 * (P_human + Q_T)
            kl_p = np.sum(np.where(P_human > 0, P_human * np.log(np.maximum(P_human, 1e-12) / m_mix), 0.0), axis=1)
            kl_q = np.sum(np.where(Q_T > 0, Q_T * np.log(np.maximum(Q_T, 1e-12) / m_mix), 0.0), axis=1)
            jsd_T = float(np.mean(0.5 * (kl_p + kl_q) / np.log(2.0)))
            
            sqrt_Q_T = np.sqrt(np.clip(Q_T, 1e-12, 1.0))
            BC_T = np.dot(sqrt_Q_T, sqrt_Q_T.T)
            D_T = np.sqrt(np.maximum(0.0, 1.0 - np.clip(BC_T, 0.0, 1.0)))
            
            idx_T = np.argsort(D_T, axis=1)[:, 1:11]
            W_T_mask = np.zeros((N, N), dtype=bool)
            np.put_along_axis(W_T_mask, idx_T, True, axis=1)
            
            q_supp_T = float(np.mean(np.sum(W_human_mask & W_T_mask, axis=1) / 10.0))
            r_norm_T = float((q_supp_T - q_null) / (q_hh - q_null))
            turnover = float(1.0 - np.mean(np.sum(W_raw_mask & W_T_mask, axis=1) / 10.0))
            
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
                "r_normalized": r_norm_T,
                "graph_turnover_frac": turnover
            })
            
        max_r_gain = best_q_r_norm - r_norm_raw
        
        results_by_model[mname] = {
            "model_name": mname,
            "raw_nll": nll_raw,
            "raw_r_norm": r_norm_raw,
            "opt_nll_temp": best_nll_temp,
            "opt_nll_val": best_nll,
            "opt_nll_r_norm": best_nll_r_norm,
            "opt_q_temp": best_q_temp,
            "opt_q_r_norm": best_q_r_norm,
            "max_r_gain": max_r_gain,
            "grid_points": grid_points
        }
        print(f"Model {mname:15s} | Raw R: {r_norm_raw*100.0:5.2f}% | Opt NLL T: {best_nll_temp:5.2f} (R: {best_nll_r_norm*100.0:5.2f}%) | Max R Gain: {max_r_gain*100.0:+5.2f}%")
        
    summary = {
        "experiment_id": "E009",
        "title": "Temperature-Topology Phase Diagram Engine",
        "subset": "full",
        "object_count": N,
        "q_hh_relational": q_hh,
        "q_null_stratified": q_null,
        "temperature_grid": temps.tolist(),
        "models": results_by_model
    }
    
    out_dir = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E009" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = out_dir / "E009_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    from e009_postprocess import generate_e009_markdown
    generate_e009_markdown()
    
    print(f"\nE009 full-data execution complete! Summary exported to {json_path}")

if __name__ == "__main__":
    run_e009_fast()
