"""Module 1: E016 — The Calibration Ray Theorem & Ambiguity Angles.

Decomposes human-model mismatch into:
1. Calibration-reachable component (along positive CLR ray)
2. Directional ambiguity-type error (orthogonal to CLR ray)

Reports 3 distinct reachability metrics:
- Itemwise Oracle Reachability
- Global Scalar Reachability
- Relational Graph Reachability
"""

import os
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
from pathlib import Path
import numpy as np
import polars as pl

import sys
sys.path.insert(0, str(Path(__file__).parent))

from metric_atlas import distance_hellinger_matrix, compute_topk_weights, soft_overlap

def clr_transform_from_logits(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = logits / temp
    mean_z = np.mean(z, axis=-1, keepdims=True)
    return z - mean_z

def clr_transform_from_probs(p: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    counts = p * 100.0 + alpha
    p_smooth = counts / np.sum(counts, axis=-1, keepdims=True)
    log_p = np.log(p_smooth)
    mean_log_p = np.mean(log_p, axis=-1, keepdims=True)
    return log_p - mean_log_p

def softmax_temp(logits: np.ndarray, temp: float) -> np.ndarray:
    z = logits / temp
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def run_calibration_ray_theorem() -> dict:
    items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    models_path = Path("data/chaosnli/processed/canonical_models.parquet")
    
    df_items = pl.read_parquet(items_path)
    df_models = pl.read_parquet(models_path)
    
    model_names = sorted(df_models["model_name"].unique().to_list())
    print(f"Executing Calibration Ray Theorem (E016) across {len(model_names)} models...")
    
    # Human Dirichlet-smoothed CLR targets
    P_human = df_items.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    H_clr = clr_transform_from_probs(P_human, alpha=0.5)
    norm_H = np.linalg.norm(H_clr, axis=1)
    
    D_human = distance_hellinger_matrix(P_human)
    W_human = compute_topk_weights(D_human, k=10)
    
    temps_grid = np.logspace(np.log10(0.05), np.log10(100.0), num=50)
    results_by_model = {}
    
    for mname in model_names:
        sub_m = df_models.filter(pl.col("model_name") == mname)
        joined = df_items.join(sub_m, on="object_id", how="inner")
        
        Logits = joined.select(["logit_entailment", "logit_neutral", "logit_contradiction"]).to_numpy()
        M_clr1 = clr_transform_from_logits(Logits, temp=1.0)
        norm_M1 = np.linalg.norm(M_clr1, axis=1)
        
        # 1. Ambiguity Angle (100% invariant under temperature scaling)
        dot_HM = np.sum(H_clr * M_clr1, axis=1)
        cos_theta = np.clip(dot_HM / (norm_H * norm_M1 + 1e-12), -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)
        
        # 2. Exact Orthogonal Decomposition in CLR space
        # Optimal scalar alpha^* = max(0, <H, M> / ||M||^2)
        alpha_star = np.maximum(0.0, dot_HM / (norm_M1**2 + 1e-12))
        
        # Reachable component alpha^* M
        H_reachable = alpha_star[:, np.newaxis] * M_clr1
        # Orthogonal directional error
        H_orthogonal = H_clr - H_reachable
        
        norm_reachable = np.linalg.norm(H_reachable, axis=1)
        norm_orthogonal = np.linalg.norm(H_orthogonal, axis=1)
        
        # Ratio of error orthogonal to temperature ray
        orthogonal_error_ratio = np.mean(norm_orthogonal / (norm_H + 1e-12))
        
        # 3. Disaggregate 3-level reachabilities:
        # A. Itemwise Oracle Reachability (each item picks T_i^* = 1 / alpha_star)
        T_oracle = np.where(alpha_star > 1e-6, 1.0 / np.maximum(alpha_star, 1e-6), 100.0)
        T_oracle = np.clip(T_oracle, 0.05, 100.0)
        
        Q_raw = softmax_temp(Logits, 1.0)
        Q_opt = softmax_temp(Logits, T_oracle[:, np.newaxis])
        bc_raw = np.sum(np.sqrt(np.clip(P_human * Q_raw, 0.0, 1.0)), axis=1)
        bc_opt = np.sum(np.sqrt(np.clip(P_human * Q_opt, 0.0, 1.0)), axis=1)
        d_raw = np.sqrt(np.clip(1.0 - bc_raw, 0.0, 1.0))
        d_oracle = np.sqrt(np.clip(1.0 - bc_opt, 0.0, 1.0))
        
        itemwise_oracle_reachability = float(np.mean((d_raw - d_oracle) / np.maximum(d_raw, 1e-6)) * 100.0)
        
        # B. Global Scalar Reachability (single global T^* minimizing average Hellinger)
        global_d_means = []
        global_q_overlaps = []
        for T in temps_grid:
            Q_T = softmax_temp(Logits, T)
            d_T = np.mean(np.sqrt(np.clip(1.0 - np.sum(np.sqrt(P_human * Q_T), axis=1), 0.0, 1.0)))
            global_d_means.append(d_T)
            
            D_T = distance_hellinger_matrix(Q_T)
            W_T = compute_topk_weights(D_T, k=10)
            global_q_overlaps.append(soft_overlap(W_human, W_T, k=10))
            
        best_global_idx = np.argmin(global_d_means)
        global_scalar_reachability = float((d_raw.mean() - global_d_means[best_global_idx]) / d_raw.mean() * 100.0)
        
        # C. Relational Graph Reachability (10 log-spaced temps for graph reachability)
        temps_graph = np.logspace(np.log10(0.05), np.log10(100.0), num=10)
        global_q_overlaps = []
        for T in temps_graph:
            Q_T = softmax_temp(Logits, T)
            D_T = distance_hellinger_matrix(Q_T)
            W_T = compute_topk_weights(D_T, k=10)
            global_q_overlaps.append(soft_overlap(W_human, W_T, k=10))
            
        best_global_idx = np.argmin(global_d_means)
        global_scalar_reachability = float((d_raw.mean() - global_d_means[best_global_idx]) / d_raw.mean() * 100.0)
        
        best_relational_idx = np.argmax(global_q_overlaps)
        relational_graph_reachability = float(global_q_overlaps[best_relational_idx] * 100.0)
        
        results_by_model[mname] = {
            "ambiguity_angle_deg_mean": float(np.mean(theta_deg)),
            "ambiguity_angle_deg_median": float(np.median(theta_deg)),
            "orthogonal_error_ratio_mean": float(orthogonal_error_ratio * 100.0),
            "itemwise_oracle_reachability_pct": itemwise_oracle_reachability,
            "global_scalar_reachability_pct": global_scalar_reachability,
            "best_global_temp": float(temps_grid[best_global_idx]),
            "relational_graph_reachability_pct": relational_graph_reachability,
            "best_relational_temp": float(temps_graph[best_relational_idx]),
        }
        
    summary = {
        "n_items": len(P_human),
        "dirichlet_alpha": 0.5,
        "models": results_by_model,
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = run_calibration_ray_theorem()
    out_file = out_dir / "calibration_ray_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Calibration Ray Theorem summary written to {out_file}")
