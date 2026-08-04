"""Aitchison Boundary & Zero-Replacement Sensitivity Audit.

Disentangles zero-policy numerical instability from genuine log-ratio geometry divergence.
"""

import os
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent))

from metric_atlas import (
    distance_hellinger_matrix,
    compute_topk_weights,
    soft_overlap,
)

def clr_with_policy(P: np.ndarray, mode: str = "multiplicative", value: float = 1e-4) -> np.ndarray:
    P_adj = P.copy()
    if mode == "multiplicative":
        zeros_mask = P_adj <= 0
        if np.any(zeros_mask):
            P_adj = np.where(zeros_mask, value, P_adj)
            P_adj = P_adj / np.sum(P_adj, axis=1, keepdims=True)
    elif mode == "dirichlet":
        # Dirichlet smoothing: P_dir = (100 * P + alpha) / (100 + 3 * alpha)
        alpha = value
        P_adj = (100.0 * P + alpha) / (100.0 + 3.0 * alpha)
    
    log_P = np.log(np.clip(P_adj, 1e-15, 1.0))
    mean_log = np.mean(log_P, axis=1, keepdims=True)
    return log_P - mean_log

def distance_aitchison_custom(P: np.ndarray, mode: str = "multiplicative", value: float = 1e-4) -> np.ndarray:
    Z = clr_with_policy(P, mode=mode, value=value)
    diff = Z[:, np.newaxis, :] - Z[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

def run_aitchison_boundary_audit(k: int = 10) -> dict:
    parquet_path = Path("data/chaosnli/processed/canonical_items.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}")
        
    df = pl.read_parquet(parquet_path)
    P = df.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    has_zero = df["has_zero_count"].to_numpy()
    
    zero_idx = np.where(has_zero)[0]
    interior_idx = np.where(~has_zero)[0]
    
    print(f"Audit Population: Total N={len(P)}, Boundary Zero Items N={len(zero_idx)}, Interior Items N={len(interior_idx)}")
    
    D_hellinger = distance_hellinger_matrix(P)
    W_hellinger = compute_topk_weights(D_hellinger, k=k)
    
    # 1. Multiplicative Replacement Sweep
    eps_values = [1e-12, 1e-9, 1e-6, 1e-4, 1e-3]
    mult_results = {}
    for eps in eps_values:
        D_ait = distance_aitchison_custom(P, mode="multiplicative", value=eps)
        W_ait = compute_topk_weights(D_ait, k=k)
        
        overlap_all = soft_overlap(W_hellinger, W_ait, k)
        overlap_zero = soft_overlap(W_hellinger[zero_idx][:, zero_idx], W_ait[zero_idx][:, zero_idx], k)
        overlap_int = soft_overlap(W_hellinger[interior_idx][:, interior_idx], W_ait[interior_idx][:, interior_idx], k)
        
        mult_results[f"eps_{eps:.0e}"] = {
            "epsilon": eps,
            "overlap_all_n3113": overlap_all,
            "overlap_boundary_n1007": overlap_zero,
            "overlap_interior_n2106": overlap_int,
        }
        
    # 2. Dirichlet Smoothing Sweep
    alpha_values = [0.1, 0.5, 1.0]
    dir_results = {}
    for alpha in alpha_values:
        D_ait = distance_aitchison_custom(P, mode="dirichlet", value=alpha)
        W_ait = compute_topk_weights(D_ait, k=k)
        
        overlap_all = soft_overlap(W_hellinger, W_ait, k)
        overlap_zero = soft_overlap(W_hellinger[zero_idx][:, zero_idx], W_ait[zero_idx][:, zero_idx], k)
        overlap_int = soft_overlap(W_hellinger[interior_idx][:, interior_idx], W_ait[interior_idx][:, interior_idx], k)
        
        dir_results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "overlap_all_n3113": overlap_all,
            "overlap_boundary_n1007": overlap_zero,
            "overlap_interior_n2106": overlap_int,
        }
        
    summary = {
        "n_total": len(P),
        "n_boundary_zero": len(zero_idx),
        "n_strictly_interior": len(interior_idx),
        "k": k,
        "multiplicative_epsilon_sweep": mult_results,
        "dirichlet_smoothing_sweep": dir_results,
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("research/chaosnli/artifacts/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = run_aitchison_boundary_audit(k=10)
    out_file = out_dir / "aitchison_boundary_audit_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Aitchison Boundary Audit written to {out_file}")
