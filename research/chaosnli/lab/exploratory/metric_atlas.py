"""Metric Atlas — Comparing Hellinger, Fisher-Rao, JSD, and Aitchison Geometry.

Operates on frozen baseline human distributions with low thread footprint.
"""

import os
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

# Metrics implementation
def distance_hellinger_matrix(P: np.ndarray) -> np.ndarray:
    sqrt_P = np.sqrt(np.clip(P, 1e-12, 1.0))
    bc = np.dot(sqrt_P, sqrt_P.T)
    bc = np.clip(bc, 0.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - bc))

def distance_fisher_rao_matrix(P: np.ndarray) -> np.ndarray:
    sqrt_P = np.sqrt(np.clip(P, 1e-12, 1.0))
    bc = np.dot(sqrt_P, sqrt_P.T)
    bc = np.clip(bc, 0.0, 1.0)
    return 2.0 * np.arccos(bc)

def distance_jsd_matrix(P: np.ndarray) -> np.ndarray:
    N, C = P.shape
    D = np.zeros((N, N), dtype=np.float64)
    P_safe = np.clip(P, 1e-12, 1.0)
    
    for i in range(N):
        p_i = P_safe[i]
        M = 0.5 * (p_i + P_safe)
        kl_pm = np.sum(p_i * np.log2(p_i / M), axis=1)
        kl_qm = np.sum(P_safe * np.log2(P_safe / M), axis=1)
        D[i] = np.maximum(0.0, 0.5 * kl_pm + 0.5 * kl_qm)
    return D

def clr_transform(P: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    P_adj = P.copy()
    zeros_mask = P_adj <= 0
    if np.any(zeros_mask):
        P_adj = np.where(zeros_mask, eps, P_adj)
        P_adj = P_adj / np.sum(P_adj, axis=1, keepdims=True)
    log_P = np.log(P_adj)
    mean_log = np.mean(log_P, axis=1, keepdims=True)
    return log_P - mean_log

def distance_aitchison_matrix(P: np.ndarray) -> np.ndarray:
    Z = clr_transform(P)
    diff = Z[:, np.newaxis, :] - Z[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

def compute_topk_weights(D: np.ndarray, k: int = 10) -> np.ndarray:
    N = D.shape[0]
    D_self = D.copy()
    np.fill_diagonal(D_self, np.inf)
    k_dists = np.partition(D_self, k - 1, axis=1)[:, k - 1, np.newaxis]
    ATOL = 1e-7
    closer = D_self < (k_dists - ATOL)
    tied = np.abs(D_self - k_dists) <= ATOL
    n_closer = np.sum(closer, axis=1, keepdims=True)
    n_tied = np.sum(tied, axis=1, keepdims=True)
    frac = np.where(n_tied > 0, (k - n_closer) / np.maximum(1.0, n_tied.astype(float)), 0.0)
    W = np.where(closer, 1.0, np.where(tied, frac, 0.0))
    np.fill_diagonal(W, 0.0)
    return W

def soft_overlap(W1: np.ndarray, W2: np.ndarray, k: int = 10) -> float:
    min_w = np.minimum(W1, W2)
    return float(np.mean(np.sum(min_w, axis=1) / float(k)))

def run_metric_atlas(P: np.ndarray, k: int = 10) -> dict:
    print(f"Computing Metric Atlas for N={len(P)} items at k={k}...")
    
    d_hellinger = distance_hellinger_matrix(P)
    d_fisher_rao = distance_fisher_rao_matrix(P)
    d_jsd = distance_jsd_matrix(P)
    d_aitchison = distance_aitchison_matrix(P)
    
    w_hellinger = compute_topk_weights(d_hellinger, k)
    w_fisher_rao = compute_topk_weights(d_fisher_rao, k)
    w_jsd = compute_topk_weights(d_jsd, k)
    w_aitchison = compute_topk_weights(d_aitchison, k)
    
    # Distance matrix Spearman correlation (upper triangle)
    iu = np.triu_indices(len(P), k=1)
    
    corr_h_fr, _ = spearmanr(d_hellinger[iu], d_fisher_rao[iu])
    corr_h_jsd, _ = spearmanr(d_hellinger[iu], d_jsd[iu])
    corr_h_ait, _ = spearmanr(d_hellinger[iu], d_aitchison[iu])
    
    overlap_h_fr = soft_overlap(w_hellinger, w_fisher_rao, k)
    overlap_h_jsd = soft_overlap(w_hellinger, w_jsd, k)
    overlap_h_ait = soft_overlap(w_hellinger, w_aitchison, k)
    
    summary = {
        "n_items": int(len(P)),
        "k": k,
        "spearman_correlations": {
            "hellinger_vs_fisher_rao": float(corr_h_fr),
            "hellinger_vs_jsd": float(corr_h_jsd),
            "hellinger_vs_aitchison": float(corr_h_ait),
        },
        "soft_neighborhood_overlaps": {
            "hellinger_vs_fisher_rao": float(overlap_h_fr),
            "hellinger_vs_jsd": float(overlap_h_jsd),
            "hellinger_vs_aitchison": float(overlap_h_ait),
        }
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("research/chaosnli/artifacts/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_path = Path("data/chaosnli/processed/canonical_items.parquet")
    if parquet_path.exists():
        import polars as pl
        df = pl.read_parquet(parquet_path)
        P = df.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
        print(f"Loaded {len(P)} canonical human distributions from {parquet_path}")
    else:
        P = np.random.dirichlet([0.5, 0.5, 0.5], size=200)
        print("Using synthetic sample distributions")
        
    summary = run_metric_atlas(P, k=10)
    out_file = out_dir / "metric_atlas_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Metric Atlas summary written to {out_file}")
