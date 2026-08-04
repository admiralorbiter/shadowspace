"""Persistent Disagreement Geometry — Entrance Scales b_ij & Dirichlet Posterior Confidence P_ij.

Quantifies birth scale and posterior edge stability across 3,113 ChaosNLI items.
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

from metric_atlas import distance_hellinger_matrix, compute_topk_weights

def compute_entrance_scales(dist_matrix: np.ndarray, max_k: int = 100) -> np.ndarray:
    """Compute entrance scale matrix b_ij = min { k : j is in top-k soft neighborhood of i }."""
    N = len(dist_matrix)
    B = np.full((N, N), max_k + 1, dtype=np.int32)
    
    # Sort distances per row
    D_self = dist_matrix.copy()
    np.fill_diagonal(D_self, np.inf)
    
    for i in range(N):
        ranks = np.argsort(D_self[i])
        for r, j in enumerate(ranks[:max_k]):
            B[i, j] = r + 1
    return B

def compute_posterior_edge_confidence(P: np.ndarray, k: int = 10, n_draws: int = 50, seed: int = 20260803) -> np.ndarray:
    """Compute P_ij = Pr(j in top-k neighborhood of i | Dirichlet posterior theta ~ Dir(100*P + 0.5))."""
    N = len(P)
    edge_counts = np.zeros((N, N), dtype=np.float64)
    counts = P * 100.0 + 0.5
    rng = np.random.default_rng(seed)
    
    for _ in range(n_draws):
        P_draw = np.zeros_like(P)
        for i in range(N):
            P_draw[i] = rng.dirichlet(counts[i])
        D_draw = distance_hellinger_matrix(P_draw)
        W_draw = compute_topk_weights(D_draw, k=k)
        edge_counts += (W_draw > 0).astype(float)
        
    return edge_counts / float(n_draws)

def run_persistent_disagreement_geometry(k_ref: int = 10, n_draws: int = 50) -> dict:
    parquet_path = Path("data/chaosnli/processed/canonical_items.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}")
        
    df = pl.read_parquet(parquet_path)
    P = df.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    entropy = df["human_entropy_bits"].to_numpy()
    
    D_hellinger = distance_hellinger_matrix(P)
    
    print(f"Computing entrance scales (b_ij) for N={len(P)}...")
    B_matrix = compute_entrance_scales(D_hellinger, max_k=50)
    
    print(f"Computing Dirichlet posterior confidence (P_ij) across {n_draws} draws...")
    P_matrix = compute_posterior_edge_confidence(P, k=k_ref, n_draws=n_draws)
    
    # Identify Persistent Human Core edges: P_ij >= 0.8 & b_ij <= 10
    W_k10 = compute_topk_weights(D_hellinger, k=k_ref)
    topk_edges = W_k10 > 0
    
    persistent_core_mask = (P_matrix >= 0.8) & (topk_edges)
    volatile_edges_mask = (P_matrix < 0.3) & (topk_edges)
    
    summary = {
        "n_items": len(P),
        "k_reference": k_ref,
        "n_dirichlet_draws": n_draws,
        "mean_posterior_edge_confidence": float(np.mean(P_matrix[topk_edges])),
        "persistent_core_edges_count": int(np.sum(persistent_core_mask)),
        "volatile_edges_count": int(np.sum(volatile_edges_mask)),
        "persistent_core_percentage": float((np.sum(persistent_core_mask) / np.sum(topk_edges)) * 100.0),
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("research/chaosnli/artifacts/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = run_persistent_disagreement_geometry(k_ref=10, n_draws=50)
    out_file = out_dir / "persistent_disagreement_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Persistent Disagreement Geometry summary written to {out_file}")
