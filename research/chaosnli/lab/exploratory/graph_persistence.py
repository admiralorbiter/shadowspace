"""Multi-Scale Graph Persistence — Edge survival across scales k and posterior draws.
"""

import os
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from metric_atlas import distance_hellinger_matrix, compute_topk_weights, soft_overlap

def run_graph_persistence(P: np.ndarray, scales: list[int] = [5, 10, 20, 50, 100], n_draws: int = 20) -> dict:
    print(f"Computing Graph Persistence for N={len(P)} across scales {scales}...")
    
    D_obs = distance_hellinger_matrix(P)
    graphs_obs = {k: compute_topk_weights(D_obs, k) for k in scales}
    
    # Track cross-scale persistence: overlap of G(k) with G(k_ref=10)
    ref_k = 10
    W_ref = graphs_obs[ref_k]
    cross_scale_survival = {}
    for k in scales:
        cross_scale_survival[f"k_{k}_vs_k_10"] = soft_overlap(graphs_obs[k], W_ref, k=max(k, ref_k))
        
    # Dirichlet posterior stability
    # Draw P_draw ~ Dir(100 * P + 0.5)
    rng = np.random.default_rng(20260803)
    posterior_overlaps = []
    
    counts = P * 100.0 + 0.5
    for _ in range(n_draws):
        P_draw = np.zeros_like(P)
        for i in range(len(P)):
            P_draw[i] = rng.dirichlet(counts[i])
        D_draw = distance_hellinger_matrix(P_draw)
        W_draw = compute_topk_weights(D_draw, k=ref_k)
        posterior_overlaps.append(soft_overlap(W_ref, W_draw, k=ref_k))
        
    summary = {
        "n_items": int(len(P)),
        "reference_k": ref_k,
        "cross_scale_survival": cross_scale_survival,
        "posterior_stability_k10": {
            "mean_overlap": float(np.mean(posterior_overlaps)),
            "std_overlap": float(np.std(posterior_overlaps)),
            "min_overlap": float(np.min(posterior_overlaps)),
            "max_overlap": float(np.max(posterior_overlaps)),
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
        
    summary = run_graph_persistence(P)
    out_file = out_dir / "graph_persistence_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Graph Persistence summary written to {out_file}")
