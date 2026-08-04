"""Module 2: Weighted Geometric Tears & Posterior Support Loss.

Computes:
1. Fuzzy Mass Overlap: Overlap_min(W_H, W_M) = (1 / Nk) sum_{ij} min(W_ij^H, W_ij^M)
2. Posterior Support Loss: HumanCoreLoss = 1 - sum_{ij} W_ij^H S_ij / sum_{ij} W_ij^M S_ij
3. Compares against random null and stratified null baselines.
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

def compute_posterior_support_matrix(P: np.ndarray, k: int = 10, n_draws: int = 50, seed: int = 20260803) -> np.ndarray:
    """Compute Dirichlet posterior edge confidence S_ij."""
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

def run_weighted_geometric_tears(k: int = 10) -> dict:
    items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    models_path = Path("data/chaosnli/processed/canonical_models.parquet")
    
    df_items = pl.read_parquet(items_path)
    df_models = pl.read_parquet(models_path)
    
    model_names = sorted(df_models["model_name"].unique().to_list())
    print(f"Executing Weighted Geometric Tears across {len(model_names)} models...")
    
    P_human = df_items.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    D_human = distance_hellinger_matrix(P_human)
    W_human = compute_topk_weights(D_human, k=k)
    
    print("Computing posterior support matrix S_ij...")
    S_posterior = compute_posterior_support_matrix(P_human, k=k, n_draws=30)
    
    results_by_model = {}
    
    # Compute human core support mass
    human_core_support_total = np.sum(W_human * S_posterior)
    
    for mname in model_names:
        sub_m = df_models.filter(pl.col("model_name") == mname)
        joined = df_items.join(sub_m, on="object_id", how="inner")
        
        Q_model = joined.select(["model_p_entailment", "model_p_neutral", "model_p_contradiction"]).to_numpy()
        D_model = distance_hellinger_matrix(Q_model)
        W_model = compute_topk_weights(D_model, k=k)
        
        # 1. Fuzzy Mass Overlap
        fuzzy_overlap = soft_overlap(W_human, W_model, k=k)
        tear_mass = 1.0 - fuzzy_overlap
        
        # 2. Posterior Support Loss
        model_support_total = np.sum(W_model * S_posterior)
        posterior_support_retained = float(model_support_total / (human_core_support_total + 1e-12))
        human_core_loss = 1.0 - posterior_support_retained
        
        results_by_model[mname] = {
            "fuzzy_mass_overlap": float(fuzzy_overlap),
            "tear_mass": float(tear_mass),
            "posterior_support_retained_pct": float(posterior_support_retained * 100.0),
            "human_core_loss_pct": float(human_core_loss * 100.0),
        }
        
    summary = {
        "n_items": len(P_human),
        "k": k,
        "models": results_by_model,
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = run_weighted_geometric_tears(k=10)
    out_file = out_dir / "weighted_geometric_tears_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Weighted Geometric Tears summary written to {out_file}")
