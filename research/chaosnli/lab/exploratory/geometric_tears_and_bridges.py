"""Module 3: Geometric Tears & False Bridges.

Compares human top-k neighborhood graphs against model graphs.
Classifies edges into:
- Preserved Edges (green): Human neighbors retained by model
- Torn Edges (red): Human neighbors severed by model
- False Bridges (amber): Model neighbors unsupported by humans
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

def run_geometric_tears_and_bridges(k: int = 10) -> dict:
    items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    models_path = Path("data/chaosnli/processed/canonical_models.parquet")
    
    df_items = pl.read_parquet(items_path)
    df_models = pl.read_parquet(models_path)
    
    model_names = sorted(df_models["model_name"].unique().to_list())
    print(f"Executing Geometric Tears & False Bridges across {len(model_names)} models & N=3113 items...")
    
    P_human = df_items.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    D_human = distance_hellinger_matrix(P_human)
    W_human = compute_topk_weights(D_human, k=k)
    
    human_edge_mask = W_human > 0.0
    total_human_edges = int(np.sum(human_edge_mask))
    
    results_by_model = {}
    
    for mname in model_names:
        sub_m = df_models.filter(pl.col("model_name") == mname)
        joined = df_items.join(sub_m, on="object_id", how="inner")
        
        Q_model = joined.select(["model_p_entailment", "model_p_neutral", "model_p_contradiction"]).to_numpy()
        D_model = distance_hellinger_matrix(Q_model)
        W_model = compute_topk_weights(D_model, k=k)
        
        model_edge_mask = W_model > 0.0
        total_model_edges = int(np.sum(model_edge_mask))
        
        # Preserved edges (both human and model)
        preserved_mask = human_edge_mask & model_edge_mask
        preserved_count = int(np.sum(preserved_mask))
        
        # Torn edges (human edge missing in model)
        torn_mask = human_edge_mask & (~model_edge_mask)
        torn_count = int(np.sum(torn_mask))
        
        # False bridges (model edge unsupported by humans)
        false_bridge_mask = (~human_edge_mask) & model_edge_mask
        false_bridge_count = int(np.sum(false_bridge_mask))
        
        results_by_model[mname] = {
            "k": k,
            "total_human_edges": total_human_edges,
            "total_model_edges": total_model_edges,
            "preserved_edges_count": preserved_count,
            "torn_edges_count": torn_count,
            "false_bridges_count": false_bridge_count,
            "preserved_edge_percentage": float((preserved_count / total_human_edges) * 100.0),
            "torn_edge_percentage": float((torn_count / total_human_edges) * 100.0),
            "false_bridge_ratio": float((false_bridge_count / total_model_edges) * 100.0),
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
    
    summary = run_geometric_tears_and_bridges(k=10)
    out_file = out_dir / "geometric_tears_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Geometric Tears & False Bridges summary written to {out_file}")
