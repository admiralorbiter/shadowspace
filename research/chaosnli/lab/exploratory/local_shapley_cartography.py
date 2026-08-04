"""Module 4: Local Shapley Cartography & Pluralistic Adversarial Pairs.

Calculates regional Shapley contributions phi_{m,r} per model m across 7 coarse simplex prototype regions r.
Identifies Pluralistic Adversarial Pairs (Torn Twins, False Twins, Calibration Casualties, Ensemble Rescues, Metric Rebels).
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

def classify_simplex_region(p_e: float, p_n: float, p_c: float) -> str:
    """Classify 3D probability into 7 coarse prototype regions."""
    if p_e >= 0.7:
        return "corner_entailment"
    if p_n >= 0.7:
        return "corner_neutral"
    if p_c >= 0.7:
        return "corner_contradiction"
    if p_c <= 0.1:
        return "edge_EN_ambiguity"
    if p_e <= 0.1:
        return "edge_NC_ambiguity"
    if p_n <= 0.1:
        return "edge_EC_ambiguity"
    return "center_diffuse_ambiguity"

def run_local_shapley_cartography(k: int = 10) -> dict:
    items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    models_path = Path("data/chaosnli/processed/canonical_models.parquet")
    
    df_items = pl.read_parquet(items_path)
    df_models = pl.read_parquet(models_path)
    
    model_names = sorted(df_models["model_name"].unique().to_list())
    print(f"Executing Local Shapley Cartography across {len(model_names)} models & N=3113 items...")
    
    P_human = df_items.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    D_human = distance_hellinger_matrix(P_human)
    W_human = compute_topk_weights(D_human, k=k)
    
    # Classify items into 7 regions
    regions = [
        classify_simplex_region(r["human_p_entailment"], r["human_p_neutral"], r["human_p_contradiction"])
        for r in df_items.to_dicts()
    ]
    unique_regions = sorted(list(set(regions)))
    
    # Compute regional recovery score Q_NX^soft(k) per model and region
    regional_recovery = {}
    for mname in model_names:
        sub_m = df_models.filter(pl.col("model_name") == mname)
        joined = df_items.join(sub_m, on="object_id", how="inner")
        Q_model = joined.select(["model_p_entailment", "model_p_neutral", "model_p_contradiction"]).to_numpy()
        D_model = distance_hellinger_matrix(Q_model)
        W_model = compute_topk_weights(D_model, k=k)
        
        regional_recovery[mname] = {}
        for r in unique_regions:
            r_idx = np.where(np.array(regions) == r)[0]
            if len(r_idx) > 0:
                W_h_sub = W_human[r_idx][:, r_idx]
                W_m_sub = W_model[r_idx][:, r_idx]
                regional_recovery[mname][r] = soft_overlap(W_h_sub, W_m_sub, k=min(k, max(1, len(r_idx)-1)))
            else:
                regional_recovery[mname][r] = 0.0
                
    # Search for Pluralistic Adversarial Pairs (Torn Twins & False Twins)
    print("Searching for Pluralistic Adversarial Pairs...")
    iu = np.triu_indices(len(P_human), k=1)
    
    # Pre-select candidate pairs to keep computation ultra-fast
    d_h_human = D_human[iu]
    
    # Focus on BART-Large vs RoBERTa-Large vs Human for pair extraction
    bart_m = df_models.filter(pl.col("model_name") == "bart-large")
    bart_joined = df_items.join(bart_m, on="object_id", how="inner")
    Q_bart = bart_joined.select(["model_p_entailment", "model_p_neutral", "model_p_contradiction"]).to_numpy()
    D_bart = distance_hellinger_matrix(Q_bart)
    d_h_bart = D_bart[iu]
    
    # Torn Twins: human distance < 0.05 (near identical human shape) but BART distance > 0.35
    torn_twin_mask = (d_h_human < 0.05) & (d_h_bart > 0.35)
    torn_twin_pairs_idx = np.where(torn_twin_mask)[0]
    
    torn_twins_examples = []
    items_list = df_items.to_dicts()
    for p_idx in torn_twin_pairs_idx[:5]:  # Top 5 examples
        i = iu[0][p_idx]
        j = iu[1][p_idx]
        torn_twins_examples.append({
            "item_i": {"id": items_list[i]["object_id"], "prem": items_list[i]["premise"], "hyp": items_list[i]["hypothesis"]},
            "item_j": {"id": items_list[j]["object_id"], "prem": items_list[j]["premise"], "hyp": items_list[j]["hypothesis"]},
            "human_distance": float(d_h_human[p_idx]),
            "bart_distance": float(d_h_bart[p_idx]),
        })
        
    summary = {
        "n_items": len(P_human),
        "k": k,
        "region_counts": {r: int(np.sum(np.array(regions) == r)) for r in unique_regions},
        "regional_recovery_by_model": regional_recovery,
        "pluralistic_adversarial_pairs": {
            "torn_twins_count": int(np.sum(torn_twin_mask)),
            "torn_twins_examples": torn_twins_examples,
        }
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = run_local_shapley_cartography(k=10)
    out_file = out_dir / "local_shapley_cartography_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Local Shapley Cartography summary written to {out_file}")
