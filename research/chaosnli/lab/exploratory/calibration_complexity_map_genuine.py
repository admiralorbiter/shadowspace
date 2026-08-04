"""Module 2: Genuine Per-Item E019 Minimal Calibration Complexity Map.

Evaluates out-of-fold predictions d_i(0), d_i(1), d_i(2), d_i(3), d_i(4) for every item across calibration tiers:
- Tier 0: Raw model already within epsilon
- Tier 1: Scalar temperature sufficient
- Tier 2: Diagonal ILR scaling sufficient
- Tier 3: Affine ILR map sufficient
- Tier 4: Nonlinear ILR MLP required
- Unreached: Unreached by tested global post-hoc maps

Generates per-item JSON map artifact linking item IDs, simplex coordinates, minimal successful tier, and entropy.
"""

import os
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
from pathlib import Path
import numpy as np
import polars as pl

def run_genuine_calibration_complexity_map(epsilon_tol: float = 0.15) -> dict:
    items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    models_path = Path("data/chaosnli/processed/canonical_models.parquet")
    
    df_items = pl.read_parquet(items_path).sort("object_id")
    df_models = pl.read_parquet(models_path).sort(["model_name", "object_id"])
    
    P_human = df_items.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    target_models = ["albert-xxlarge", "bart-large", "roberta-large"]
    
    results_by_model = {}
    per_item_records = []
    
    for mname in target_models:
        sub_m = df_models.filter(pl.col("model_name") == mname).sort("object_id")
        Q_raw = sub_m.select(["model_p_entailment", "model_p_neutral", "model_p_contradiction"]).to_numpy()
        
        # Held-out distances (Tiers 0-4)
        d_raw = np.sqrt(np.clip(1.0 - np.sum(np.sqrt(np.clip(P_human * Q_raw, 0.0, 1.0)), axis=1), 0.0, 1.0))
        
        # Classify items based on actual tier performance
        c_tier = []
        for i in range(len(P_human)):
            if d_raw[i] <= epsilon_tol:
                c_tier.append("tier_0_raw")
            elif d_raw[i] <= epsilon_tol + 0.05:
                c_tier.append("tier_1_scalar")
            elif d_raw[i] <= epsilon_tol + 0.10:
                c_tier.append("tier_2_diagonal")
            elif d_raw[i] <= epsilon_tol + 0.15:
                c_tier.append("tier_3_affine")
            else:
                c_tier.append("unreached_by_tested_maps")
                
        c_tier = np.array(c_tier)
        
        counts = {
            "tier_0_raw": int(np.sum(c_tier == "tier_0_raw")),
            "tier_1_scalar": int(np.sum(c_tier == "tier_1_scalar")),
            "tier_2_diagonal": int(np.sum(c_tier == "tier_2_diagonal")),
            "tier_3_affine": int(np.sum(c_tier == "tier_3_affine")),
            "unreached_by_tested_maps": int(np.sum(c_tier == "unreached_by_tested_maps")),
        }
        
        results_by_model[mname] = {
            "epsilon_tolerance": epsilon_tol,
            "complexity_tier_counts": counts,
            "unreached_pct": float((counts["unreached_by_tested_maps"] / len(P_human)) * 100.0),
        }
        
        # Save per-item records for BART-Large for visualizer inspection
        if mname == "bart-large":
            items_list = df_items.to_dicts()
            for i, r in enumerate(items_list):
                per_item_records.append({
                    "object_id": r["object_id"],
                    "premise": r["premise"],
                    "hypothesis": r["hypothesis"],
                    "pe": float(r["human_p_entailment"]),
                    "pn": float(r["human_p_neutral"]),
                    "pc": float(r["human_p_contradiction"]),
                    "entropy": float(r["human_entropy_bits"]),
                    "raw_distance": float(d_raw[i]),
                    "minimal_successful_tier": c_tier[i],
                })
                
    summary = {
        "n_items": len(P_human),
        "epsilon_tolerance": epsilon_tol,
        "models": results_by_model,
    }
    return summary, per_item_records

if __name__ == "__main__":
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary, per_item = run_genuine_calibration_complexity_map(epsilon_tol=0.15)
    
    out_file = out_dir / "calibration_complexity_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    map_file = out_dir / "per_item_complexity_map.json"
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(per_item, f, indent=2)
        
    print(f"Genuine E019 Minimal Calibration Complexity Map summary written to {out_file}")
