"""Module 2: Genuine Per-Item E019 Minimal Calibration Complexity Map (Audited).

Reads out-of-fold predictions and distances d_i(0), d_i(1), d_i(2), d_i(3), d_i(4) from oof_predictions.parquet
and evaluates minimal successful calibration tier c_i(epsilon) = min { t in {0,1,2,3,4} : d_i(t) <= epsilon }:
- Tier 0: Raw model already within epsilon
- Tier 1: Scalar temperature sufficient
- Tier 2: Diagonal ILR scaling sufficient
- Tier 3: Affine ILR map sufficient
- Tier 4: Nonlinear ILR MLP required
- Unreached: Unreached by tested global post-hoc maps

Tracks non-monotonic tier regressions (d_i(t+1) > d_i(t)) across epsilon in {0.05, 0.10, 0.15, 0.20}.
Generates per_item_complexity_map.json and calibration_complexity_summary.json.
"""

import os
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
from pathlib import Path
import numpy as np
import polars as pl

def run_genuine_calibration_complexity_map() -> tuple[dict, list]:
    oof_parquet = Path("results/exploratory/oof_predictions.parquet")
    items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    
    if not oof_parquet.exists():
        raise FileNotFoundError(f"Missing required OOF predictions artifact: {oof_parquet}")
        
    df_oof = pl.read_parquet(oof_parquet)
    df_items = pl.read_parquet(items_path).sort("object_id")
    items_dict = {r["object_id"]: r for r in df_items.to_dicts()}
    
    target_models = sorted(df_oof["model_name"].unique().to_list())
    epsilons = [0.05, 0.10, 0.15, 0.20]
    
    results_by_model = {}
    per_item_records = []
    
    tier_names = ["tier_0_raw", "tier_1_scalar", "tier_2_diagonal", "tier_3_affine", "tier_4_mlp"]
    
    for mname in target_models:
        sub_oof = df_oof.filter(pl.col("model_name") == mname).sort("object_id")
        
        d_raw = sub_oof["d_raw"].to_numpy()
        d_t1 = sub_oof["d_t1"].to_numpy()
        d_t2 = sub_oof["d_t2"].to_numpy()
        d_t3 = sub_oof["d_t3"].to_numpy()
        d_t4 = sub_oof["d_t4"].to_numpy()
        
        object_ids = sub_oof["object_id"].to_list()
        fold_ids = sub_oof["fold_id"].to_list()
        
        D_matrix = np.stack([d_raw, d_t1, d_t2, d_t3, d_t4], axis=1) # (N, 5)
        
        eps_summaries = {}
        for eps in epsilons:
            minimal_tiers = []
            regressions_count = 0
            
            for i in range(len(object_ids)):
                d_row = D_matrix[i]
                # Check for later-tier regression
                if np.any(d_row[1:] > d_row[:-1]):
                    regressions_count += 1
                    
                successful_indices = np.where(d_row <= eps)[0]
                if len(successful_indices) > 0:
                    min_idx = int(successful_indices[0])
                    minimal_tiers.append(tier_names[min_idx])
                else:
                    minimal_tiers.append("unreached_by_tested_maps")
                    
            minimal_tiers = np.array(minimal_tiers)
            counts = {
                "tier_0_raw": int(np.sum(minimal_tiers == "tier_0_raw")),
                "tier_1_scalar": int(np.sum(minimal_tiers == "tier_1_scalar")),
                "tier_2_diagonal": int(np.sum(minimal_tiers == "tier_2_diagonal")),
                "tier_3_affine": int(np.sum(minimal_tiers == "tier_3_affine")),
                "tier_4_mlp": int(np.sum(minimal_tiers == "tier_4_mlp")),
                "unreached_by_tested_maps": int(np.sum(minimal_tiers == "unreached_by_tested_maps")),
            }
            
            eps_summaries[f"eps_{eps:.2f}"] = {
                "complexity_tier_counts": counts,
                "unreached_pct": float((counts["unreached_by_tested_maps"] / len(object_ids)) * 100.0),
                "items_with_tier_regressions": regressions_count,
            }
            
        results_by_model[mname] = eps_summaries
        
        # Save detailed per-item records for BART-Large at default eps=0.15
        if mname == "bart-large":
            eps_default = 0.15
            for i in range(len(object_ids)):
                oid = object_ids[i]
                item_info = items_dict[oid]
                d_row = D_matrix[i]
                
                successful = [tier_names[idx] for idx in np.where(d_row <= eps_default)[0]]
                min_tier = successful[0] if len(successful) > 0 else "unreached_by_tested_maps"
                
                has_regression = bool(np.any(d_row[1:] > d_row[:-1]))
                
                per_item_records.append({
                    "object_id": oid,
                    "fold_id": fold_ids[i],
                    "premise": item_info["premise"],
                    "hypothesis": item_info["hypothesis"],
                    "pe": float(item_info["human_p_entailment"]),
                    "pn": float(item_info["human_p_neutral"]),
                    "pc": float(item_info["human_p_contradiction"]),
                    "entropy": float(item_info["human_entropy_bits"]),
                    "d_raw": float(d_raw[i]),
                    "d_t1": float(d_t1[i]),
                    "d_t2": float(d_t2[i]),
                    "d_t3": float(d_t3[i]),
                    "d_t4": float(d_t4[i]),
                    "minimal_successful_tier": min_tier,
                    "successful_tiers": successful,
                    "later_tier_regression": has_regression,
                })
                
    summary = {
        "n_items": len(df_items),
        "epsilons_evaluated": epsilons,
        "models": results_by_model,
    }
    return summary, per_item_records

if __name__ == "__main__":
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary, per_item = run_genuine_calibration_complexity_map()
    
    out_file = out_dir / "calibration_complexity_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    map_file = out_dir / "per_item_complexity_map.json"
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(per_item, f, indent=2)
        
    print(f"Genuine E019 Minimal Calibration Complexity Map summary written to {out_file}")
    print(f"Per-item complexity map JSON written to {map_file}")
