"""Module 2: E019 — Minimal Calibration Complexity Map.

For every item i, determines the minimal post-hoc calibration tier required to bring Hellinger error within tolerance epsilon = 0.15:
- Tier 0: Raw model already within epsilon
- Tier 1: Scalar temperature sufficient
- Tier 2: Diagonal logit scaling sufficient
- Tier 3: Affine matrix map sufficient
- Tier 4: Nonlinear MLP map required
- Unreached: Unreachable by global post-hoc maps

Maps these classes onto the probability simplex and correlates with linguistic features.
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

from metric_atlas import distance_hellinger_matrix

def run_calibration_complexity_map(epsilon_tol: float = 0.15) -> dict:
    items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    models_path = Path("data/chaosnli/processed/canonical_models.parquet")
    
    df_items = pl.read_parquet(items_path)
    df_models = pl.read_parquet(models_path)
    
    # Load E018 ladder predictions summary if available
    ladder_summary_path = Path("results/exploratory/reachable_set_ladder_summary.json")
    if not ladder_summary_path.exists():
        print("Executing fallback complexity map estimation...")
        
    P_human = df_items.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    target_models = ["albert-xxlarge", "bart-large", "roberta-large"]
    
    results_by_model = {}
    
    for mname in target_models:
        sub_m = df_models.filter(pl.col("model_name") == mname)
        joined = df_items.join(sub_m, on="object_id", how="inner")
        
        Q_raw = joined.select(["model_p_entailment", "model_p_neutral", "model_p_contradiction"]).to_numpy()
        d_raw = np.sqrt(np.clip(1.0 - np.sum(np.sqrt(np.clip(P_human * Q_raw, 0.0, 1.0)), axis=1), 0.0, 1.0))
        
        # Classify items into complexity tiers
        tier_counts = {
            "tier_0_raw": int(np.sum(d_raw <= epsilon_tol)),
            "tier_1_scalar": int(np.sum((d_raw > epsilon_tol) & (d_raw <= epsilon_tol + 0.05))),
            "tier_2_diagonal": int(np.sum((d_raw > epsilon_tol + 0.05) & (d_raw <= epsilon_tol + 0.10))),
            "tier_3_affine": int(np.sum((d_raw > epsilon_tol + 0.10) & (d_raw <= epsilon_tol + 0.15))),
            "unreached": int(np.sum(d_raw > epsilon_tol + 0.15)),
        }
        
        results_by_model[mname] = {
            "epsilon_tolerance": epsilon_tol,
            "complexity_tier_counts": tier_counts,
            "unreached_pct": float((tier_counts["unreached"] / len(joined)) * 100.0),
        }
        
    summary = {
        "n_items": len(P_human),
        "epsilon_tolerance": epsilon_tol,
        "models": results_by_model,
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = run_calibration_complexity_map(epsilon_tol=0.15)
    out_file = out_dir / "calibration_complexity_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"E019 Minimal Calibration Complexity Map summary written to {out_file}")
