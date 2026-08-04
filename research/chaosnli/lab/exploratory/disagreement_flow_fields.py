"""Module 2: Disagreement Flow Fields (E014).

Estimates smoothed conditional vector fields F_m(p) = E[q_i(m) - p_i | p_i in bin(p)].
Decomposes displacement into:
- Confidence Drift (Cornerward vs Centerward)
- Ambiguity-Type Drift (Edge-parallel vs Orthogonal)
- Boundary Collapse (Interior -> Binary Edge)
- Majority Attraction (Vector alignment toward empirical majority corner)
"""

import os
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
from pathlib import Path
import numpy as np
import polars as pl

def ternary_to_2d(P: np.ndarray) -> np.ndarray:
    p_e = P[:, 0]
    p_n = P[:, 1]
    p_c = P[:, 2]
    x = p_n + 0.5 * p_c
    y = (np.sqrt(3) / 2.0) * p_c
    return np.column_stack([x, y])

def run_disagreement_flow_fields() -> dict:
    items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    models_path = Path("data/chaosnli/processed/canonical_models.parquet")
    
    df_items = pl.read_parquet(items_path)
    df_models = pl.read_parquet(models_path)
    
    model_names = sorted(df_models["model_name"].unique().to_list())
    print(f"Executing Disagreement Flow Fields across {len(model_names)} models & N=3113 items...")
    
    # Define 25 coarse barycentric grid bins over the 2D simplex
    # Bin grid spacing in 2D
    n_bins_side = 5
    x_edges = np.linspace(0.0, 1.0, n_bins_side + 1)
    y_edges = np.linspace(0.0, np.sqrt(3)/2.0, n_bins_side + 1)
    
    results_by_model = {}
    
    # Majority corners in 3D: E=(1,0,0), N=(0,1,0), C=(0,0,1)
    corners = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([0.0, 1.0, 0.0]),
        2: np.array([0.0, 0.0, 1.0]),
    }
    
    for mname in model_names:
        sub_m = df_models.filter(pl.col("model_name") == mname)
        joined = df_items.join(sub_m, on="object_id", how="inner")
        
        P_human = joined.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
        Q_model = joined.select(["model_p_entailment", "model_p_neutral", "model_p_contradiction"]).to_numpy()
        
        # Displacements V_i = Q_i - P_i in 3D probability space
        V_3d = Q_model - P_human
        
        # 1. Majority Attraction: cos theta between V_i and (corner_maj - P_i)
        maj_idx = np.argmax(P_human, axis=1)
        maj_corners = np.array([corners[m] for m in maj_idx])
        vec_to_maj = maj_corners - P_human
        
        norm_v = np.linalg.norm(V_3d, axis=1, keepdims=True) + 1e-12
        norm_maj = np.linalg.norm(vec_to_maj, axis=1, keepdims=True) + 1e-12
        
        cos_maj = np.sum((V_3d / norm_v) * (vec_to_maj / norm_maj), axis=1)
        
        # 2. Confidence Drift: Cornerward (over-sharpening) vs Centerward (over-softening)
        # Distance to simplex center (1/3, 1/3, 1/3)
        center = np.array([1/3, 1/3, 1/3])
        dist_p_center = np.linalg.norm(P_human - center, axis=1)
        dist_q_center = np.linalg.norm(Q_model - center, axis=1)
        cornerward_drift = dist_q_center - dist_p_center  # >0 means moved toward corners
        
        # 3. Boundary Collapse: Motion from interior (min p_c > 0.05) to boundary (min q_c < 0.02)
        min_p = np.min(P_human, axis=1)
        min_q = np.min(Q_model, axis=1)
        boundary_collapsed_items = np.sum((min_p >= 0.05) & (min_q < 0.02))
        
        # Bin flow vectors in 2D
        P_2d = ternary_to_2d(P_human)
        grid_flows = []
        
        for ix in range(n_bins_side):
            for iy in range(n_bins_side):
                mask_bin = (
                    (P_2d[:, 0] >= x_edges[ix]) & (P_2d[:, 0] < x_edges[ix+1]) &
                    (P_2d[:, 1] >= y_edges[iy]) & (P_2d[:, 1] < y_edges[iy+1])
                )
                bin_count = int(np.sum(mask_bin))
                if bin_count >= 5:  # Minimum 5 items per bin
                    mean_p2d = np.mean(P_2d[mask_bin], axis=0)
                    mean_v3d = np.mean(V_3d[mask_bin], axis=0)
                    grid_flows.append({
                        "bin_x": float(mean_p2d[0]),
                        "bin_y": float(mean_p2d[1]),
                        "n_items": bin_count,
                        "v_entailment": float(mean_v3d[0]),
                        "v_neutral": float(mean_v3d[1]),
                        "v_contradiction": float(mean_v3d[2]),
                    })
                    
        results_by_model[mname] = {
            "n_items": len(joined),
            "majority_attraction_cos_mean": float(np.mean(cos_maj)),
            "cornerward_oversharpening_mean": float(np.mean(cornerward_drift)),
            "boundary_collapsed_items_count": int(boundary_collapsed_items),
            "boundary_collapsed_percentage": float((boundary_collapsed_items / len(joined)) * 100.0),
            "grid_flows": grid_flows,
        }
        
    summary = {
        "n_items": 3113,
        "n_models": len(model_names),
        "models": results_by_model,
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = run_disagreement_flow_fields()
    out_file = out_dir / "disagreement_flow_fields_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Disagreement Flow Fields summary written to {out_file}")
