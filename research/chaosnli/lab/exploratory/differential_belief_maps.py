"""Module 4: E017 — Differential Belief Maps (Compression, Rotation, Dispersion).

Estimates conditional model map mu_m(p) = E[q^{(m)} | p] and Jacobian J_m(p) = d mu_m / d p.
Calculates:
- Area Compression A_m(p) = |det J_m(p)|
- Anisotropic Flattening kappa_m(p) = sigma_2 / sigma_1
- Conditional Dispersion Sigma_m(p) = Var(q^{(m)} | p)
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

def run_differential_belief_maps() -> dict:
    items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    models_path = Path("data/chaosnli/processed/canonical_models.parquet")
    
    df_items = pl.read_parquet(items_path)
    df_models = pl.read_parquet(models_path)
    
    model_names = sorted(df_models["model_name"].unique().to_list())
    print(f"Executing Differential Belief Maps (E017) across {len(model_names)} models...")
    
    P_human = df_items.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    P_2d = ternary_to_2d(P_human)
    
    # Grid binning (5x5 grid in 2D simplex)
    n_bins = 5
    x_edges = np.linspace(0.0, 1.0, n_bins + 1)
    y_edges = np.linspace(0.0, np.sqrt(3)/2.0, n_bins + 1)
    
    results_by_model = {}
    
    for mname in model_names:
        sub_m = df_models.filter(pl.col("model_name") == mname)
        joined = df_items.join(sub_m, on="object_id", how="inner")
        
        Q_model = joined.select(["model_p_entailment", "model_p_neutral", "model_p_contradiction"]).to_numpy()
        Q_2d = ternary_to_2d(Q_model)
        
        cells = []
        cell_compressions = []
        cell_anisotropies = []
        cell_dispersions = []
        
        for ix in range(n_bins):
            for iy in range(n_bins):
                mask = (
                    (P_2d[:, 0] >= x_edges[ix]) & (P_2d[:, 0] < x_edges[ix+1]) &
                    (P_2d[:, 1] >= y_edges[iy]) & (P_2d[:, 1] < y_edges[iy+1])
                )
                bin_count = int(np.sum(mask))
                if bin_count >= 10:  # Require 10 items for reliable covariance estimation
                    P_sub = P_2d[mask]
                    Q_sub = Q_2d[mask]
                    
                    # Estimate Jacobian via linear regression Q_sub = P_sub @ J^T + c
                    # Center the data
                    P_centered = P_sub - np.mean(P_sub, axis=0)
                    Q_centered = Q_sub - np.mean(Q_sub, axis=0)
                    
                    # Pseudo-inverse regression for 2x2 Jacobian
                    J = np.linalg.pinv(P_centered) @ Q_centered
                    
                    det_J = float(np.abs(np.linalg.det(J)))
                    svd_vals = np.linalg.svd(J, compute_uv=False)
                    sigma1, sigma2 = svd_vals[0], svd_vals[1]
                    anisotropy = float(sigma2 / (sigma1 + 1e-12))
                    
                    # Conditional Dispersion Covariance Tr(Var(Q | P_bin))
                    cov_Q = np.cov(Q_sub.T)
                    trace_dispersion = float(np.trace(cov_Q)) if cov_Q.ndim == 2 else 0.0
                    
                    cell_compressions.append(det_J)
                    cell_anisotropies.append(anisotropy)
                    cell_dispersions.append(trace_dispersion)
                    
                    cells.append({
                        "bin_center_x": float(np.mean(P_sub[:, 0])),
                        "bin_center_y": float(np.mean(P_sub[:, 1])),
                        "n_items": bin_count,
                        "area_compression": det_J,
                        "anisotropic_ratio": anisotropy,
                        "conditional_dispersion_trace": trace_dispersion,
                    })
                    
        results_by_model[mname] = {
            "n_valid_cells": len(cells),
            "area_compression_mean": float(np.mean(cell_compressions)) if cell_compressions else 1.0,
            "anisotropic_flattening_mean": float(np.mean(cell_anisotropies)) if cell_anisotropies else 1.0,
            "conditional_dispersion_trace_mean": float(np.mean(cell_dispersions)) if cell_dispersions else 0.0,
            "grid_cells": cells,
        }
        
    summary = {
        "n_items": len(P_human),
        "models": results_by_model,
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = run_differential_belief_maps()
    out_file = out_dir / "differential_belief_maps_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Differential Belief Maps summary written to {out_file}")
