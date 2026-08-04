"""Module 1: Calibration Orbit Atlas (E015).

Traces temperature scaling paths T -> q_i(T) for T in [0.05, 100] across 50 log-spaced steps for all 9 baseline models.
Compares d_i^orbit = min_T H(p_i, q_i(T)) vs d_i^raw = H(p_i, q_i(1.0)).
Geometrically separates sharpness error from directional / ambiguity-type error.
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

def softmax_with_temp(logits: np.ndarray, temp: float) -> np.ndarray:
    z = logits / temp
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def hellinger_single(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Vectorized pointwise Hellinger distance between rows of p and q."""
    sqrt_p = np.sqrt(np.clip(p, 0.0, 1.0))
    sqrt_q = np.sqrt(np.clip(q, 0.0, 1.0))
    bc = np.sum(sqrt_p * sqrt_q, axis=1)
    bc = np.clip(bc, 0.0, 1.0)
    return np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))

def run_calibration_orbit_atlas() -> dict:
    items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    models_path = Path("data/chaosnli/processed/canonical_models.parquet")
    
    if not items_path.exists() or not models_path.exists():
        raise FileNotFoundError("Missing canonical dataset files")
        
    df_items = pl.read_parquet(items_path)
    df_models = pl.read_parquet(models_path)
    
    # 50 log-spaced temperatures from 0.05 to 100.0
    temps = np.logspace(np.log10(0.05), np.log10(100.0), num=50)
    
    model_names = sorted(df_models["model_name"].unique().to_list())
    print(f"Executing Calibration Orbit Atlas over {len(model_names)} models & N=3113 items across {len(temps)} temperature steps...")
    
    results_by_model = {}
    
    for mname in model_names:
        sub_m = df_models.filter(pl.col("model_name") == mname)
        # Join items to ensure exact row alignment
        joined = df_items.join(sub_m, on="object_id", how="inner")
        
        P_human = joined.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
        Logits = joined.select(["logit_entailment", "logit_neutral", "logit_contradiction"]).to_numpy()
        
        N = len(joined)
        
        # Compute Hellinger distance matrix across all 50 temps: (N, 50)
        d_orbit_matrix = np.zeros((N, len(temps)), dtype=np.float64)
        
        for t_idx, T in enumerate(temps):
            Q_t = softmax_with_temp(Logits, T)
            d_orbit_matrix[:, t_idx] = hellinger_single(P_human, Q_t)
            
        # Raw distance (at T=1.0, index closest to 1.0)
        idx_t1 = int(np.argmin(np.abs(temps - 1.0)))
        d_raw = d_orbit_matrix[:, idx_t1]
        
        # Orbit min distance
        idx_min_orbit = np.argmin(d_orbit_matrix, axis=1)
        d_min_orbit = np.min(d_orbit_matrix, axis=1)
        t_optimal = temps[idx_min_orbit]
        
        # Directional error = d_min_orbit (unreachable error by temperature scaling)
        # Sharpness error = d_raw - d_min_orbit (fixable error by temperature scaling)
        sharpness_error = d_raw - d_min_orbit
        directional_error = d_min_orbit
        
        # Percentage of distance unreachable by scalar temperature calibration
        unreachable_ratio = np.mean(directional_error / np.maximum(d_raw, 1e-6))
        
        results_by_model[mname] = {
            "n_items": N,
            "d_raw_mean": float(np.mean(d_raw)),
            "d_min_orbit_mean": float(np.mean(d_min_orbit)),
            "sharpness_error_mean": float(np.mean(sharpness_error)),
            "directional_error_mean": float(np.mean(directional_error)),
            "unreachable_error_percentage": float(unreachable_ratio * 100.0),
            "optimal_temp_median": float(np.median(t_optimal)),
            "optimal_temp_mean": float(np.mean(t_optimal)),
        }
        
    summary = {
        "n_items": 3113,
        "n_models": len(model_names),
        "temperature_grid_count": len(temps),
        "temp_min": float(temps[0]),
        "temp_max": float(temps[-1]),
        "models": results_by_model,
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = run_calibration_orbit_atlas()
    out_file = out_dir / "calibration_orbit_atlas_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Calibration Orbit Atlas summary written to {out_file}")
