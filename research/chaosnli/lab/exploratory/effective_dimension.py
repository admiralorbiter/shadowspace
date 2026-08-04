"""Effective Dimension Estimates — Manifold dimensionality on the 2-simplex.
"""

import os
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
from pathlib import Path
import numpy as np

def run_effective_dimension(P: np.ndarray) -> dict:
    # 2-simplex has full ambient dimension 3, constrained dimension 2 (sum to 1).
    # Estimate local PCA eigenvalues across 3-class simplex representation.
    centered = P - np.mean(P, axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    
    var_explained = eigvals / np.sum(eigvals)
    cum_var = np.cumsum(var_explained)
    
    # Participation ratio (effective dimension)
    eff_dim_pr = float((np.sum(eigvals) ** 2) / np.sum(eigvals ** 2))
    
    # Square-root simplex representation
    sqrt_P = np.sqrt(np.clip(P, 1e-12, 1.0))
    centered_sqrt = sqrt_P - np.mean(sqrt_P, axis=0)
    cov_sqrt = np.cov(centered_sqrt, rowvar=False)
    eigvals_sqrt = np.sort(np.linalg.eigvalsh(cov_sqrt))[::-1]
    var_explained_sqrt = eigvals_sqrt / np.sum(eigvals_sqrt)
    eff_dim_sqrt_pr = float((np.sum(eigvals_sqrt) ** 2) / np.sum(eigvals_sqrt ** 2))
    
    summary = {
        "n_items": int(len(P)),
        "probability_space": {
            "eigenvalues": eigvals.tolist(),
            "variance_explained": var_explained.tolist(),
            "cumulative_variance": cum_var.tolist(),
            "participation_ratio_effective_dim": eff_dim_pr,
        },
        "square_root_simplex_space": {
            "eigenvalues": eigvals_sqrt.tolist(),
            "variance_explained": var_explained_sqrt.tolist(),
            "participation_ratio_effective_dim": eff_dim_sqrt_pr,
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
        
    summary = run_effective_dimension(P)
    out_file = out_dir / "effective_dimension_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Effective Dimension summary written to {out_file}")
