"""Local Intrinsic Dimensionality (LID) Module — TwoNN Estimator & Local PCA Spectrum.

Evaluates whether human ambiguity regions occupy 2D manifolds while models or projections collapse them onto 1D curves.
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

def compute_twonn_lid(dist_matrix: np.ndarray) -> np.ndarray:
    """Compute point-level TwoNN intrinsic dimension estimator (Facco et al., 2017).
    
    mu_i = d_{i,2} / d_{i,1} (excluding self)
    d_i = - ln(1 - F(mu_i)) / ln(mu_i) ~ empirical log(mu_i) scaling
    """
    D = dist_matrix.copy()
    np.fill_diagonal(D, np.inf)
    sorted_D = np.sort(D, axis=1)
    
    r1 = sorted_D[:, 0]
    r2 = sorted_D[:, 1]
    
    # Avoid zero division
    r1 = np.maximum(r1, 1e-7)
    r2 = np.maximum(r2, r1 + 1e-7)
    
    mu = r2 / r1
    # Global TwoNN MLE: d_global = N / sum(ln(mu))
    return mu

def compute_local_pca_participation_ratio(P: np.ndarray, dist_matrix: np.ndarray, k: int = 15) -> np.ndarray:
    """Compute local participation ratio across k-nearest neighborhoods."""
    N = len(P)
    D = dist_matrix.copy()
    np.fill_diagonal(D, np.inf)
    
    local_pr = np.zeros(N, dtype=np.float64)
    for i in range(N):
        nn_idx = np.argsort(D[i])[:k]
        local_P = P[nn_idx]
        centered = local_P - np.mean(local_P, axis=0)
        cov = np.cov(centered, rowvar=False)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigvals = np.maximum(0.0, eigvals)
        sum_e = np.sum(eigvals)
        if sum_e > 1e-12:
            pr = (sum_e ** 2) / np.sum(eigvals ** 2)
        else:
            pr = 1.0
        local_pr[i] = pr
    return local_pr

def run_local_intrinsic_dimension(k_local: int = 15) -> dict:
    parquet_path = Path("data/chaosnli/processed/canonical_items.parquet")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}")
        
    df = pl.read_parquet(parquet_path)
    P = df.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    entropy = df["human_entropy_bits"].to_numpy()
    dataset = df["source_dataset"].to_numpy()
    majority = df["human_majority_label"].to_numpy()
    
    D_hellinger = distance_hellinger_matrix(P)
    
    mu = compute_twonn_lid(D_hellinger)
    global_twonn_dim = float(len(P) / np.sum(np.log(mu)))
    
    local_pr = compute_local_pca_participation_ratio(P, D_hellinger, k=k_local)
    
    # Stratify by Entropy Band
    mask_consensus = entropy < 0.5
    mask_edge = (entropy >= 0.5) & (entropy < 1.0)
    mask_diffuse = entropy >= 1.0
    
    summary = {
        "n_items": len(P),
        "k_local_pca": k_local,
        "global_twonn_intrinsic_dim": global_twonn_dim,
        "global_local_pca_pr_mean": float(np.mean(local_pr)),
        "entropy_band_disaggregation": {
            "consensus_H_lt_05": {
                "n": int(np.sum(mask_consensus)),
                "local_pca_pr_mean": float(np.mean(local_pr[mask_consensus])),
                "twonn_mu_mean": float(np.mean(mu[mask_consensus])),
            },
            "edge_ambiguity_05_10": {
                "n": int(np.sum(mask_edge)),
                "local_pca_pr_mean": float(np.mean(local_pr[mask_edge])),
                "twonn_mu_mean": float(np.mean(mu[mask_edge])),
            },
            "diffuse_center_H_gte_10": {
                "n": int(np.sum(mask_diffuse)),
                "local_pca_pr_mean": float(np.mean(local_pr[mask_diffuse])),
                "twonn_mu_mean": float(np.mean(mu[mask_diffuse])),
            }
        },
        "dataset_disaggregation": {
            "snli": {
                "n": int(np.sum([d.endswith("snli") for d in dataset])),
                "local_pca_pr_mean": float(np.mean(local_pr[np.array([d.endswith("snli") for d in dataset])])),
            },
            "mnli": {
                "n": int(np.sum([d.endswith("mnli") for d in dataset])),
                "local_pca_pr_mean": float(np.mean(local_pr[np.array([d.endswith("mnli") for d in dataset])])),
            }
        }
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("research/chaosnli/artifacts/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = run_local_intrinsic_dimension(k_local=15)
    out_file = out_dir / "local_intrinsic_dimension_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Local Intrinsic Dimension summary written to {out_file}")
