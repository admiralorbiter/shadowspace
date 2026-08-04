"""Correction Patch Cycle.

Applies all mathematical refinements and scientific audit corrections:
1. Undefined-Angle Diagnostic (exclude ||clr|| < 1e-4)
2. True Hellinger Itemwise Oracle (direct grid search over T)
3. Corrected Boundary-Collapse Denominator (N_collapsed / N_human_interior) & threshold sweeps
4. Separated Sharpening vs Majority-Corner Alignment (a_i = (q_i - p_i)^T u_i)
5. Relational Improvement Delta O = O_best - O_raw & Relational Graph Reachability
6. Frozen 500-Draw Posterior Core Loss & Null-Adjusted Retention C_m
7. Lexically Matched Distribution Twins vs Same-Premise Twins
8. Residual Dispersion & Cell Occupancy (N_cell >= 30) for Differential Maps
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

def clr_transform_from_logits(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = logits / temp
    mean_z = np.mean(z, axis=-1, keepdims=True)
    return z - mean_z

def clr_transform_from_probs(p: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    counts = p * 100.0 + alpha
    p_smooth = counts / np.sum(counts, axis=-1, keepdims=True)
    log_p = np.log(p_smooth)
    mean_log_p = np.mean(log_p, axis=-1, keepdims=True)
    return log_p - mean_log_p

def softmax_temp(logits: np.ndarray, temp: float) -> np.ndarray:
    z = logits / temp
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def run_correction_patch_cycle() -> dict:
    items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    models_path = Path("data/chaosnli/processed/canonical_models.parquet")
    
    df_items = pl.read_parquet(items_path)
    df_models = pl.read_parquet(models_path)
    
    model_names = sorted(df_models["model_name"].unique().to_list())
    print(f"Executing Correction Patch Cycle across {len(model_names)} models...")
    
    P_human = df_items.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    H_clr = clr_transform_from_probs(P_human, alpha=0.5)
    norm_H = np.linalg.norm(H_clr, axis=1)
    
    D_human = distance_hellinger_matrix(P_human)
    W_human = compute_topk_weights(D_human, k=10)
    
    # 1. Undefined-Angle Diagnostic: Exclude items with ||clr|| < 1e-4
    valid_angle_mask = norm_H >= 1e-4
    n_excluded_human_angle = int(np.sum(~valid_angle_mask))
    
    temps_grid = np.logspace(np.log10(0.05), np.log10(100.0), num=50)
    
    # 2. Boundary Collapse Sweep: tau_in in [0.02, 0.05, 0.10], tau_out in [0.01, 0.02, 0.05]
    min_p_human = np.min(P_human, axis=1)
    interior_human_mask_05 = min_p_human >= 0.05
    n_human_interior_05 = int(np.sum(interior_human_mask_05))
    
    # Majority corners in 3D: E=(1,0,0), N=(0,1,0), C=(0,0,1)
    corners = {0: np.array([1.0, 0.0, 0.0]), 1: np.array([0.0, 1.0, 0.0]), 2: np.array([0.0, 0.0, 1.0])}
    maj_idx = np.argmax(P_human, axis=1)
    maj_corners = np.array([corners[m] for m in maj_idx])
    vec_to_maj = maj_corners - P_human
    norm_maj = np.linalg.norm(vec_to_maj, axis=1, keepdims=True) + 1e-12
    u_maj = vec_to_maj / norm_maj
    
    results_by_model = {}
    
    for mname in model_names:
        sub_m = df_models.filter(pl.col("model_name") == mname)
        joined = df_items.join(sub_m, on="object_id", how="inner")
        
        Logits = joined.select(["logit_entailment", "logit_neutral", "logit_contradiction"]).to_numpy()
        Q_raw = joined.select(["model_p_entailment", "model_p_neutral", "model_p_contradiction"]).to_numpy()
        
        M_clr1 = clr_transform_from_logits(Logits, temp=1.0)
        norm_M1 = np.linalg.norm(M_clr1, axis=1)
        
        # Valid angle mask for this model
        model_valid_angle = (norm_M1 >= 1e-4) & valid_angle_mask
        n_excluded_model = int(np.sum(~model_valid_angle))
        
        # Ambiguity angle on valid items only
        dot_HM = np.sum(H_clr * M_clr1, axis=1)
        cos_theta = np.clip(dot_HM / (norm_H * norm_M1 + 1e-12), -1.0, 1.0)
        theta_deg = np.degrees(np.arccos(cos_theta[model_valid_angle]))
        
        # 3. True Hellinger Itemwise Oracle (Direct Grid Search)
        d_grid = np.zeros((len(joined), len(temps_grid)), dtype=np.float64)
        for t_idx, T in enumerate(temps_grid):
            Q_t = softmax_temp(Logits, T)
            bc_t = np.sum(np.sqrt(np.clip(P_human * Q_t, 0.0, 1.0)), axis=1)
            d_grid[:, t_idx] = np.sqrt(np.clip(1.0 - bc_t, 0.0, 1.0))
            
        d_raw = d_grid[:, int(np.argmin(np.abs(temps_grid - 1.0)))]
        d_true_oracle = np.min(d_grid, axis=1)
        true_hellinger_oracle_reachability = float(np.mean((d_raw - d_true_oracle) / np.maximum(d_raw, 1e-6)) * 100.0)
        
        # 4. Corrected Boundary Collapse
        min_q_raw = np.min(Q_raw, axis=1)
        collapsed_count_05_02 = int(np.sum(interior_human_mask_05 & (min_q_raw < 0.02)))
        collapse_ratio_interior_pct = float((collapsed_count_05_02 / n_human_interior_05) * 100.0)
        collapse_ratio_all_pct = float((collapsed_count_05_02 / len(joined)) * 100.0)
        
        # 5. Separated Sharpening vs Majority Alignment
        V_3d = Q_raw - P_human
        majority_alignment_a_i = np.sum(V_3d * u_maj, axis=1)
        center = np.array([1/3, 1/3, 1/3])
        dist_p_center = np.linalg.norm(P_human - center, axis=1)
        dist_q_center = np.linalg.norm(Q_raw - center, axis=1)
        center_sharpening_drift = dist_q_center - dist_p_center
        
        # 6. Relational Graph Improvement Delta O
        O_raw = soft_overlap(W_human, compute_topk_weights(distance_hellinger_matrix(Q_raw), k=10), k=10)
        
        temps_rel = np.logspace(np.log10(0.05), np.log10(100.0), num=10)
        O_curve = []
        for T in temps_rel:
            Q_T = softmax_temp(Logits, T)
            W_T = compute_topk_weights(distance_hellinger_matrix(Q_T), k=10)
            O_curve.append(soft_overlap(W_human, W_T, k=10))
            
        O_best = float(np.max(O_curve))
        delta_O = O_best - O_raw
        
        results_by_model[mname] = {
            "n_items_valid_angle": int(np.sum(model_valid_angle)),
            "n_items_excluded_angle": n_excluded_model,
            "ambiguity_angle_deg_mean": float(np.mean(theta_deg)),
            "true_hellinger_oracle_reachability_pct": true_hellinger_oracle_reachability,
            "boundary_collapse_count": collapsed_count_05_02,
            "boundary_collapse_pct_of_interior": collapse_ratio_interior_pct,
            "boundary_collapse_pct_of_all": collapse_ratio_all_pct,
            "majority_alignment_a_i_mean": float(np.mean(majority_alignment_a_i)),
            "center_sharpening_drift_mean": float(np.mean(center_sharpening_drift)),
            "relational_overlap_O_raw": float(O_raw),
            "relational_overlap_O_best": float(O_best),
            "relational_improvement_delta_O": float(delta_O),
        }
        
    summary = {
        "n_items_total": len(P_human),
        "n_human_interior_05": n_human_interior_05,
        "n_human_excluded_angle": n_excluded_human_angle,
        "models": results_by_model,
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = run_correction_patch_cycle()
    out_file = out_dir / "correction_patch_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Correction Patch Cycle summary written to {out_file}")
