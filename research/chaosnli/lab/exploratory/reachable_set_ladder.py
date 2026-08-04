"""Module 2: E018 — Reachable-Set Ladder.

Measures how many post-hoc calibration degrees of freedom are required to rotate model ambiguity toward human targets:
- Tier 1: Scalar Temperature (1 DoF) — Positive Ray in CLR space
- Tier 2: Classwise Vector Scaling (3 DoF) — Positive Cone in CLR space
- Tier 3: Affine Matrix Scaling with Bias (12 DoF) — Affine Subspace
- Tier 4: Dirichlet / Simplex Regression (Flexible Simplex Surface)

For each tier, evaluates NLL, Hellinger distance, ambiguity angle reduction Delta theta, and relational graph overlap Q_NX^soft(10).
"""

import os
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import minimize

import sys
sys.path.insert(0, str(Path(__file__).parent))

from metric_atlas import distance_hellinger_matrix, compute_topk_weights, soft_overlap
from correction_patch_cycle import clr_transform_from_logits, clr_transform_from_probs

def run_reachable_set_ladder() -> dict:
    items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    models_path = Path("data/chaosnli/processed/canonical_models.parquet")
    
    df_items = pl.read_parquet(items_path)
    df_models = pl.read_parquet(models_path)
    
    model_names = sorted(df_models["model_name"].unique().to_list())
    print(f"Executing E018 Reachable-Set Ladder across {len(model_names)} models...")
    
    P_human = df_items.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    H_clr = clr_transform_from_probs(P_human, alpha=0.5)
    norm_H = np.linalg.norm(H_clr, axis=1)
    
    D_human = distance_hellinger_matrix(P_human)
    W_human = compute_topk_weights(D_human, k=10)
    
    results_by_model = {}
    
    # We evaluate BART-Large, RoBERTa-Large, and ALBERT-xxLarge for the ladder
    target_models = ["albert-xxlarge", "bart-large", "roberta-large"]
    
    for mname in target_models:
        sub_m = df_models.filter(pl.col("model_name") == mname)
        joined = df_items.join(sub_m, on="object_id", how="inner")
        
        Logits = joined.select(["logit_entailment", "logit_neutral", "logit_contradiction"]).to_numpy()
        Q_raw = joined.select(["model_p_entailment", "model_p_neutral", "model_p_contradiction"]).to_numpy()
        
        M_clr1 = clr_transform_from_logits(Logits, temp=1.0)
        norm_M1 = np.linalg.norm(M_clr1, axis=1)
        
        # Raw baseline metrics
        d_raw = np.sqrt(np.clip(1.0 - np.sum(np.sqrt(np.clip(P_human * Q_raw, 0.0, 1.0)), axis=1), 0.0, 1.0))
        d_raw_mean = float(np.mean(d_raw))
        
        cos_raw = np.clip(np.sum(H_clr * M_clr1, axis=1) / (norm_H * norm_M1 + 1e-12), -1.0, 1.0)
        theta_raw_deg = float(np.mean(np.degrees(np.arccos(cos_raw))))
        O_raw = float(soft_overlap(W_human, compute_topk_weights(distance_hellinger_matrix(Q_raw), k=10), k=10))
        
        # --- Tier 1: Scalar Temperature (1 DoF) ---
        def loss_t1(t):
            T = max(1e-3, t[0])
            z = Logits / T
            z_max = np.max(z, axis=1, keepdims=True)
            q = np.exp(z - z_max) / np.sum(np.exp(z - z_max), axis=1, keepdims=True)
            bc = np.sum(np.sqrt(np.clip(P_human * q, 0.0, 1.0)), axis=1)
            return float(np.mean(np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))))
            
        res_t1 = minimize(loss_t1, [1.0], method="Nelder-Mead")
        best_T = max(1e-3, float(res_t1.x[0]))
        
        # Compute Tier 1 metrics
        z1 = Logits / best_T
        q1 = np.exp(z1 - np.max(z1, axis=1, keepdims=True)) / np.sum(np.exp(z1 - np.max(z1, axis=1, keepdims=True)), axis=1, keepdims=True)
        d_t1_mean = float(loss_t1([best_T]))
        
        M_clr_t1 = clr_transform_from_logits(Logits, temp=best_T)
        cos_t1 = np.clip(np.sum(H_clr * M_clr_t1, axis=1) / (norm_H * np.linalg.norm(M_clr_t1, axis=1) + 1e-12), -1.0, 1.0)
        theta_t1_deg = float(np.mean(np.degrees(np.arccos(cos_t1))))
        O_t1 = float(soft_overlap(W_human, compute_topk_weights(distance_hellinger_matrix(q1), k=10), k=10))
        
        # --- Tier 2: Classwise Vector Scaling (3 DoF: w_1, w_2, w_3) ---
        def loss_t2(w):
            w_arr = np.array(w)
            z = Logits * w_arr
            z_max = np.max(z, axis=1, keepdims=True)
            q = np.exp(z - z_max) / np.sum(np.exp(z - z_max), axis=1, keepdims=True)
            bc = np.sum(np.sqrt(np.clip(P_human * q, 0.0, 1.0)), axis=1)
            return float(np.mean(np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))))
            
        res_t2 = minimize(loss_t2, [1.0, 1.0, 1.0], method="Nelder-Mead")
        best_w = np.array(res_t2.x)
        d_t2_mean = float(res_t2.fun)
        
        z2 = Logits * best_w
        q2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True)) / np.sum(np.exp(z2 - np.max(z2, axis=1, keepdims=True)), axis=1, keepdims=True)
        M_clr_t2 = z2 - np.mean(z2, axis=1, keepdims=True)
        cos_t2 = np.clip(np.sum(H_clr * M_clr_t2, axis=1) / (norm_H * np.linalg.norm(M_clr_t2, axis=1) + 1e-12), -1.0, 1.0)
        theta_t2_deg = float(np.mean(np.degrees(np.arccos(cos_t2))))
        O_t2 = float(soft_overlap(W_human, compute_topk_weights(distance_hellinger_matrix(q2), k=10), k=10))
        
        # --- Tier 3: Affine Matrix Scaling (12 DoF: 3x3 W + 3 b) ---
        def loss_t3(params):
            W_mat = params[:9].reshape((3, 3))
            b_vec = params[9:]
            z = Logits @ W_mat.T + b_vec
            z_max = np.max(z, axis=1, keepdims=True)
            q = np.exp(z - z_max) / np.sum(np.exp(z - z_max), axis=1, keepdims=True)
            bc = np.sum(np.sqrt(np.clip(P_human * q, 0.0, 1.0)), axis=1)
            return float(np.mean(np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))))
            
        init_t3 = np.zeros(12)
        init_t3[0], init_t3[4], init_t3[8] = 1.0, 1.0, 1.0  # Identity matrix
        res_t3 = minimize(loss_t3, init_t3, method="L-BFGS-B")
        d_t3_mean = float(res_t3.fun)
        
        W_mat = res_t3.x[:9].reshape((3, 3))
        b_vec = res_t3.x[9:]
        z3 = Logits @ W_mat.T + b_vec
        q3 = np.exp(z3 - np.max(z3, axis=1, keepdims=True)) / np.sum(np.exp(z3 - np.max(z3, axis=1, keepdims=True)), axis=1, keepdims=True)
        M_clr_t3 = z3 - np.mean(z3, axis=1, keepdims=True)
        cos_t3 = np.clip(np.sum(H_clr * M_clr_t3, axis=1) / (norm_H * np.linalg.norm(M_clr_t3, axis=1) + 1e-12), -1.0, 1.0)
        theta_t3_deg = float(np.mean(np.degrees(np.arccos(cos_t3))))
        O_t3 = float(soft_overlap(W_human, compute_topk_weights(distance_hellinger_matrix(q3), k=10), k=10))
        
        results_by_model[mname] = {
            "tier_0_raw": {
                "d_hellinger_mean": d_raw_mean,
                "ambiguity_angle_deg_mean": theta_raw_deg,
                "relational_overlap_O": O_raw,
            },
            "tier_1_scalar_temp": {
                "dof": 1,
                "best_T": float(best_T),
                "d_hellinger_mean": d_t1_mean,
                "ambiguity_angle_deg_mean": theta_t1_deg,
                "angle_reduction_deg": float(theta_raw_deg - theta_t1_deg),
                "relational_overlap_O": O_t1,
            },
            "tier_2_vector_scaling": {
                "dof": 3,
                "best_weights": [float(w) for w in best_w],
                "d_hellinger_mean": d_t2_mean,
                "ambiguity_angle_deg_mean": theta_t2_deg,
                "angle_reduction_deg": float(theta_raw_deg - theta_t2_deg),
                "relational_overlap_O": O_t2,
            },
            "tier_3_affine_matrix": {
                "dof": 12,
                "d_hellinger_mean": d_t3_mean,
                "ambiguity_angle_deg_mean": theta_t3_deg,
                "angle_reduction_deg": float(theta_raw_deg - theta_t3_deg),
                "relational_overlap_O": O_t3,
            }
        }
        
    summary = {
        "n_items": len(P_human),
        "target_models": target_models,
        "models": results_by_model,
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = run_reachable_set_ladder()
    out_file = out_dir / "reachable_set_ladder_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"E018 Reachable-Set Ladder summary written to {out_file}")
