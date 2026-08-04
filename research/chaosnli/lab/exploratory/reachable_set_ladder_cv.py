"""Module 1: 5-Fold Cross-Validated E018 Reachable-Set Ladder (Tiers 1-4).

Evaluates calibration reachability on held-out test folds across 4 tiers:
- Tier 0: Raw Uncalibrated Baseline (Exact T=1.0 frozen Q_raw)
- Tier 1: Scalar Temperature (1 DoF: Positive Ray in CLR space)
- Tier 2: Diagonal Logit Scaling (3 params / 2 CLR DoF, constrained w = exp(s) > 0)
- Tier 3: Affine Softmax Map (12 params / 8 identifiable DoF / 6 CLR DoF)
- Tier 4: Nonlinear Simplex Map (2-layer MLP z -> MLP(z))
"""

import os
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

import sys
sys.path.insert(0, str(Path(__file__).parent))

from metric_atlas import distance_hellinger_matrix, compute_topk_weights, soft_overlap
from correction_patch_cycle import clr_transform_from_logits, clr_transform_from_probs

def softmax_np(z: np.ndarray) -> np.ndarray:
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def run_reachable_set_ladder_cv() -> dict:
    items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    models_path = Path("data/chaosnli/processed/canonical_models.parquet")
    
    df_items = pl.read_parquet(items_path)
    df_models = pl.read_parquet(models_path)
    
    model_names = sorted(df_models["model_name"].unique().to_list())
    print(f"Executing 5-Fold CV E018 Reachable-Set Ladder across {len(model_names)} models...")
    
    P_human = df_items.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    H_clr = clr_transform_from_probs(P_human, alpha=0.5)
    norm_H = np.linalg.norm(H_clr, axis=1)
    
    D_human = distance_hellinger_matrix(P_human)
    W_human = compute_topk_weights(D_human, k=10)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=20260803)
    splits = list(kf.split(P_human))
    
    target_models = ["albert-xxlarge", "bart-large", "roberta-large"]
    results_by_model = {}
    
    for mname in target_models:
        sub_m = df_models.filter(pl.col("model_name") == mname)
        joined = df_items.join(sub_m, on="object_id", how="inner")
        
        Logits = joined.select(["logit_entailment", "logit_neutral", "logit_contradiction"]).to_numpy()
        Q_raw = joined.select(["model_p_entailment", "model_p_neutral", "model_p_contradiction"]).to_numpy()
        
        # Raw baseline (Exact T=1.0)
        d_raw = np.sqrt(np.clip(1.0 - np.sum(np.sqrt(np.clip(P_human * Q_raw, 0.0, 1.0)), axis=1), 0.0, 1.0))
        d_raw_mean = float(np.mean(d_raw))
        
        M_clr_raw = clr_transform_from_logits(Logits, temp=1.0)
        norm_M_raw = np.linalg.norm(M_clr_raw, axis=1)
        cos_raw = np.clip(np.sum(H_clr * M_clr_raw, axis=1) / (norm_H * norm_M_raw + 1e-12), -1.0, 1.0)
        theta_raw_deg = float(np.mean(np.degrees(np.arccos(cos_raw))))
        O_raw = float(soft_overlap(W_human, compute_topk_weights(distance_hellinger_matrix(Q_raw), k=10), k=10))
        
        # Store held-out predictions for 5 folds
        Q_t1_cv = np.zeros_like(Q_raw)
        Q_t2_cv = np.zeros_like(Q_raw)
        Q_t3_cv = np.zeros_like(Q_raw)
        Q_t4_cv = np.zeros_like(Q_raw)
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            P_tr, Logits_tr = P_human[train_idx], Logits[train_idx]
            P_te, Logits_te = P_human[test_idx], Logits[test_idx]
            
            # --- Tier 1: Scalar Temp ---
            def loss_t1(t):
                T = max(1e-3, t[0])
                q = softmax_np(Logits_tr / T)
                bc = np.sum(np.sqrt(np.clip(P_tr * q, 0.0, 1.0)), axis=1)
                return float(np.mean(np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))))
            res_t1 = minimize(loss_t1, [1.0], method="Nelder-Mead")
            T_opt = max(1e-3, float(res_t1.x[0]))
            Q_t1_cv[test_idx] = softmax_np(Logits_te / T_opt)
            
            # --- Tier 2: Diagonal Logit Scaling (w = exp(s) > 0) ---
            def loss_t2(s):
                w = np.exp(np.array(s))
                q = softmax_np(Logits_tr * w)
                bc = np.sum(np.sqrt(np.clip(P_tr * q, 0.0, 1.0)), axis=1)
                return float(np.mean(np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))))
            res_t2 = minimize(loss_t2, [0.0, 0.0, 0.0], method="Nelder-Mead")
            w_opt = np.exp(np.array(res_t2.x))
            Q_t2_cv[test_idx] = softmax_np(Logits_te * w_opt)
            
            # --- Tier 3: Affine Softmax Map (12 params) ---
            def loss_t3(params):
                W_m = params[:9].reshape((3, 3))
                b_v = params[9:]
                q = softmax_np(Logits_tr @ W_m.T + b_v)
                bc = np.sum(np.sqrt(np.clip(P_tr * q, 0.0, 1.0)), axis=1)
                return float(np.mean(np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))))
            init_t3 = np.zeros(12)
            init_t3[0], init_t3[4], init_t3[8] = 1.0, 1.0, 1.0
            res_t3 = minimize(loss_t3, init_t3, method="L-BFGS-B")
            W_opt = res_t3.x[:9].reshape((3, 3))
            b_opt = res_t3.x[9:]
            Q_t3_cv[test_idx] = softmax_np(Logits_te @ W_opt.T + b_opt)
            
            # --- Tier 4: Nonlinear Simplex Map (MLP) ---
            mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=20260803)
            mlp.fit(Logits_tr, P_tr)
            out_te_logits = mlp.predict(Logits_te)
            Q_t4_cv[test_idx] = softmax_np(out_te_logits)

        # Compute held-out 5-fold CV metrics for all tiers
        def eval_predictions(Q_pred):
            d_err = np.sqrt(np.clip(1.0 - np.sum(np.sqrt(np.clip(P_human * Q_pred, 0.0, 1.0)), axis=1), 0.0, 1.0))
            M_clr = np.log(np.clip(Q_pred, 1e-8, 1.0)) - np.mean(np.log(np.clip(Q_pred, 1e-8, 1.0)), axis=1, keepdims=True)
            norm_M = np.linalg.norm(M_clr, axis=1)
            cos_th = np.clip(np.sum(H_clr * M_clr, axis=1) / (norm_H * norm_M + 1e-12), -1.0, 1.0)
            th_deg = float(np.mean(np.degrees(np.arccos(cos_th))))
            O_overlap = float(soft_overlap(W_human, compute_topk_weights(distance_hellinger_matrix(Q_pred), k=10), k=10))
            return float(np.mean(d_err)), th_deg, O_overlap

        d_t1, th_t1, O_t1 = eval_predictions(Q_t1_cv)
        d_t2, th_t2, O_t2 = eval_predictions(Q_t2_cv)
        d_t3, th_t3, O_t3 = eval_predictions(Q_t3_cv)
        d_t4, th_t4, O_t4 = eval_predictions(Q_t4_cv)
        
        results_by_model[mname] = {
            "tier_0_raw": {
                "d_hellinger_mean": d_raw_mean,
                "ambiguity_angle_deg_mean": theta_raw_deg,
                "relational_overlap_O": O_raw,
            },
            "tier_1_scalar_temp": {
                "dof_description": "1 DoF (Positive Ray in CLR space)",
                "heldout_d_hellinger_mean": d_t1,
                "heldout_ambiguity_angle_deg_mean": th_t1,
                "angle_reduction_deg": float(theta_raw_deg - th_t1),
                "heldout_relational_overlap_O": O_t1,
            },
            "tier_2_diagonal_scaling": {
                "dof_description": "3 params / 2 CLR DoF (Positive Cone, w = exp(s) > 0)",
                "heldout_d_hellinger_mean": d_t2,
                "heldout_ambiguity_angle_deg_mean": th_t2,
                "angle_reduction_deg": float(theta_raw_deg - th_t2),
                "heldout_relational_overlap_O": O_t2,
            },
            "tier_3_affine_softmax": {
                "dof_description": "12 params / 8 identifiable DoF / 6 CLR DoF (Affine Subspace)",
                "heldout_d_hellinger_mean": d_t3,
                "heldout_ambiguity_angle_deg_mean": th_t3,
                "angle_reduction_deg": float(theta_raw_deg - th_t3),
                "heldout_relational_overlap_O": O_t3,
            },
            "tier_4_nonlinear_mlp": {
                "dof_description": "2-Layer MLP (Flexible Simplex Surface)",
                "heldout_d_hellinger_mean": d_t4,
                "heldout_ambiguity_angle_deg_mean": th_t4,
                "angle_reduction_deg": float(theta_raw_deg - th_t4),
                "heldout_relational_overlap_O": O_t4,
            }
        }
        
    summary = {
        "n_items": len(P_human),
        "cv_folds": 5,
        "models": results_by_model,
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = run_reachable_set_ladder_cv()
    out_file = out_dir / "reachable_set_ladder_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"5-Fold CV E018 Reachable-Set Ladder summary written to {out_file}")
