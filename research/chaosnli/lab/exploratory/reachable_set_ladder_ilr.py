"""Module 1: Gauge-Invariant ILR E018 Ladder & Fold-Coherent Relational Evaluation (Audited).

Executes the calibration ladder in 2D Orthonormal Isometric Log-Ratio (ILR) space:
x1 = (log q1 - log q2) / sqrt(2)
x2 = (log q1 + log q2 - 2 log q3) / sqrt(6)

Tiers:
- Tier 0 (Raw): x_i' = x_i (exact frozen Q_raw)
- Tier 1 (Scalar Temp, 1 DoF): x_i' = alpha * x_i (alpha = 1/T > 0)
- Tier 2 (Diagonal ILR Scaling, 2 DoF): x_i' = diag(a, b) * x_i (a, b > 0)
- Tier 3 (Affine ILR Map, 6 DoF): x_i' = A * x_i + b (A in R^{2x2}, b in R^2)
- Tier 4 (Nonlinear ILR MLP): x_i' = MLP(x_i) trained in ILR space with Dirichlet-smoothed targets.

Evaluates fold-coherent relational recovery: for each fold f, applies fold f's fitted map to ALL 3,113 items
to build graph W^{(f)}, scoring ONLY focal rows i in TestFold_f against posterior support matrix S_ij.
Exports held-out out-of-fold predictions to results/exploratory/oof_predictions.parquet.
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
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent))

from metric_atlas import distance_hellinger_matrix, compute_topk_weights, soft_overlap
from correction_patch_cycle import clr_transform_from_probs

def probs_to_ilr(p: np.ndarray, alpha: float = 1e-12) -> np.ndarray:
    """Isometric Log-Ratio (ILR) transform for 3-class probability vectors to 2D orthonormal space."""
    p_safe = np.clip(p, alpha, 1.0)
    p_safe = p_safe / np.sum(p_safe, axis=-1, keepdims=True)
    log_p = np.log(p_safe)
    x1 = (log_p[..., 0] - log_p[..., 1]) / np.sqrt(2.0)
    x2 = (log_p[..., 0] + log_p[..., 1] - 2.0 * log_p[..., 2]) / np.sqrt(6.0)
    return np.stack([x1, x2], axis=-1)

def ilr_to_probs(x: np.ndarray) -> np.ndarray:
    """Inverse ILR transform from 2D orthonormal space to 3-class probability simplex."""
    x1 = x[..., 0]
    x2 = x[..., 1]
    clr1 = x1 / np.sqrt(2.0) + x2 / np.sqrt(6.0)
    clr2 = -x1 / np.sqrt(2.0) + x2 / np.sqrt(6.0)
    clr3 = -2.0 * x2 / np.sqrt(6.0)
    clr = np.stack([clr1, clr2, clr3], axis=-1)
    clr_max = np.max(clr, axis=-1, keepdims=True)
    exp_clr = np.exp(clr - clr_max)
    return exp_clr / np.sum(exp_clr, axis=-1, keepdims=True)

def compute_posterior_support_matrix(P: np.ndarray, k: int = 10, n_draws: int = 50, seed: int = 20260803) -> np.ndarray:
    N = len(P)
    edge_counts = np.zeros((N, N), dtype=np.float64)
    counts = P * 100.0 + 0.5
    rng = np.random.default_rng(seed)
    
    for _ in range(n_draws):
        P_draw = np.zeros_like(P)
        for i in range(N):
            P_draw[i] = rng.dirichlet(counts[i])
        D_draw = distance_hellinger_matrix(P_draw)
        W_draw = compute_topk_weights(D_draw, k=k)
        edge_counts += (W_draw > 0).astype(float)
        
    return edge_counts / float(n_draws)

def run_reachable_set_ladder_ilr() -> tuple[dict, pl.DataFrame]:
    items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    models_path = Path("data/chaosnli/processed/canonical_models.parquet")
    
    df_items = pl.read_parquet(items_path).sort("object_id")
    df_models = pl.read_parquet(models_path).sort(["model_name", "object_id"])
    
    item_ids = df_items["object_id"].to_list()
    
    P_human = df_items.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    # Dirichlet-smoothed human ILR targets for Tier 4 MLP
    P_human_smooth = (P_human * 100.0 + 0.5) / 101.5
    X_human_ilr = probs_to_ilr(P_human_smooth)
    
    H_clr = clr_transform_from_probs(P_human, alpha=0.5)
    norm_H = np.linalg.norm(H_clr, axis=1)
    
    D_human = distance_hellinger_matrix(P_human)
    W_human = compute_topk_weights(D_human, k=10)
    
    print("Computing Dirichlet posterior support matrix S_ij (50 draws)...")
    S_posterior = compute_posterior_support_matrix(P_human, k=10, n_draws=50)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=20260803)
    splits = list(kf.split(P_human))
    
    target_models = ["albert-xxlarge", "bart-large", "roberta-large"]
    results_by_model = {}
    oof_dfs = []
    
    for mname in target_models:
        sub_m = df_models.filter(pl.col("model_name") == mname).sort("object_id")
        model_ids = sub_m["object_id"].to_list()
        assert item_ids == model_ids, f"Object ID mismatch for model {mname}"
        
        Q_raw = sub_m.select(["model_p_entailment", "model_p_neutral", "model_p_contradiction"]).to_numpy()
        X_raw_ilr = probs_to_ilr(Q_raw)
        
        # Raw Hellinger baseline (Exact frozen Q_raw)
        d_raw = np.sqrt(np.clip(1.0 - np.sum(np.sqrt(np.clip(P_human * Q_raw, 0.0, 1.0)), axis=1), 0.0, 1.0))
        d_raw_mean = float(np.mean(d_raw))
        
        M_clr_raw = np.log(np.clip(Q_raw, 1e-8, 1.0)) - np.mean(np.log(np.clip(Q_raw, 1e-8, 1.0)), axis=1, keepdims=True)
        cos_raw = np.clip(np.sum(H_clr * M_clr_raw, axis=1) / (norm_H * np.linalg.norm(M_clr_raw, axis=1) + 1e-12), -1.0, 1.0)
        theta_raw_deg = float(np.mean(np.degrees(np.arccos(cos_raw))))
        
        # CORRECTED Raw model posterior support Q_support_raw (using raw model graph W_raw)
        W_raw = compute_topk_weights(distance_hellinger_matrix(Q_raw), k=10)
        Q_support_raw = float(np.mean(np.sum(W_raw * S_posterior, axis=1) / 10.0))
        
        # Out-of-fold predictions container
        Q_t1_oof = np.zeros_like(Q_raw)
        Q_t2_oof = np.zeros_like(Q_raw)
        Q_t3_oof = np.zeros_like(Q_raw)
        Q_t4_oof = np.zeros_like(Q_raw)
        
        W_t1_focal = np.zeros_like(W_human)
        W_t2_focal = np.zeros_like(W_human)
        W_t3_focal = np.zeros_like(W_human)
        W_t4_focal = np.zeros_like(W_human)
        fold_assignment = np.zeros(len(P_human), dtype=int)
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            fold_assignment[test_idx] = fold
            P_tr, X_tr = P_human[train_idx], X_raw_ilr[train_idx]
            
            # --- Tier 1: Scalar Temp (1 DoF: x' = alpha * x) ---
            def loss_t1(params):
                alpha = max(1e-4, params[0])
                X_pred = X_tr * alpha
                Q_pred = ilr_to_probs(X_pred)
                bc = np.sum(np.sqrt(np.clip(P_tr * Q_pred, 0.0, 1.0)), axis=1)
                return float(np.mean(np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))))
            res_t1 = minimize(loss_t1, [1.0], method="Nelder-Mead")
            alpha_opt = max(1e-4, float(res_t1.x[0]))
            
            Q_t1_full = ilr_to_probs(X_raw_ilr * alpha_opt)
            Q_t1_oof[test_idx] = Q_t1_full[test_idx]
            W_t1_fold = compute_topk_weights(distance_hellinger_matrix(Q_t1_full), k=10)
            W_t1_focal[test_idx] = W_t1_fold[test_idx]
            
            # --- Tier 2: Diagonal ILR Scaling (2 DoF: x' = diag(a, b) * x) ---
            def loss_t2(params):
                a, b = max(1e-4, params[0]), max(1e-4, params[1])
                X_pred = X_tr * np.array([a, b])
                Q_pred = ilr_to_probs(X_pred)
                bc = np.sum(np.sqrt(np.clip(P_tr * Q_pred, 0.0, 1.0)), axis=1)
                return float(np.mean(np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))))
            res_t2 = minimize(loss_t2, [1.0, 1.0], method="Nelder-Mead")
            a_opt, b_opt = max(1e-4, float(res_t2.x[0])), max(1e-4, float(res_t2.x[1]))
            
            Q_t2_full = ilr_to_probs(X_raw_ilr * np.array([a_opt, b_opt]))
            Q_t2_oof[test_idx] = Q_t2_full[test_idx]
            W_t2_fold = compute_topk_weights(distance_hellinger_matrix(Q_t2_full), k=10)
            W_t2_focal[test_idx] = W_t2_fold[test_idx]
            
            # --- Tier 3: Affine ILR Map (6 DoF: x' = A x + b) ---
            def loss_t3(params):
                A = params[:4].reshape((2, 2))
                b_v = params[4:]
                X_pred = X_tr @ A.T + b_v
                Q_pred = ilr_to_probs(X_pred)
                bc = np.sum(np.sqrt(np.clip(P_tr * Q_pred, 0.0, 1.0)), axis=1)
                return float(np.mean(np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))))
            init_t3 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
            res_t3 = minimize(loss_t3, init_t3, method="L-BFGS-B")
            A_opt = res_t3.x[:4].reshape((2, 2))
            b_opt = res_t3.x[4:]
            
            Q_t3_full = ilr_to_probs(X_raw_ilr @ A_opt.T + b_opt)
            Q_t3_oof[test_idx] = Q_t3_full[test_idx]
            W_t3_fold = compute_topk_weights(distance_hellinger_matrix(Q_t3_full), k=10)
            W_t3_focal[test_idx] = W_t3_fold[test_idx]
            
            # --- Tier 4: Nonlinear ILR MLP (MLP on Dirichlet-smoothed targets) ---
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_full_scaled = scaler.transform(X_raw_ilr)
            
            Y_tr_ilr = X_human_ilr[train_idx]
            mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=20260803, early_stopping=True)
            mlp.fit(X_tr_scaled, Y_tr_ilr)
            
            X_t4_full = mlp.predict(X_full_scaled)
            Q_t4_full = ilr_to_probs(X_t4_full)
            Q_t4_oof[test_idx] = Q_t4_full[test_idx]
            W_t4_fold = compute_topk_weights(distance_hellinger_matrix(Q_t4_full), k=10)
            W_t4_focal[test_idx] = W_t4_fold[test_idx]

        # Evaluate OOF Held-Out Metrics
        def eval_oof(Q_oof, W_focal):
            d_err = np.sqrt(np.clip(1.0 - np.sum(np.sqrt(np.clip(P_human * Q_oof, 0.0, 1.0)), axis=1), 0.0, 1.0))
            M_clr = np.log(np.clip(Q_oof, 1e-8, 1.0)) - np.mean(np.log(np.clip(Q_oof, 1e-8, 1.0)), axis=1, keepdims=True)
            norm_M = np.linalg.norm(M_clr, axis=1)
            cos_th = np.clip(np.sum(H_clr * M_clr, axis=1) / (norm_H * norm_M + 1e-12), -1.0, 1.0)
            th_deg = float(np.mean(np.degrees(np.arccos(cos_th))))
            
            Q_supp = float(np.mean(np.sum(W_focal * S_posterior, axis=1) / 10.0))
            return float(np.mean(d_err)), th_deg, Q_supp, d_err

        d_t1, th_t1, Q_s_t1, d_err_t1 = eval_oof(Q_t1_oof, W_t1_focal)
        d_t2, th_t2, Q_s_t2, d_err_t2 = eval_oof(Q_t2_oof, W_t2_focal)
        d_t3, th_t3, Q_s_t3, d_err_t3 = eval_oof(Q_t3_oof, W_t3_focal)
        d_t4, th_t4, Q_s_t4, d_err_t4 = eval_oof(Q_t4_oof, W_t4_focal)
        
        results_by_model[mname] = {
            "tier_0_raw": {
                "d_hellinger_mean": d_raw_mean,
                "ambiguity_angle_deg_mean": theta_raw_deg,
                "Q_support_raw": Q_support_raw,
            },
            "tier_1_scalar_temp": {
                "dof_description": "1 DoF (Positive Ray in CLR/ILR space)",
                "heldout_d_hellinger_mean": d_t1,
                "heldout_ambiguity_angle_deg_mean": th_t1,
                "angle_reduction_deg": float(theta_raw_deg - th_t1),
                "fold_coherent_Q_support": Q_s_t1,
            },
            "tier_2_diagonal_ilr_scaling": {
                "dof_description": "2 DoF (Positive Cone in ILR space, diag(a, b) > 0)",
                "heldout_d_hellinger_mean": d_t2,
                "heldout_ambiguity_angle_deg_mean": th_t2,
                "angle_reduction_deg": float(theta_raw_deg - th_t2),
                "fold_coherent_Q_support": Q_s_t2,
            },
            "tier_3_affine_ilr_map": {
                "dof_description": "6 DoF (Affine Subspace in ILR space, A x + b)",
                "heldout_d_hellinger_mean": d_t3,
                "heldout_ambiguity_angle_deg_mean": th_t3,
                "angle_reduction_deg": float(theta_raw_deg - th_t3),
                "fold_coherent_Q_support": Q_s_t3,
            },
            "tier_4_nonlinear_ilr_mlp": {
                "dof_description": "Nonlinear ILR MLP (Simplex Surface)",
                "heldout_d_hellinger_mean": d_t4,
                "heldout_ambiguity_angle_deg_mean": th_t4,
                "angle_reduction_deg": float(theta_raw_deg - th_t4),
                "fold_coherent_Q_support": Q_s_t4,
            }
        }
        
        df_oof_m = pl.DataFrame({
            "object_id": item_ids,
            "fold_id": fold_assignment,
            "model_name": [mname] * len(item_ids),
            "d_raw": d_raw,
            "d_t1": d_err_t1,
            "d_t2": d_err_t2,
            "d_t3": d_err_t3,
            "d_t4": d_err_t4,
            "q_raw_e": Q_raw[:, 0], "q_raw_n": Q_raw[:, 1], "q_raw_c": Q_raw[:, 2],
            "q_t1_e": Q_t1_oof[:, 0], "q_t1_n": Q_t1_oof[:, 1], "q_t1_c": Q_t1_oof[:, 2],
            "q_t2_e": Q_t2_oof[:, 0], "q_t2_n": Q_t2_oof[:, 1], "q_t2_c": Q_t2_oof[:, 2],
            "q_t3_e": Q_t3_oof[:, 0], "q_t3_n": Q_t3_oof[:, 1], "q_t3_c": Q_t3_oof[:, 2],
            "q_t4_e": Q_t4_oof[:, 0], "q_t4_n": Q_t4_oof[:, 1], "q_t4_c": Q_t4_oof[:, 2],
        })
        oof_dfs.append(df_oof_m)
        
    summary = {
        "n_items": len(P_human),
        "cv_folds": 5,
        "models": results_by_model,
    }
    df_all_oof = pl.concat(oof_dfs)
    return summary, df_all_oof

if __name__ == "__main__":
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary, df_oof = run_reachable_set_ladder_ilr()
    
    out_file = out_dir / "reachable_set_ladder_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    oof_parquet = out_dir / "oof_predictions.parquet"
    df_oof.write_parquet(oof_parquet)
    
    print(f"Gauge-Invariant ILR E018 summary written to {out_file}")
    print(f"OOF predictions Parquet written to {oof_parquet}")
