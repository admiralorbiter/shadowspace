"""Frozen model prediction doppelgänger retention audit."""

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from typing import Dict, Any, List, Tuple
from .geometry import hellinger_distance
from .summaries import compute_minority_orientation

TIER_COLS = {
    "raw": ("q_raw_e", "q_raw_n", "q_raw_c"),
    "t1": ("q_t1_e", "q_t1_n", "q_t1_c"),
    "t2": ("q_t2_e", "q_t2_n", "q_t2_c"),
    "t3": ("q_t3_e", "q_t3_n", "q_t3_c"),
    "t4": ("q_t4_e", "q_t4_n", "q_t4_c"),
}


def compute_pair_model_retention(
    pair: Dict[str, Any],
    model_preds_a: Dict[str, Any],
    model_preds_b: Dict[str, Any],
    tier: str = "raw"
) -> Dict[str, Any]:
    """Compute model retention metrics for a single item pair and calibration tier."""
    e_col, n_col, c_col = TIER_COLS[tier]
    
    q_a = np.array([model_preds_a[e_col], model_preds_a[n_col], model_preds_a[c_col]], dtype=np.float64)
    q_b = np.array([model_preds_b[e_col], model_preds_b[n_col], model_preds_b[c_col]], dtype=np.float64)
    
    maj_lbl = pair["majority_label"]
    maj_idx = 0 if maj_lbl == "entailment" else (1 if maj_lbl == "neutral" else 2)
    
    delta_h_a = pair["minority_orientation_a"]
    delta_h_b = pair["minority_orientation_b"]
    human_contrast = delta_h_a - delta_h_b
    
    delta_m_a = compute_minority_orientation(q_a, majority_idx=maj_idx)
    delta_m_b = compute_minority_orientation(q_b, majority_idx=maj_idx)
    model_contrast = delta_m_a - delta_m_b
    
    if abs(human_contrast) > 1e-6:
        retention_ratio = float(model_contrast / human_contrast)
    else:
        retention_ratio = 0.0
        
    dh_human = pair["d_hellinger"]
    dh_model = float(hellinger_distance(q_a, q_b))
    dist_ratio = float(dh_model / dh_human) if dh_human > 1e-6 else 0.0
    
    sign_accurate = bool((human_contrast * model_contrast) > 0)
    
    if retention_ratio < -0.10:
        retention_category = "INVERTED"
    elif abs(retention_ratio) <= 0.10:
        retention_category = "COLLAPSED"
    elif 0.10 < retention_ratio < 0.50:
        retention_category = "ATTENUATED"
    elif 0.50 <= retention_ratio <= 1.50:
        retention_category = "PRESERVED"
    else:
        retention_category = "AMPLIFIED"
        
    return {
        "tier": tier,
        "human_contrast": human_contrast,
        "model_contrast": model_contrast,
        "retention_ratio": retention_ratio,
        "dh_human": dh_human,
        "dh_model": dh_model,
        "distance_retention_ratio": dist_ratio,
        "sign_accurate": sign_accurate,
        "retention_category": retention_category,
    }


def evaluate_model_retention(
    df_pairs: pl.DataFrame,
    df_oof: pl.DataFrame
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Evaluate doppelgänger contrast retention across all models and calibration tiers."""
    models = df_oof["model_name"].unique().to_list()
    oof_dicts = {}
    for row in df_oof.to_dicts():
        oof_dicts[(row["object_id"], row["model_name"])] = row
        
    records = []
    
    for pair in df_pairs.to_dicts():
        obj_a = pair["object_id_a"]
        obj_b = pair["object_id_b"]
        
        for model in models:
            preds_a = oof_dicts.get((obj_a, model))
            preds_b = oof_dicts.get((obj_b, model))
            
            if not preds_a or not preds_b:
                continue
                
            for tier in TIER_COLS.keys():
                metrics = compute_pair_model_retention(pair, preds_a, preds_b, tier=tier)
                
                record = {
                    "pair_id": pair["pair_id"],
                    "object_id_a": obj_a,
                    "object_id_b": obj_b,
                    "model_name": model,
                    "tier": tier,
                    "majority_label": pair["majority_label"],
                    "d_hellinger_human": pair["d_hellinger"],
                }
                record.update(metrics)
                records.append(record)

    df_ret = pl.DataFrame(records)
    
    # Compute aggregate model/tier summaries
    summary_records = []
    for (model, tier), group in df_ret.group_by(["model_name", "tier"]):
        rets = group["retention_ratio"].to_numpy()
        cats = group["retention_category"].value_counts().to_dicts()
        
        cat_counts = {c["retention_category"]: c["count"] for c in cats}
        total = group.height
        
        dh_h = group["d_hellinger_human"].to_numpy()
        dh_m = group["dh_model"].to_numpy()
        
        rho, _ = spearmanr(dh_h, dh_m) if len(dh_h) > 5 else (0.0, 1.0)
        
        summary_records.append({
            "model_name": model,
            "tier": tier,
            "total_pairs": total,
            "mean_retention_ratio": float(np.mean(rets)),
            "median_retention_ratio": float(np.median(rets)),
            "collapse_rate": cat_counts.get("COLLAPSED", 0) / total,
            "inversion_rate": cat_counts.get("INVERTED", 0) / total,
            "attenuation_rate": cat_counts.get("ATTENUATED", 0) / total,
            "preservation_rate": cat_counts.get("PRESERVED", 0) / total,
            "amplification_rate": cat_counts.get("AMPLIFIED", 0) / total,
            "sign_accuracy": float(group["sign_accurate"].mean()),
            "distance_spearman_rho": float(rho),
        })

    summary_df = pl.DataFrame(summary_records)
    return df_ret, summary_df
