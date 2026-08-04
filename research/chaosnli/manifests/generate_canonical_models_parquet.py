"""Generate data/chaosnli/processed/canonical_models.parquet from research/chaosnli/rust_manifest/model_probs.json."""

import json
from pathlib import Path
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def generate_canonical_models():
    probs_path = PROJECT_ROOT / "research" / "chaosnli" / "rust_manifest" / "model_probs.json"
    items_path = PROJECT_ROOT / "data" / "chaosnli" / "processed" / "canonical_items.parquet"
    out_path = PROJECT_ROOT / "data" / "chaosnli" / "processed" / "canonical_models.parquet"
    
    with open(probs_path, "r", encoding="utf-8") as f:
        model_probs = json.load(f)
        
    canon_df = pl.read_parquet(items_path)
    object_ids = canon_df["object_id"].to_list()
    p_human = canon_df.select(
        ["human_p_entailment", "human_p_neutral", "human_p_contradiction"]
    ).to_numpy()
    
    records = []
    
    for i, obj_id in enumerate(object_ids):
        p_i = p_human[i]
        
        for model_name, probs_list in model_probs.items():
            q_i = np.array(probs_list[i], dtype=np.float64)
            q_i = np.clip(q_i, 1e-12, 1.0)
            q_i = q_i / np.sum(q_i)
            
            # Log probabilities as logits (additive constant shift drops out in softmax / CLR)
            logits_i = np.log(q_i)
            
            # Pointwise Hellinger distance
            d_h = float(np.sqrt(0.5 * np.sum((np.sqrt(p_i) - np.sqrt(q_i)) ** 2)))
            
            # Pointwise Jensen-Shannon Divergence (bits)
            m_mix = 0.5 * (p_i + q_i)
            kl_p = np.sum(np.where(p_i > 0, p_i * np.log(np.maximum(p_i, 1e-12) / m_mix), 0.0))
            kl_q = np.sum(np.where(q_i > 0, q_i * np.log(np.maximum(q_i, 1e-12) / m_mix), 0.0))
            jsd = float(0.5 * (kl_p + kl_q) / np.log(2.0))
            
            # Brier Score
            brier = float(np.sum((q_i - p_i) ** 2))
            
            records.append({
                "object_id": obj_id,
                "model_name": model_name,
                "logit_entailment": float(logits_i[0]),
                "logit_neutral": float(logits_i[1]),
                "logit_contradiction": float(logits_i[2]),
                "model_p_entailment": float(q_i[0]),
                "model_p_neutral": float(q_i[1]),
                "model_p_contradiction": float(q_i[2]),
                "pointwise_hellinger": d_h,
                "pointwise_jsd_bits": jsd,
                "pointwise_brier": brier,
            })
            
    df_out = pl.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.write_parquet(out_path)
    print(f"Successfully generated {out_path} with {len(df_out)} records across {len(model_probs)} models.")

if __name__ == "__main__":
    generate_canonical_models()
