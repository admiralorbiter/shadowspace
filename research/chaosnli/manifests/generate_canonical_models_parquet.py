"""Generate data/chaosnli/processed/canonical_models.parquet with strict alignment & source object-order assertions."""

import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def generate_canonical_models_hardened():
    probs_path = PROJECT_ROOT / "research" / "chaosnli" / "rust_manifest" / "model_probs.json"
    probs_manifest_path = probs_path.with_suffix(".manifest.json")
    items_path = PROJECT_ROOT / "data" / "chaosnli" / "processed" / "canonical_items.parquet"
    out_path = PROJECT_ROOT / "data" / "chaosnli" / "processed" / "canonical_models.parquet"
    
    canon_df = pl.read_parquet(items_path).sort("object_id")
    object_ids = canon_df["object_id"].to_list()
    
    # Compute canonical item object-order SHA-256
    canon_obj_ids_json = json.dumps(object_ids).encode("utf-8")
    canon_obj_ids_sha256 = hashlib.sha256(canon_obj_ids_json).hexdigest()
    
    # Assert source sidecar manifest matches canonical item object-order SHA-256
    assert probs_manifest_path.exists(), f"Source manifest missing: {probs_manifest_path}"
    with open(probs_manifest_path, "r", encoding="utf-8") as f:
        src_manifest = json.load(f)
        
    src_obj_ids_sha256 = src_manifest["ordered_object_ids_sha256"]
    assert src_obj_ids_sha256 == canon_obj_ids_sha256, (
        f"Source object-order SHA256 mismatch! Source: {src_obj_ids_sha256}, Canonical: {canon_obj_ids_sha256}"
    )
    
    with open(probs_path, "rb") as f:
        probs_content = f.read()
        sha256_probs = hashlib.sha256(probs_content).hexdigest()
        
    model_probs = json.loads(probs_content.decode("utf-8"))
    p_human = canon_df.select(
        ["human_p_entailment", "human_p_neutral", "human_p_contradiction"]
    ).to_numpy()
    
    assert len(object_ids) == 3113, f"Expected 3113 canonical items, got {len(object_ids)}"
    for mname, probs_list in model_probs.items():
        assert len(probs_list) == 3113, f"Model {mname} length mismatch: {len(probs_list)} != 3113"
        probs_arr = np.array(probs_list, dtype=np.float64)
        assert probs_arr.shape == (3113, 3), f"Model {mname} shape mismatch: {probs_arr.shape}"
        row_sums = probs_arr.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), f"Model {mname} probability mass sum != 1.0"
        
    records = []
    for i, obj_id in enumerate(object_ids):
        p_i = p_human[i]
        
        for model_name, probs_list in model_probs.items():
            q_i = np.array(probs_list[i], dtype=np.float64)
            q_i = np.clip(q_i, 1e-12, 1.0)
            q_i = q_i / np.sum(q_i)
            
            logits_i = np.log(q_i)
            
            d_h = float(np.sqrt(0.5 * np.sum((np.sqrt(p_i) - np.sqrt(q_i)) ** 2)))
            
            m_mix = 0.5 * (p_i + q_i)
            kl_p = np.sum(np.where(p_i > 0, p_i * np.log(np.maximum(p_i, 1e-12) / m_mix), 0.0))
            kl_q = np.sum(np.where(q_i > 0, q_i * np.log(np.maximum(q_i, 1e-12) / m_mix), 0.0))
            jsd = float(0.5 * (kl_p + kl_q) / np.log(2.0))
            
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
    
    out_bytes = out_path.read_bytes()
    sha256_out_parquet = hashlib.sha256(out_bytes).hexdigest()
    
    manifest_info = {
        "output_path": str(out_path),
        "output_parquet_sha256": sha256_out_parquet,
        "canonical_object_ids_sha256": canon_obj_ids_sha256,
        "source_probs_sha256": sha256_probs,
        "label_order": ["entailment", "neutral", "contradiction"],
        "n_items": len(object_ids),
        "n_models": len(model_probs),
        "models": list(model_probs.keys())
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_info, f, indent=2)
        
    print(f"Successfully verified object-order alignment & exported canonical models table to {out_path}")
    print(f"Sidecar manifest written to {manifest_path} (Parquet SHA256: {sha256_out_parquet[:12]}...)")

if __name__ == "__main__":
    generate_canonical_models_hardened()
