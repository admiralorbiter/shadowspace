"""Module 3: Semantic Torn Twins vs Distribution Twins.

Distinguishes:
1. Distribution Twins: H(p_i, p_j) < 0.05 (vote distribution match only)
2. Semantic Torn Twins: H(p_i, p_j) < 0.05 AND text similarity > threshold AND H(q_i, q_j) > 0.35
"""

import os
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, str(Path(__file__).parent))

from metric_atlas import distance_hellinger_matrix

def run_semantic_torn_twins() -> dict:
    items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    models_path = Path("data/chaosnli/processed/canonical_models.parquet")
    
    df_items = pl.read_parquet(items_path)
    df_models = pl.read_parquet(models_path)
    
    print("Computing text embeddings cosine similarity matrix...")
    texts = [f"{r['premise']} {r['hypothesis']}" for r in df_items.to_dicts()]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_tfidf = vectorizer.fit_transform(texts)
    S_text = cosine_similarity(X_tfidf)
    
    P_human = df_items.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    D_human = distance_hellinger_matrix(P_human)
    
    iu = np.triu_indices(len(P_human), k=1)
    d_h_human = D_human[iu]
    s_text_pairs = S_text[iu]
    
    # 1. Distribution Twins: H(p_i, p_j) < 0.05
    dist_twin_mask = d_h_human < 0.05
    total_dist_twins = int(np.sum(dist_twin_mask))
    
    # 2. Semantic Twins: H(p_i, p_j) < 0.05 AND text similarity > 0.40
    semantic_twin_mask = dist_twin_mask & (s_text_pairs >= 0.40)
    total_semantic_twins = int(np.sum(semantic_twin_mask))
    
    # Analyze per model
    model_names = sorted(df_models["model_name"].unique().to_list())
    results_by_model = {}
    
    items_list = df_items.to_dicts()
    
    for mname in model_names:
        sub_m = df_models.filter(pl.col("model_name") == mname)
        joined = df_items.join(sub_m, on="object_id", how="inner")
        
        Q_model = joined.select(["model_p_entailment", "model_p_neutral", "model_p_contradiction"]).to_numpy()
        D_model = distance_hellinger_matrix(Q_model)
        d_h_model = D_model[iu]
        
        # Distribution Twins torn by model (d_h_model > 0.35)
        dist_torn_mask = dist_twin_mask & (d_h_model > 0.35)
        dist_torn_count = int(np.sum(dist_torn_mask))
        
        # Semantic Twins torn by model
        semantic_torn_mask = semantic_twin_mask & (d_h_model > 0.35)
        semantic_torn_count = int(np.sum(semantic_torn_mask))
        
        examples = []
        if semantic_torn_count > 0:
            s_idx = np.where(semantic_torn_mask)[0]
            for p_idx in s_idx[:5]:
                i = iu[0][p_idx]
                j = iu[1][p_idx]
                examples.append({
                    "item_i": {"id": items_list[i]["object_id"], "prem": items_list[i]["premise"], "hyp": items_list[i]["hypothesis"]},
                    "item_j": {"id": items_list[j]["object_id"], "prem": items_list[j]["premise"], "hyp": items_list[j]["hypothesis"]},
                    "human_distance": float(d_h_human[p_idx]),
                    "model_distance": float(d_h_model[p_idx]),
                    "text_similarity": float(s_text_pairs[p_idx]),
                })
                
        results_by_model[mname] = {
            "distribution_twins_torn_count": dist_torn_count,
            "distribution_twins_torn_pct": float((dist_torn_count / total_dist_twins) * 100.0) if total_dist_twins > 0 else 0.0,
            "semantic_twins_torn_count": semantic_torn_count,
            "semantic_twins_torn_pct": float((semantic_torn_count / total_semantic_twins) * 100.0) if total_semantic_twins > 0 else 0.0,
            "examples": examples,
        }
        
    summary = {
        "n_items": len(P_human),
        "total_distribution_twins_pairs": total_dist_twins,
        "total_semantic_twins_pairs": total_semantic_twins,
        "models": results_by_model,
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("results/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = run_semantic_torn_twins()
    out_file = out_dir / "semantic_torn_twins_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Semantic Torn Twins summary written to {out_file}")
