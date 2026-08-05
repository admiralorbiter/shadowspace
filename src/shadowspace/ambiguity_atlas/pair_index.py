"""Exact and approximate doppelgänger pair index algorithm in Polars and NumPy."""

import numpy as np
import polars as pl
from typing import Tuple, Dict, Any, List
from .geometry import (
    hellinger_distance,
    fisher_rao_distance,
    js_distance,
    aitchison_distance,
)
from .summaries import compute_minority_orientation, compute_shannon_entropy, LABEL_MAP

LABEL_COLS = {
    "entailment": "human_count_entailment",
    "neutral": "human_count_neutral",
    "contradiction": "human_count_contradiction",
}


def find_strict_doppelgaenger_pairs(df_canon: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Find all exact human doppelgänger pairs in canonical items dataset.
    
    A strict doppelgänger pair consists of two items A and B that share:
    - Same majority_label
    - Same majority_count
    - Same unordered minority counts (minority_low_count, minority_high_count)
    - Unequal minority counts (minority_low_count != minority_high_count)
    - Opposite assignment of minority_high_count (item A assigns high to class X, item B to class Y)
    """
    records = []
    for row in df_canon.to_dicts():
        maj_lbl = row["human_majority_label"]
        counts = {
            "entailment": row["human_count_entailment"],
            "neutral": row["human_count_neutral"],
            "contradiction": row["human_count_contradiction"],
        }
        maj_cnt = counts[maj_lbl]
        
        min_items = [(lbl, cnt) for lbl, cnt in counts.items() if lbl != maj_lbl]
        min_sorted = sorted(min_items, key=lambda x: (x[1], x[0]))
        
        min_low_lbl, min_low_cnt = min_sorted[0]
        min_high_lbl, min_high_cnt = min_sorted[1]
        
        p_vec = np.array([
            row["human_p_entailment"],
            row["human_p_neutral"],
            row["human_p_contradiction"]
        ], dtype=np.float64)
        
        maj_idx = 0 if maj_lbl == "entailment" else (1 if maj_lbl == "neutral" else 2)
        delta = compute_minority_orientation(p_vec, majority_idx=maj_idx)
        
        record = dict(row)
        record.update({
            "maj_count": maj_cnt,
            "min_low_count": min_low_cnt,
            "min_high_count": min_high_cnt,
            "min_high_label": min_high_lbl,
            "min_low_label": min_low_lbl,
            "minority_orientation": delta,
        })
        records.append(record)

    enriched_df = pl.DataFrame(records)
    asymmetric_df = enriched_df.filter(pl.col("min_low_count") < pl.col("min_high_count"))
    
    group_cols = ["human_majority_label", "maj_count", "min_low_count", "min_high_count"]
    groups = asymmetric_df.group_by(group_cols)
    
    pair_records = []
    group_count = 0
    
    for _, group in groups:
        items = group.to_dicts()
        if len(items) < 2:
            continue
            
        labels_present = set(it["min_high_label"] for it in items)
        if len(labels_present) < 2:
            continue
            
        group_count += 1
        
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item_a, item_b = items[i], items[j]
                if item_a["min_high_label"] == item_b["min_high_label"]:
                    continue
                    
                if item_a["object_id"] > item_b["object_id"]:
                    item_a, item_b = item_b, item_a
                    
                p_a = np.array([item_a["human_p_entailment"], item_a["human_p_neutral"], item_a["human_p_contradiction"]])
                p_b = np.array([item_b["human_p_entailment"], item_b["human_p_neutral"], item_b["human_p_contradiction"]])
                
                dh = float(hellinger_distance(p_a, p_b))
                dfr = float(fisher_rao_distance(p_a, p_b))
                djs = float(js_distance(p_a, p_b))
                da = float(aitchison_distance(p_a, p_b, alpha=0.5))
                
                pair_id = f"strict_{item_a['object_id']}_{item_b['object_id']}"
                
                pair_records.append({
                    "pair_id": pair_id,
                    "object_id_a": item_a["object_id"],
                    "object_id_b": item_b["object_id"],
                    "majority_label": item_a["human_majority_label"],
                    "majority_count": item_a["maj_count"],
                    "majority_probability": item_a["human_p_" + item_a["human_majority_label"]],
                    "entropy_bits": item_a["human_entropy_bits"],
                    "minority_low_count": item_a["min_low_count"],
                    "minority_high_count": item_a["min_high_count"],
                    "minority_label_high_a": item_a["min_high_label"],
                    "minority_label_high_b": item_b["min_high_label"],
                    "minority_orientation_a": item_a["minority_orientation"],
                    "minority_orientation_b": item_b["minority_orientation"],
                    "d_hellinger": dh,
                    "d_fisher_rao": dfr,
                    "d_js": djs,
                    "d_aitchison": da,
                    "premise_a": item_a["premise"],
                    "hypothesis_a": item_a["hypothesis"],
                    "premise_b": item_b["premise"],
                    "hypothesis_b": item_b["hypothesis"],
                    "source_dataset_a": item_a["source_dataset"],
                    "source_dataset_b": item_b["source_dataset"],
                })

    pairs_df = pl.DataFrame(pair_records) if pair_records else pl.DataFrame()
    
    summary = {
        "exact_groups_count": group_count,
        "exact_pairs_count": len(pair_records),
        "participating_items_count": len(set([p["object_id_a"] for p in pair_records] + [p["object_id_b"] for p in pair_records])) if pair_records else 0,
    }
    
    return pairs_df, summary


def find_approximate_doppelgaenger_pairs(
    df_canon: pl.DataFrame,
    max_conf_diff: float = 0.02,
    max_entropy_diff: float = 0.05,
) -> pl.DataFrame:
    """Find approximate human doppelgänger pairs within confidence & entropy tolerances.
    
    Rule:
    - Same human_majority_label
    - Opposite minority orientation sign: delta_a * delta_b < 0
    - |conf_a - conf_b| <= max_conf_diff
    - |entropy_a - entropy_b| <= max_entropy_diff
    """
    items = df_canon.to_dicts()
    n_items = len(items)
    
    # Pre-extract arrays
    p_vecs = np.zeros((n_items, 3), dtype=np.float64)
    maj_indices = np.zeros(n_items, dtype=np.int32)
    confidences = np.zeros(n_items, dtype=np.float64)
    entropies = np.zeros(n_items, dtype=np.float64)
    deltas = np.zeros(n_items, dtype=np.float64)
    
    for idx, item in enumerate(items):
        p = np.array([item["human_p_entailment"], item["human_p_neutral"], item["human_p_contradiction"]], dtype=np.float64)
        p_vecs[idx] = p
        m_idx = 0 if item["human_majority_label"] == "entailment" else (1 if item["human_majority_label"] == "neutral" else 2)
        maj_indices[idx] = m_idx
        confidences[idx] = p[m_idx]
        entropies[idx] = item["human_entropy_bits"]
        deltas[idx] = compute_minority_orientation(p, majority_idx=m_idx)
        
    pair_records = []
    
    for i in range(n_items):
        for j in range(i + 1, n_items):
            # 1. Same majority label
            if maj_indices[i] != maj_indices[j]:
                continue
                
            # 2. Opposite minority orientation
            d_i, d_j = deltas[i], deltas[j]
            if d_i * d_j >= 0 or abs(d_i) < 1e-4 or abs(d_j) < 1e-4:
                continue
                
            # 3. Summary differences
            conf_diff = abs(confidences[i] - confidences[j])
            if conf_diff > max_conf_diff:
                continue
                
            ent_diff = abs(entropies[i] - entropies[j])
            if ent_diff > max_entropy_diff:
                continue
                
            item_a, item_b = items[i], items[j]
            p_a, p_b = p_vecs[i], p_vecs[j]
            
            # Canonicalize pair order by object_id
            if item_a["object_id"] > item_b["object_id"]:
                item_a, item_b = item_b, item_a
                p_a, p_b = p_b, p_a
                d_i, d_j = d_j, d_i
                
            dh = float(hellinger_distance(p_a, p_b))
            dfr = float(fisher_rao_distance(p_a, p_b))
            djs = float(js_distance(p_a, p_b))
            da = float(aitchison_distance(p_a, p_b, alpha=0.5))
            
            # Summary discrepancy distance
            summary_dist = np.sqrt(conf_diff**2 + ent_diff**2)
            
            pair_id = f"approx_{item_a['object_id']}_{item_b['object_id']}"
            
            pair_records.append({
                "pair_id": pair_id,
                "object_id_a": item_a["object_id"],
                "object_id_b": item_b["object_id"],
                "majority_label": item_a["human_majority_label"],
                "confidence_a": float(p_a[maj_indices[i]]),
                "confidence_b": float(p_b[maj_indices[j]]),
                "confidence_diff": float(conf_diff),
                "entropy_a": float(entropies[i]),
                "entropy_b": float(entropies[j]),
                "entropy_diff": float(ent_diff),
                "summary_dist": float(summary_dist),
                "minority_orientation_a": float(d_i),
                "minority_orientation_b": float(d_j),
                "d_hellinger": dh,
                "d_fisher_rao": dfr,
                "d_js": djs,
                "d_aitchison": da,
                "premise_a": item_a["premise"],
                "hypothesis_a": item_a["hypothesis"],
                "premise_b": item_b["premise"],
                "hypothesis_b": item_b["hypothesis"],
                "source_dataset_a": item_a["source_dataset"],
                "source_dataset_b": item_b["source_dataset"],
            })

    return pl.DataFrame(pair_records) if pair_records else pl.DataFrame()
