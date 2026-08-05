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


def source_family(value: str) -> str:
    """Normalize source dataset label to canonical family ('snli' or 'mnli')."""
    if not value:
        raise ValueError("Empty source dataset value")
    normalized = str(value).strip().lower()
    if normalized in {"snli", "chaosnli_snli", "chaosnli-snli"}:
        return "snli"
    if normalized in {"mnli", "chaosnli_mnli", "chaosnli-mnli"}:
        return "mnli"
    raise ValueError(f"Unknown source dataset: {value!r}")


def compute_source_splits(ds_a: List[str], ds_b: List[str]) -> Dict[str, int]:
    """Compute mutually exclusive and exhaustive source dataset split counts."""
    split_counts = {
        "within_snli": 0,
        "within_mnli": 0,
        "cross_source": 0,
    }
    for source_a, source_b in zip(ds_a, ds_b):
        family_a = source_family(source_a)
        family_b = source_family(source_b)
        if family_a == "snli" and family_b == "snli":
            split_counts["within_snli"] += 1
        elif family_a == "mnli" and family_b == "mnli":
            split_counts["within_mnli"] += 1
        else:
            split_counts["cross_source"] += 1
            
    assert sum(split_counts.values()) == len(ds_a), "Source split counts must sum exactly to total pair count"
    return split_counts


def find_strict_doppelgaenger_pairs(df_canon: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """Find all exact human doppelgänger pairs in canonical items dataset.
    
    A strict doppelgänger pair consists of two items A and B that share:
    - Same majority_label
    - Same majority_count
    - Same unordered minority counts (minority_low_count, minority_high_count)
    - Unequal minority counts (minority_low_count != minority_high_count)
    - Opposite assignment of minority_high_count
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
    
    Fixes metadata alignment issue by bundling endpoint objects before object_id swap.
    Computes true Pareto non-domination frontier flag.
    """
    items = df_canon.to_dicts()
    n_items = len(items)
    
    endpoints = []
    for item in items:
        p = np.array([item["human_p_entailment"], item["human_p_neutral"], item["human_p_contradiction"]], dtype=np.float64)
        m_idx = 0 if item["human_majority_label"] == "entailment" else (1 if item["human_majority_label"] == "neutral" else 2)
        delta = compute_minority_orientation(p, majority_idx=m_idx)
        
        endpoints.append({
            "item": item,
            "p": p,
            "maj_idx": m_idx,
            "confidence": float(p[m_idx]),
            "entropy": float(item["human_entropy_bits"]),
            "delta": float(delta),
        })
        
    pair_records = []
    
    for i in range(n_items):
        for j in range(i + 1, n_items):
            ep_i, ep_j = endpoints[i], endpoints[j]
            
            # 1. Same majority label
            if ep_i["maj_idx"] != ep_j["maj_idx"]:
                continue
                
            # 2. Opposite minority orientation
            d_i, d_j = ep_i["delta"], ep_j["delta"]
            if d_i * d_j >= 0 or abs(d_i) < 1e-4 or abs(d_j) < 1e-4:
                continue
                
            # 3. Summary differences
            conf_diff = abs(ep_i["confidence"] - ep_j["confidence"])
            if conf_diff > max_conf_diff:
                continue
                
            ent_diff = abs(ep_i["entropy"] - ep_j["entropy"])
            if ent_diff > max_entropy_diff:
                continue
                
            # Bundle endpoints and swap together to keep entropy_a/b, delta_a/b aligned
            ep_a, ep_b = ep_i, ep_j
            if ep_a["item"]["object_id"] > ep_b["item"]["object_id"]:
                ep_a, ep_b = ep_b, ep_a
                
            p_a, p_b = ep_a["p"], ep_b["p"]
            dh = float(hellinger_distance(p_a, p_b))
            dfr = float(fisher_rao_distance(p_a, p_b))
            djs = float(js_distance(p_a, p_b))
            da = float(aitchison_distance(p_a, p_b, alpha=0.5))
            
            summary_dist = np.sqrt(conf_diff**2 + ent_diff**2)
            pair_id = f"approx_{ep_a['item']['object_id']}_{ep_b['item']['object_id']}"
            
            pair_records.append({
                "pair_id": pair_id,
                "object_id_a": ep_a["item"]["object_id"],
                "object_id_b": ep_b["item"]["object_id"],
                "majority_label": ep_a["item"]["human_majority_label"],
                "confidence_a": ep_a["confidence"],
                "confidence_b": ep_b["confidence"],
                "confidence_diff": float(conf_diff),
                "entropy_a": ep_a["entropy"],
                "entropy_b": ep_b["entropy"],
                "entropy_diff": float(ent_diff),
                "summary_dist": float(summary_dist),
                "minority_orientation_a": ep_a["delta"],
                "minority_orientation_b": ep_b["delta"],
                "d_hellinger": dh,
                "d_fisher_rao": dfr,
                "d_js": djs,
                "d_aitchison": da,
                "premise_a": ep_a["item"]["premise"],
                "hypothesis_a": ep_a["item"]["hypothesis"],
                "premise_b": ep_b["item"]["premise"],
                "hypothesis_b": ep_b["item"]["hypothesis"],
                "source_dataset_a": ep_a["item"]["source_dataset"],
                "source_dataset_b": ep_b["item"]["source_dataset"],
            })

    if not pair_records:
        return pl.DataFrame()
        
    df_pairs = pl.DataFrame(pair_records)
    
    # Compute Pareto non-domination: x dominates y if summary_dist(x) <= summary_dist(y) AND d_hellinger(x) >= d_hellinger(y)
    summary_dists = df_pairs["summary_dist"].to_numpy()
    dh_dists = df_pairs["d_hellinger"].to_numpy()
    n_p = len(df_pairs)
    
    is_pareto = np.ones(n_p, dtype=bool)
    for u in range(n_p):
        for v in range(n_p):
            if u == v:
                continue
            # v dominates u if v has <= summary_dist and >= d_hellinger, with at least one strict inequality
            if (summary_dists[v] <= summary_dists[u]) and (dh_dists[v] >= dh_dists[u]):
                if (summary_dists[v] < summary_dists[u]) or (dh_dists[v] > dh_dists[u]):
                    is_pareto[u] = False
                    break
                    
    return df_pairs.with_columns(pl.Series("is_pareto_optimal", is_pareto))
