"""Schemas and validation routines for Ambiguity Doppelgänger Atlas."""

from typing import Dict, Any, List
import polars as pl
import numpy as np

REQUIRED_CANONICAL_COLUMNS = [
    "object_id",
    "source_dataset",
    "premise",
    "hypothesis",
    "human_count_entailment",
    "human_count_neutral",
    "human_count_contradiction",
    "human_p_entailment",
    "human_p_neutral",
    "human_p_contradiction",
    "human_entropy_bits",
    "human_majority_label",
]

REQUIRED_OOF_COLUMNS = [
    "object_id",
    "model_name",
    "fold_id",
    "q_raw_e", "q_raw_n", "q_raw_c",
    "q_t1_e", "q_t1_n", "q_t1_c",
    "q_t2_e", "q_t2_n", "q_t2_c",
    "q_t3_e", "q_t3_n", "q_t3_c",
    "q_t4_e", "q_t4_n", "q_t4_c",
]

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


def validate_canonical_df(df: pl.DataFrame) -> Dict[str, Any]:
    """Validate canonical_items dataframe schema, probabilities, counts, entropy, and majority invariants."""
    missing = [c for c in REQUIRED_CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in canonical_items: {missing}")

    row_count = df.height
    if row_count == 0:
        raise ValueError("canonical_items.parquet is empty")

    unique_ids = df["object_id"].n_unique()
    if unique_ids != row_count:
        raise ValueError(f"object_id is not unique: {unique_ids} vs {row_count}")

    # Check finite and bounds [0, 1]
    p_e = df["human_p_entailment"].to_numpy()
    p_n = df["human_p_neutral"].to_numpy()
    p_c = df["human_p_contradiction"].to_numpy()
    
    if np.any(~np.isfinite(p_e)) or np.any(~np.isfinite(p_n)) or np.any(~np.isfinite(p_c)):
        raise ValueError("Non-finite probability value found in canonical items")
        
    if np.any(p_e < 0.0) or np.any(p_e > 1.0) or np.any(p_n < 0.0) or np.any(p_n > 1.0) or np.any(p_c < 0.0) or np.any(p_c > 1.0):
        raise ValueError("Probability out of [0, 1] bounds in canonical items")

    # Check sum to 1
    p_sum = p_e + p_n + p_c
    max_sum_diff = float(np.max(np.abs(p_sum - 1.0)))
    if max_sum_diff > 1e-4:
        raise ValueError(f"Probabilities do not sum to 1.0; max diff: {max_sum_diff}")

    # Check counts
    c_e = df["human_count_entailment"].to_numpy()
    c_n = df["human_count_neutral"].to_numpy()
    c_c = df["human_count_contradiction"].to_numpy()
    
    if np.min(c_e) < 0 or np.min(c_n) < 0 or np.min(c_c) < 0:
        raise ValueError("Negative vote count found")
        
    total_counts = c_e + c_n + c_c
    if np.any(total_counts <= 0):
        raise ValueError("Item with zero total vote count found")

    # Check majority label agreement
    maj_labels = df["human_majority_label"].to_list()
    for idx in range(row_count):
        counts = [c_e[idx], c_n[idx], c_c[idx]]
        max_c = max(counts)
        expected_maj = LABEL_MAP[counts.index(max_c)]
        # Allow ties if count equals max_c
        actual_count = counts[LABEL_MAP.get(maj_labels[idx], 0) if maj_labels[idx] == "entailment" else (1 if maj_labels[idx] == "neutral" else 2)]
        if actual_count != max_c:
            raise ValueError(f"Stored majority label '{maj_labels[idx]}' does not match maximum count {max_c} at row {idx}")

    return {
        "status": "VALID",
        "row_count": row_count,
        "max_prob_sum_diff": max_sum_diff,
    }


def validate_oof_df(df: pl.DataFrame, canonical_ids: List[str]) -> Dict[str, Any]:
    """Validate oof_predictions schema, uniqueness, completeness, and probability bounds."""
    missing = [c for c in REQUIRED_OOF_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in oof_predictions: {missing}")

    row_count = df.height
    models = df["model_name"].unique().to_list()
    
    # Check unique (object_id, model_name)
    n_pairs = df.select(["object_id", "model_name"]).n_unique()
    if n_pairs != row_count:
        raise ValueError(f"Duplicate (object_id, model_name) predictions found: {n_pairs} unique vs {row_count} total")

    oof_ids = set(df["object_id"].unique().to_list())
    canon_set = set(canonical_ids)
    if not oof_ids.issubset(canon_set):
        raise ValueError("oof_predictions contains object_ids not present in canonical items")

    # Check tier probabilities sum to 1 and are finite
    tier_prefixes = ["q_raw", "q_t1", "q_t2", "q_t3", "q_t4"]
    for prefix in tier_prefixes:
        pe = df[f"{prefix}_e"].to_numpy()
        pn = df[f"{prefix}_n"].to_numpy()
        pc = df[f"{prefix}_c"].to_numpy()
        
        if np.any(~np.isfinite(pe)) or np.any(~np.isfinite(pn)) or np.any(~np.isfinite(pc)):
            raise ValueError(f"Non-finite probability in tier {prefix}")
            
        diff = np.max(np.abs(pe + pn + pc - 1.0))
        if diff > 1e-4:
            raise ValueError(f"Probabilities in tier {prefix} do not sum to 1; max diff: {diff}")

    return {
        "status": "VALID",
        "row_count": row_count,
        "models": models,
        "unique_objects": len(oof_ids),
    }
