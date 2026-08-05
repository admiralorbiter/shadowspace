"""Schemas and validation routines for Ambiguity Doppelgänger Atlas."""

from typing import Dict, Any, List
import polars as pl
import numpy as np
from .summaries import compute_shannon_entropy

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

VALID_MAJORITY_LABELS = {"entailment", "neutral", "contradiction"}
LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


def validate_canonical_df(df: pl.DataFrame) -> Dict[str, Any]:
    """Validate canonical_items schema, integer counts, recomputed entropy, and majority invariants."""
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

    # Check integer counts
    c_e = df["human_count_entailment"].to_numpy()
    c_n = df["human_count_neutral"].to_numpy()
    c_c = df["human_count_contradiction"].to_numpy()
    
    if np.any(c_e != np.round(c_e)) or np.any(c_n != np.round(c_n)) or np.any(c_c != np.round(c_c)):
        raise ValueError("Non-integer vote counts found in canonical items")
        
    if np.min(c_e) < 0 or np.min(c_n) < 0 or np.min(c_c) < 0:
        raise ValueError("Negative vote count found")
        
    total_counts = c_e + c_n + c_c
    if np.any(total_counts <= 0):
        raise ValueError("Item with zero total vote count found")

    # Check recomputed entropy
    p_mat = np.column_stack([p_e, p_n, p_c])
    recomputed_entropy = compute_shannon_entropy(p_mat)
    stored_entropy = df["human_entropy_bits"].to_numpy()
    max_ent_diff = float(np.max(np.abs(recomputed_entropy - stored_entropy)))
    if max_ent_diff > 1e-4:
        raise ValueError(f"Stored entropy does not match recomputed entropy; max diff: {max_ent_diff}")

    # Check majority label agreement & valid label set
    maj_labels = df["human_majority_label"].to_list()
    for idx in range(row_count):
        lbl = maj_labels[idx]
        if lbl not in VALID_MAJORITY_LABELS:
            raise ValueError(f"Unknown majority label '{lbl}' at row {idx}")
            
        counts = [c_e[idx], c_n[idx], c_c[idx]]
        max_c = max(counts)
        expected_idx = 0 if lbl == "entailment" else (1 if lbl == "neutral" else 2)
        actual_count = counts[expected_idx]
        if actual_count != max_c:
            raise ValueError(f"Stored majority label '{lbl}' does not match maximum count {max_c} at row {idx}")

    return {
        "status": "VALID",
        "row_count": row_count,
        "max_prob_sum_diff": max_sum_diff,
        "max_entropy_diff": max_ent_diff,
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

    # Check tier probabilities sum to 1, are finite, and within [0, 1]
    tier_prefixes = ["q_raw", "q_t1", "q_t2", "q_t3", "q_t4"]
    for prefix in tier_prefixes:
        pe = df[f"{prefix}_e"].to_numpy()
        pn = df[f"{prefix}_n"].to_numpy()
        pc = df[f"{prefix}_c"].to_numpy()
        
        if np.any(~np.isfinite(pe)) or np.any(~np.isfinite(pn)) or np.any(~np.isfinite(pc)):
            raise ValueError(f"Non-finite probability in tier {prefix}")
            
        if np.any(pe < 0.0) or np.any(pe > 1.0) or np.any(pn < 0.0) or np.any(pn > 1.0) or np.any(pc < 0.0) or np.any(pc > 1.0):
            raise ValueError(f"Probability out of [0, 1] bounds in tier {prefix}")
            
        diff = np.max(np.abs(pe + pn + pc - 1.0))
        if diff > 1e-4:
            raise ValueError(f"Probabilities in tier {prefix} do not sum to 1; max diff: {diff}")

    return {
        "status": "VALID",
        "row_count": row_count,
        "models": models,
        "unique_objects": len(oof_ids),
    }
