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


def validate_canonical_df(df: pl.DataFrame) -> Dict[str, Any]:
    """Validate canonical_items dataframe schema and invariants."""
    missing = [c for c in REQUIRED_CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in canonical_items: {missing}")

    # Check non-empty
    row_count = df.height
    if row_count == 0:
        raise ValueError("canonical_items.parquet is empty")

    # Check unique object_id
    unique_ids = df["object_id"].n_unique()
    if unique_ids != row_count:
        raise ValueError(f"object_id is not unique: {unique_ids} vs {row_count}")

    # Check probabilities sum to 1
    p_sum = df["human_p_entailment"] + df["human_p_neutral"] + df["human_p_contradiction"]
    max_sum_diff = float((p_sum - 1.0).abs().max())
    if max_sum_diff > 1e-4:
        raise ValueError(f"Probabilities do not sum to 1.0; max diff: {max_sum_diff}")

    # Check non-negative counts
    c_e = df["human_count_entailment"].min()
    c_n = df["human_count_neutral"].min()
    c_c = df["human_count_contradiction"].min()
    if min(c_e, c_n, c_c) < 0:
        raise ValueError("Negative vote count found")

    return {
        "status": "VALID",
        "row_count": row_count,
        "max_prob_sum_diff": max_sum_diff,
    }


def validate_oof_df(df: pl.DataFrame, canonical_ids: List[str]) -> Dict[str, Any]:
    """Validate oof_predictions dataframe schema and alignment."""
    missing = [c for c in REQUIRED_OOF_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in oof_predictions: {missing}")

    row_count = df.height
    models = df["model_name"].unique().to_list()
    
    # Check alignment with canonical IDs
    oof_ids = set(df["object_id"].unique().to_list())
    canon_set = set(canonical_ids)
    if not oof_ids.issubset(canon_set):
        raise ValueError("oof_predictions contains object_ids not present in canonical items")

    return {
        "status": "VALID",
        "row_count": row_count,
        "models": models,
        "unique_objects": len(oof_ids),
    }
