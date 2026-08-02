"""Level 1 Opinion Profile Graph and Level 2 Item Profile Analysis module."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.neighbors import extract_knn_graph


def build_level1_profile_graph(
    df: pl.DataFrame,
    metric: str = "hellinger",
    k: int = 10,
) -> dict[str, Any]:
    """Construct Level 1 Opinion-Profile Graph over unique count vectors.

    Nodes are the unique count vectors (1,604 nodes), weighted by frequency.
    Edges represent Hellinger distance between distinct opinion profiles.
    """
    # Group items by exact 3-class probability profile
    prob_cols = ["human_p_entailment", "human_p_neutral", "human_p_contradiction"]

    profile_df = df.group_by(prob_cols).agg([
        pl.len().alias("profile_frequency"),
        pl.col("object_id").alias("item_ids"),
        pl.col("source_dataset").alias("datasets"),
        pl.col("human_entropy_bits").first().alias("entropy_bits"),
    ]).sort("profile_frequency", descending=True)

    n_profiles = len(profile_df)
    profile_ids = [f"profile_{i:04d}" for i in range(n_profiles)]
    profile_df = profile_df.with_columns(pl.Series("profile_id", profile_ids))

    p_profiles = profile_df.select(prob_cols).to_numpy()

    # Exact distance matrix between unique profiles
    dist_matrix = build_distance_matrix(p_profiles, metric=metric)

    # Extract k-NN graph over unique profiles (no ties at zero distance!)
    effective_k = min(k, n_profiles - 1)
    knn_idx, neighbor_df = extract_knn_graph(
        dist_matrix, profile_ids, k=effective_k, space_id="opinion_profile", metric_id=metric
    )

    return {
        "n_profiles": n_profiles,
        "n_total_items": len(df),
        "profile_df": profile_df,
        "dist_matrix": dist_matrix,
        "knn_idx": knn_idx,
        "neighbor_df": neighbor_df,
    }


def analyze_level2_profile_heterogeneity(
    df: pl.DataFrame,
    profile_df: pl.DataFrame,
) -> pl.DataFrame:
    """Analyze Level 2 item heterogeneity within top multi-item opinion profiles."""
    multi_profiles = profile_df.filter(pl.col("profile_frequency") > 1)

    records = []
    for row in multi_profiles.iter_rows(named=True):
        items = df.filter(pl.col("object_id").is_in(row["item_ids"]))

        n_snli = items.filter(pl.col("source_dataset") == "chaosnli_snli").height
        n_mnli = items.filter(pl.col("source_dataset") == "chaosnli_mnli").height

        records.append({
            "profile_id": row["profile_id"],
            "frequency": row["profile_frequency"],
            "entropy_bits": row["entropy_bits"],
            "n_snli": n_snli,
            "n_mnli": n_mnli,
            "p_entailment": row["human_p_entailment"],
            "p_neutral": row["human_p_neutral"],
            "p_contradiction": row["human_p_contradiction"],
        })

    return pl.DataFrame(records)
