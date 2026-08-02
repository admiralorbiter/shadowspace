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


def compute_profile_level_model_dispersion(
    df: pl.DataFrame,
    profile_df: pl.DataFrame,
    model_results: dict[str, dict[str, np.ndarray]],
    metric: str = "hellinger",
) -> pl.DataFrame:
    """Calculate profile-level model dispersion for all multi-item human opinion profiles.

    For profile g with |g| > 1 items, ModelDispersion(g) = (2 / (|g|(|g|-1))) * sum_{i<j in g} d(q_i, q_j).
    """
    from shadowspace.chaosnli.distances import build_distance_matrix
    from shadowspace.chaosnli.models import compute_model_probabilities

    multi_profiles = profile_df.filter(pl.col("profile_frequency") > 1)
    obj_id_list = df["object_id"].to_list()
    obj_id_to_idx = {obj_id: idx for idx, obj_id in enumerate(obj_id_list)}

    records = []

    for row in multi_profiles.iter_rows(named=True):
        item_ids = row["item_ids"]
        item_indices = np.array([obj_id_to_idx[item_id] for item_id in item_ids if item_id in obj_id_to_idx])
        g_size = len(item_indices)

        if g_size < 2:
            continue

        rec: dict[str, Any] = {
            "profile_id": row["profile_id"],
            "profile_frequency": g_size,
            "entropy_bits": row["entropy_bits"],
            "p_entailment": row["human_p_entailment"],
            "p_neutral": row["human_p_neutral"],
            "p_contradiction": row["human_p_contradiction"],
        }

        disp_sum = 0.0
        n_models = len(model_results)

        for m_name, m_data in model_results.items():
            logits_g = m_data["logits"][item_indices]
            q_g = compute_model_probabilities(logits_g, temperature=1.0)
            d_g = build_distance_matrix(q_g, metric=metric)

            # Pairwise mean distance among items in g
            triu_idx = np.triu_indices(g_size, k=1)
            mean_disp = float(np.mean(d_g[triu_idx]))
            rec[f"dispersion_{m_name}"] = mean_disp
            disp_sum += mean_disp

        rec["mean_model_dispersion"] = float(disp_sum / max(n_models, 1))
        records.append(rec)

    return pl.DataFrame(records)


def analyze_model_dispersion_drivers(
    df: pl.DataFrame,
    dispersion_df: pl.DataFrame,
) -> dict[str, float]:
    """Analyze statistical drivers of profile-level model dispersion.

    Correlates mean_model_dispersion with:
      - entropy_bits (Shannon entropy of human opinion profile)
      - profile_frequency (Number of items sharing identical profile)
      - max_class_p (Degree of consensus / dominance of top class)
    """
    if len(dispersion_df) == 0:
        return {}

    disp = dispersion_df["mean_model_dispersion"].to_numpy()
    entropy = dispersion_df["entropy_bits"].to_numpy()
    freq = dispersion_df["profile_frequency"].to_numpy()

    p_e = dispersion_df["p_entailment"].to_numpy()
    p_n = dispersion_df["p_neutral"].to_numpy()
    p_c = dispersion_df["p_contradiction"].to_numpy()
    max_p = np.maximum(p_e, np.maximum(p_n, p_c))

    def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    return {
        "corr_dispersion_entropy": pearson_r(disp, entropy),
        "corr_dispersion_frequency": pearson_r(disp, freq),
        "corr_dispersion_max_class_p": pearson_r(disp, max_p),
    }
