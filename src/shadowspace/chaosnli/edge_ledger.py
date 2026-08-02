"""Persistent Edge Ledger and 6-Category Diagnostic Taxonomy module."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.models import compute_model_probabilities
from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights


def build_persistent_edge_ledger(
    df: pl.DataFrame,
    model_results: dict[str, dict[str, np.ndarray]],
    d_text: np.ndarray | None = None,
    k: int = 10,
    metric: str = "hellinger",
    use_quantiles: bool = True,
) -> pl.DataFrame:
    """Build persistent edge ledger across all candidate directed edges (i, j).

    Measures:
      s_ij: Dirichlet posterior human edge support / weight
      c_ij: Cross-model consensus fraction
      t_ij: Text-semantic similarity (normalized inverse text distance)

    Classifies every edge into the 6 diagnostic categories using quantile or absolute thresholds.
    """
    n = len(df)
    obj_ids = df["object_id"].to_list()

    p_human = df.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    d_human = build_distance_matrix(p_human, metric=metric)
    w_human = compute_soft_neighborhood_weights(d_human, k=k)

    # Compute model consensus matrix C_ij
    n_models = len(model_results)
    w_models_sum = np.zeros((n, n), dtype=np.float32)

    for m_name, m_data in model_results.items():
        logits = m_data["logits"]
        q_m = compute_model_probabilities(logits, temperature=1.0)
        d_m = build_distance_matrix(q_m, metric=metric)
        w_m = compute_soft_neighborhood_weights(d_m, k=k)
        w_models_sum += (w_m > 0).astype(np.float32)

    c_consensus = w_models_sum / max(n_models, 1)

    # Text support matrix T_ij (Cosine similarity = 1 - d_text)
    if d_text is not None:
        t_sim = 1.0 - d_text
    else:
        t_sim = np.zeros((n, n), dtype=np.float32)

    # Filter candidate edges where either human support > 0 or model consensus > 0
    cand_mask = (w_human > 0) | (c_consensus > 0)
    # Exclude diagonal
    np.fill_diagonal(cand_mask, False)

    # Extract values for candidate edges to compute distribution quantiles
    s_vals = w_human[cand_mask]
    c_vals = c_consensus[cand_mask]
    t_vals = t_sim[cand_mask]

    if use_quantiles:
        # Quantile thresholds based on non-zero candidate distribution
        s_high = float(np.percentile(s_vals[s_vals > 0], 75)) if np.any(s_vals > 0) else 0.5
        s_low = float(np.percentile(s_vals[s_vals > 0], 25)) if np.any(s_vals > 0) else 0.1
        c_high = float(np.percentile(c_vals[c_vals > 0], 75)) if np.any(c_vals > 0) else 0.5
        c_low = float(np.percentile(c_vals[c_vals > 0], 25)) if np.any(c_vals > 0) else 0.1
        t_high = float(np.percentile(t_vals, 75))
        t_low = float(np.percentile(t_vals, 25))
    else:
        s_high, s_low = 0.8, 0.2
        c_high, c_low = 0.8, 0.2
        t_high, t_low = 0.10, 0.05

    records = []
    indices_i, indices_j = np.where(cand_mask)

    for idx in range(len(indices_i)):
        i = int(indices_i[idx])
        j = int(indices_j[idx])

        s_ij = float(w_human[i, j])
        c_ij = float(c_consensus[i, j])
        t_ij = float(t_sim[i, j])

        high_s = s_ij >= s_high
        low_s = s_ij <= s_low
        high_c = c_ij >= c_high
        low_c = c_ij <= c_low
        high_t = t_ij >= t_high
        low_t = t_ij <= t_low

        if high_s and high_c and high_t:
            cat = "broadly_shared_relation"
        elif high_s and low_c and high_t:
            cat = "human_relation_missed_by_models"
        elif low_s and high_c and low_t:
            cat = "model_family_artifact_candidate"
        elif low_s and high_c and high_t:
            cat = "semantic_similarity_divergence"
        elif high_s and low_c and low_t:
            cat = "same_opinion_behavior_distinct_language"
        elif s_low < s_ij < s_high:
            cat = "insufficient_annotation_support"
        else:
            cat = "unclassified_intermediate"

        records.append({
            "source_id": obj_ids[i],
            "target_id": obj_ids[j],
            "s_human_support": s_ij,
            "c_model_consensus": c_ij,
            "t_text_support": t_ij,
            "diagnostic_category": cat,
        })

    return pl.DataFrame(records)
