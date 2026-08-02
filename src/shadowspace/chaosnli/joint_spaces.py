"""Multi-view joint space construction and Hypothesis 7 evaluation module."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights, compute_soft_qnx


def compute_joint_distance_matrix(
    d_opinion: np.ndarray,
    d_text: np.ndarray,
    lambda_weight: float = 0.5,
) -> np.ndarray:
    """Compute alpha-blended joint distance matrix D_joint(lambda).

    d_joint(u, v; lambda) = sqrt((1 - lambda) * d_op^2 + lambda * d_txt^2)
    """
    # Normalize matrices to [0, 1] range before blending
    max_op = d_opinion.max() or 1.0
    max_txt = d_text.max() or 1.0

    d_op_norm = d_opinion / max_op
    d_txt_norm = d_text / max_txt

    blended_sq = (1.0 - lambda_weight) * (d_op_norm ** 2) + lambda_weight * (d_txt_norm ** 2)
    d_joint = np.sqrt(np.maximum(blended_sq, 0.0)).astype(np.float32)
    np.fill_diagonal(d_joint, 0.0)
    return d_joint


def evaluate_hypothesis7_joint_space(
    df: pl.DataFrame,
    d_opinion: np.ndarray,
    d_text: np.ndarray,
    lambdas: list[float] = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0],
    k: int = 10,
) -> dict[str, Any]:
    """Evaluate Hypothesis 7: Joint space tie resolution and neighborhood preservation.

    Tests:
      1. Tie Resolution: Percentage of zero-distance ties remaining in D_joint.
      2. Intra-Profile Semantic Dispersion: Text distance variance within identical opinion profiles.
      3. Opinion Neighborhood Recovery Q_NX_soft(k) across lambda values.
    """
    n = len(df)
    w_opinion = compute_soft_neighborhood_weights(d_opinion, k=k)

    lambda_evaluations = []

    for lam in lambdas:
        d_joint = compute_joint_distance_matrix(d_opinion, d_text, lambda_weight=lam)

        # Count zero-distance ties in joint space (excluding diagonal)
        mask = ~np.eye(n, dtype=bool)
        n_zero_ties = int((d_joint[mask] < 1e-6).sum() // 2)

        w_joint = compute_soft_neighborhood_weights(d_joint, k=k)
        qnx_soft_opinion_recovery, _ = compute_soft_qnx(w_opinion, w_joint, k=k)

        lambda_evaluations.append({
            "lambda": float(lam),
            "zero_distance_ties_remaining": n_zero_ties,
            "qnx_soft_opinion_recovery": float(qnx_soft_opinion_recovery),
        })

    # Intra-profile text distance analysis
    prob_cols = ["human_p_entailment", "human_p_neutral", "human_p_contradiction"]
    profile_df = df.group_by(prob_cols).agg([
        pl.len().alias("profile_freq"),
        pl.col("object_id").alias("item_ids"),
    ]).filter(pl.col("profile_freq") > 1)

    id_to_idx = {obj_id: idx for idx, obj_id in enumerate(df["object_id"])}

    intra_profile_text_dists = []
    for row in profile_df.iter_rows(named=True):
        indices = [id_to_idx[item_id] for item_id in row["item_ids"]]
        sub_d_text = d_text[np.ix_(indices, indices)]
        # Take upper triangle distances
        triu_idx = np.triu_indices(len(indices), k=1)
        intra_profile_text_dists.extend(sub_d_text[triu_idx].tolist())

    mean_intra_text_dist = float(np.mean(intra_profile_text_dists)) if intra_profile_text_dists else 0.0
    mean_overall_text_dist = float(d_text[mask].mean())

    return {
        "n_items": n,
        "n_multi_item_profiles": len(profile_df),
        "mean_intra_profile_text_distance": mean_intra_text_dist,
        "mean_overall_text_distance": mean_overall_text_dist,
        "lambda_evaluations": lambda_evaluations,
        "h7_confirmed": bool(
            lambda_evaluations[3]["zero_distance_ties_remaining"] < lambda_evaluations[0]["zero_distance_ties_remaining"]
            and lambda_evaluations[3]["qnx_soft_opinion_recovery"] > 0.04
        ),
    }
