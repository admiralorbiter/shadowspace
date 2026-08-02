"""Model topology evaluation and hypothesis testing module (H1 and H2)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.models import compute_model_probabilities
from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights, compute_soft_qnx


def evaluate_model_topology_recovery(
    model_results: dict[str, dict[str, np.ndarray]],
    canonical_items_path: pl.DataFrame | str = "data/chaosnli/processed/canonical_items_posterior.parquet",
    k: int = 10,
    metric: str = "hellinger",
    qnx_hh_soft: float = 0.0426,
) -> dict[str, Any]:
    """Evaluate topological neighborhood recovery Q_NX_soft for all models against human opinion space.

    Tests Hypothesis 1: Model recovery vs human split-half baseline.
    """
    if isinstance(canonical_items_path, pl.DataFrame):
        canon_df = canonical_items_path
    else:
        canon_df = pl.read_parquet(canonical_items_path)

    prob_cols = ["human_p_entailment", "human_p_neutral", "human_p_contradiction"]
    p_human = canon_df.select(prob_cols).to_numpy()

    d_human = build_distance_matrix(p_human, metric=metric)
    w_human = compute_soft_neighborhood_weights(d_human, k=k)

    qnx_chance = float(k / (len(canon_df) - 1))

    model_evaluations: dict[str, dict[str, Any]] = {}

    for model_name, m_data in model_results.items():
        logits = m_data["logits"]
        q_model = compute_model_probabilities(logits, temperature=1.0)

        d_model = build_distance_matrix(q_model, metric=metric)
        w_model = compute_soft_neighborhood_weights(d_model, k=k)

        qnx_hm_soft, local_o = compute_soft_qnx(w_human, w_model, k=k)

        # Excess-over-chance ratio
        excess_ratio = (qnx_hm_soft - qnx_chance) / max(qnx_hh_soft - qnx_chance, 1e-6)

        # Pointwise mean JSD
        m_mix = 0.5 * (p_human + q_model)
        kl_p = np.sum(np.where(p_human > 0, p_human * np.log(np.maximum(p_human, 1e-12) / m_mix), 0.0), axis=1)
        kl_q = np.sum(np.where(q_model > 0, q_model * np.log(np.maximum(q_model, 1e-12) / m_mix), 0.0), axis=1)
        mean_jsd = float(np.mean(0.5 * (kl_p + kl_q) / np.log(2.0)))

        model_evaluations[model_name] = {
            "model_name": model_name,
            "qnx_soft_hm": float(qnx_hm_soft),
            "qnx_chance": qnx_chance,
            "qnx_hh_soft": qnx_hh_soft,
            "excess_ratio_vs_human": float(excess_ratio),
            "mean_pointwise_jsd_bits": mean_jsd,
            "h1_confirmed": bool(qnx_hm_soft < qnx_hh_soft),
        }

    return model_evaluations


def evaluate_hypothesis2_temperature_scaling(
    model_results: dict[str, dict[str, np.ndarray]],
    canon_df: pl.DataFrame,
    temperatures: list[float] = [0.5, 1.0, 1.5, 2.0],
    k: int = 10,
    metric: str = "hellinger",
) -> dict[str, list[dict[str, Any]]]:
    """Test Hypothesis 2: Temperature scaling effects on pointwise JSD vs topological recovery.

    H2 states: Temperature scaling alters pointwise distribution calibration (JSD) without changing
    monotonic rank ordering within distance spaces (preserving Q_NX_soft).
    """
    prob_cols = ["human_p_entailment", "human_p_neutral", "human_p_contradiction"]
    p_human = canon_df.select(prob_cols).to_numpy()

    d_human = build_distance_matrix(p_human, metric=metric)
    w_human = compute_soft_neighborhood_weights(d_human, k=k)

    results_by_model: dict[str, list[dict[str, Any]]] = {}

    for model_name, m_data in model_results.items():
        logits = m_data["logits"]
        temp_curve = []

        for T in temperatures:
            q_T = compute_model_probabilities(logits, temperature=T)

            # Pointwise JSD
            m_mix = 0.5 * (p_human + q_T)
            kl_p = np.sum(np.where(p_human > 0, p_human * np.log(np.maximum(p_human, 1e-12) / m_mix), 0.0), axis=1)
            kl_q = np.sum(np.where(q_T > 0, q_T * np.log(np.maximum(q_T, 1e-12) / m_mix), 0.0), axis=1)
            mean_jsd = float(np.mean(0.5 * (kl_p + kl_q) / np.log(2.0)))

            d_model_T = build_distance_matrix(q_T, metric=metric)
            w_model_T = compute_soft_neighborhood_weights(d_model_T, k=k)
            qnx_soft_T, _ = compute_soft_qnx(w_human, w_model_T, k=k)

            temp_curve.append({
                "temperature": T,
                "mean_jsd_bits": mean_jsd,
                "qnx_soft_hm": float(qnx_soft_T),
            })

        results_by_model[model_name] = temp_curve

    return results_by_model
