"""Model topology evaluation and hypothesis testing module (H1 and H2)."""

from __future__ import annotations

from pathlib import Path
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
        logits_full = m_data["logits"]

        # Map item object_ids to model logits indices
        if "object_ids" in m_data and len(m_data["object_ids"]) > 0:
            m_obj_to_idx = {obj_id: idx for idx, obj_id in enumerate(m_data["object_ids"])}
            df_indices = []
            for obj_id in canon_df["object_id"]:
                if obj_id not in m_obj_to_idx:
                    raise KeyError(f"Item object_id '{obj_id}' not found in model predictions for '{model_name}'.")
                df_indices.append(m_obj_to_idx[obj_id])
        elif len(logits_full) == len(canon_df):
            df_indices = list(range(len(canon_df)))
        elif Path("data/chaosnli/processed/canonical_items_posterior.parquet").exists():
            all_canon_df = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")
            obj_id_to_idx = {obj_id: idx for idx, obj_id in enumerate(all_canon_df["object_id"])}
            df_indices = [obj_id_to_idx[obj_id] for obj_id in canon_df["object_id"]]
        else:
            df_indices = list(range(len(canon_df)))

        logits = logits_full[df_indices]
        q_model = compute_model_probabilities(logits, temperature=1.0)

        d_model = build_distance_matrix(q_model, metric=metric)
        w_model = compute_soft_neighborhood_weights(d_model, k=k)

        qnx_hm_soft, local_o = compute_soft_qnx(w_human, w_model, k=k)

        # Stratified 95% Equal-Tailed Bootstrap CI over local soft overlap scores
        # Stratify by source dataset
        if "source_dataset" in canon_df.columns:
            snli_mask = (canon_df["source_dataset"] == "chaosnli_snli").to_numpy()
            mnli_mask = ~snli_mask
            n_snli = int(snli_mask.sum())
            n_mnli = int(mnli_mask.sum())

            rng = np.random.default_rng(20260801)
            boot_scores = []
            for _ in range(200):
                idx_snli = rng.choice(np.where(snli_mask)[0], size=n_snli, replace=True) if n_snli > 0 else np.array([], dtype=int)
                idx_mnli = rng.choice(np.where(mnli_mask)[0], size=n_mnli, replace=True) if n_mnli > 0 else np.array([], dtype=int)
                boot_idx = np.concatenate([idx_snli, idx_mnli]) if len(idx_snli) + len(idx_mnli) > 0 else np.arange(len(canon_df))
                boot_scores.append(float(local_o[boot_idx].mean()))
        else:
            rng = np.random.default_rng(20260801)
            n_df = len(canon_df)
            boot_scores = [float(local_o[rng.choice(n_df, size=n_df, replace=True)].mean()) for _ in range(200)]

        ci_lower = float(np.percentile(boot_scores, 2.5))
        ci_upper = float(np.percentile(boot_scores, 97.5))

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
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "qnx_chance": qnx_chance,
            "qnx_hh_soft": qnx_hh_soft,
            "excess_ratio_vs_human": float(excess_ratio),
            "mean_pointwise_jsd_bits": mean_jsd,
            "all_point_estimates_below_human": bool(ci_upper < qnx_hh_soft),
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

    Computes direct model-to-model graph turnover between T=1.0 base and other temperatures.
    """
    prob_cols = ["human_p_entailment", "human_p_neutral", "human_p_contradiction"]
    p_human = canon_df.select(prob_cols).to_numpy()

    d_human = build_distance_matrix(p_human, metric=metric)
    w_human = compute_soft_neighborhood_weights(d_human, k=k)

    results_by_model: dict[str, list[dict[str, Any]]] = {}

    for model_name, m_data in model_results.items():
        logits_full = m_data["logits"]

        # Map item object_ids to model logits indices
        if "object_ids" in m_data and len(m_data["object_ids"]) > 0:
            m_obj_to_idx = {obj_id: idx for idx, obj_id in enumerate(m_data["object_ids"])}
            df_indices = []
            for obj_id in canon_df["object_id"]:
                if obj_id not in m_obj_to_idx:
                    raise KeyError(f"Item object_id '{obj_id}' not found in model predictions for '{model_name}'.")
                df_indices.append(m_obj_to_idx[obj_id])
        elif len(logits_full) == len(canon_df):
            df_indices = list(range(len(canon_df)))
        elif Path("data/chaosnli/processed/canonical_items_posterior.parquet").exists():
            all_canon_df = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")
            obj_id_to_idx = {obj_id: idx for idx, obj_id in enumerate(all_canon_df["object_id"])}
            df_indices = [obj_id_to_idx[obj_id] for obj_id in canon_df["object_id"]]
        else:
            df_indices = list(range(len(canon_df)))

        logits = logits_full[df_indices]
        temp_curve = []

        # Baseline weights at T=1.0
        q_base = compute_model_probabilities(logits, temperature=1.0)
        d_base = build_distance_matrix(q_base, metric=metric)
        w_base = compute_soft_neighborhood_weights(d_base, k=k)

        for T in temperatures:
            q_T = compute_model_probabilities(logits, temperature=T)

            # Pointwise JSD
            m_mix = 0.5 * (p_human + q_T)
            kl_p = np.sum(np.where(p_human > 0, p_human * np.log(np.maximum(p_human, 1e-12) / m_mix), 0.0), axis=1)
            kl_q = np.sum(np.where(q_T > 0, q_T * np.log(np.maximum(q_T, 1e-12) / m_mix), 0.0), axis=1)
            mean_jsd = float(np.mean(0.5 * (kl_p + kl_q) / np.log(2.0)))

            d_model_T = build_distance_matrix(q_T, metric=metric)
            w_model_T = compute_soft_neighborhood_weights(d_model_T, k=k)

            # Model vs Human soft overlap
            qnx_soft_T, _ = compute_soft_qnx(w_human, w_model_T, k=k)

            # Direct Model-to-Model graph overlap with T=1.0 base
            qnx_model_self, _ = compute_soft_qnx(w_base, w_model_T, k=k)
            edge_turnover = 1.0 - qnx_model_self

            temp_curve.append({
                "temperature": T,
                "mean_jsd_bits": mean_jsd,
                "qnx_soft_hm": float(qnx_soft_T),
                "qnx_model_self_vs_t1": float(qnx_model_self),
                "edge_turnover_vs_t1": float(edge_turnover),
            })

        results_by_model[model_name] = temp_curve

    return results_by_model
