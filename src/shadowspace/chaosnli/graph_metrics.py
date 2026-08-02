"""Graph comparison metrics module for Q_NX, LCMC, local overlap, and split-half reliability."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.neighbors import extract_knn_graph
from shadowspace.chaosnli.posterior import compute_split_half_distributions


def compute_local_overlap(knn_ref: np.ndarray, knn_comp: np.ndarray) -> np.ndarray:
    """Compute local neighborhood overlap O_i(k) for each node i.

    O_i(k) = |N_ref(i; k) intersect N_comp(i; k)| / k
    """
    n, k = knn_ref.shape
    overlap = np.zeros(n, dtype=np.float64)

    for i in range(n):
        set_ref = set(knn_ref[i])
        set_comp = set(knn_comp[i])
        overlap[i] = len(set_ref & set_comp) / float(k)

    return overlap


def compute_qnx(knn_ref: np.ndarray, knn_comp: np.ndarray) -> float:
    """Compute global neighborhood preservation Q_NX(k).

    Q_NX(k) = 1 / (N * k) * sum_i |N_ref(i; k) intersect N_comp(i; k)|
    """
    overlap = compute_local_overlap(knn_ref, knn_comp)
    return float(overlap.mean())


def compute_lcmc(qnx_val: float, n_items: int, k: int) -> float:
    """Compute Local Continuity Meta-Criterion LCMC(k).

    LCMC(k) = Q_NX(k) - k / (N - 1)
    """
    chance_baseline = float(k) / max(float(n_items - 1), 1.0)
    return float(qnx_val - chance_baseline)


def compute_human_split_half_reliability(
    counts: np.ndarray,
    k: int = 10,
    n_repetitions: int = 100,
    metric: str = "hellinger",
    seed: int = 20260801,
) -> dict[str, Any]:
    """Compute human split-half graph reliability distribution over N repetitions.

    Repeatedly splits 100 human votes into 50/50, constructs both Hellinger graphs,
    and evaluates Q_NX(k) between the two human halves.
    """
    n_items = len(counts)
    rng = np.random.default_rng(seed)
    qnx_scores = np.zeros(n_repetitions, dtype=np.float64)
    lcmc_scores = np.zeros(n_repetitions, dtype=np.float64)

    seeds = rng.integers(0, 2**31 - 1, size=n_repetitions)

    for rep in range(n_repetitions):
        p1, p2 = compute_split_half_distributions(counts, seed=int(seeds[rep]))

        d1 = build_distance_matrix(p1, metric=metric)
        d2 = build_distance_matrix(p2, metric=metric)

        dummy_ids = [str(i) for i in range(n_items)]
        knn1, _ = extract_knn_graph(d1, dummy_ids, k=k, metric_id=metric)
        knn2, _ = extract_knn_graph(d2, dummy_ids, k=k, metric_id=metric)

        qnx = compute_qnx(knn1, knn2)
        lcmc = compute_lcmc(qnx, n_items, k=k)

        qnx_scores[rep] = qnx
        lcmc_scores[rep] = lcmc

    return {
        "k": k,
        "metric": metric,
        "n_repetitions": n_repetitions,
        "qnx_median": float(np.median(qnx_scores)),
        "qnx_mean": float(qnx_scores.mean()),
        "qnx_q025": float(np.quantile(qnx_scores, 0.025)),
        "qnx_q975": float(np.quantile(qnx_scores, 0.975)),
        "lcmc_median": float(np.median(lcmc_scores)),
        "lcmc_mean": float(lcmc_scores.mean()),
        "qnx_scores": qnx_scores.tolist(),
    }
