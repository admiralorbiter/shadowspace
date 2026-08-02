"""Tie and Multiplicity Audit module for ChaosNLI probability spaces."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.graph_metrics import compute_qnx


def run_multiplicity_and_tie_audit(
    df: pl.DataFrame,
    dist_matrix: np.ndarray,
    k: int = 10,
    n_permutations: int = 20,
    seed: int = 20260801,
) -> dict[str, Any]:
    """Perform comprehensive multiplicity, tie-density, and tie-break sensitivity audit.

    Args:
        df: Polars DataFrame of canonical items.
        dist_matrix: (N, N) distance matrix.
        k: Neighborhood size k.
        n_permutations: Number of row permutation iterations.
        seed: Random seed.

    Returns:
        Audit dictionary with exact statistics.
    """
    n = len(df)
    rng = np.random.default_rng(seed)

    # 1. Multiplicity analysis over exact 3-class count vectors
    counts = df.select(
        ["human_count_entailment", "human_count_neutral", "human_count_contradiction"]
    ).to_struct("count_vec")

    vector_counts = counts.value_counts().sort("count", descending=True)
    unique_profiles = len(vector_counts)

    # Non-singleton profiles (count > 1)
    non_singletons = vector_counts.filter(pl.col("count") > 1)
    items_in_non_singletons = int(non_singletons["count"].sum())
    max_multiplicity = int(vector_counts["count"].max())

    # Multiplicity histogram
    multiplicity_hist = vector_counts["count"].value_counts(name="multiplicity").to_dicts()

    # 2. Distance tie analysis at k-boundary
    items_with_tie_before_k = 0
    items_with_tie_at_k = 0
    boundary_tie_sizes = []

    for i in range(n):
        row = dist_matrix[i].copy()
        row[i] = np.inf  # Exclude self
        sorted_d = np.sort(row)

        k_dist = sorted_d[k - 1]

        # Check ties before k
        closer_dist = sorted_d[: k - 1]
        if len(closer_dist) > 0 and len(np.unique(closer_dist)) < len(closer_dist):
            items_with_tie_before_k += 1

        # Check ties at k-th boundary distance
        ties_at_k = np.isclose(row, k_dist, atol=1e-7).sum()
        if ties_at_k > 1:
            items_with_tie_at_k += 1
            boundary_tie_sizes.append(int(ties_at_k))

    pct_non_singletons = items_in_non_singletons / n
    pct_tie_before_k = items_with_tie_before_k / n
    pct_tie_at_k = items_with_tie_at_k / n
    median_boundary_tie = float(np.median(boundary_tie_sizes)) if boundary_tie_sizes else 1.0

    # 3. Sensitivity of Q_NX(k) under row permutations (unstable sorting tie breaks)
    qnx_permutations = []
    ids = [str(i) for i in range(n)]

    # Build reference graph with natural order
    from shadowspace.chaosnli.neighbors import extract_knn_graph

    knn_ref, _ = extract_knn_graph(dist_matrix, ids, k=k)

    for p_idx in range(n_permutations):
        perm = rng.permutation(n)
        perm_dist = dist_matrix[perm][:, perm]
        perm_ids = [ids[idx] for idx in perm]

        knn_perm_local, _ = extract_knn_graph(perm_dist, perm_ids, k=k)

        # Map permuted neighbor indices back to original indices
        knn_perm_unmapped = np.zeros_like(knn_perm_local)
        for i in range(n):
            orig_i = perm[i]
            knn_perm_unmapped[orig_i] = perm[knn_perm_local[i]]

        qnx_p = compute_qnx(knn_ref, knn_perm_unmapped)
        qnx_permutations.append(float(qnx_p))

    return {
        "n_items": n,
        "unique_profiles": unique_profiles,
        "items_in_non_singleton_profiles": items_in_non_singletons,
        "pct_items_in_non_singleton_profiles": pct_non_singletons,
        "max_profile_multiplicity": max_multiplicity,
        "items_with_tie_before_k": items_with_tie_before_k,
        "pct_with_tie_before_k": pct_tie_before_k,
        "items_with_tie_at_k": items_with_tie_at_k,
        "pct_with_tie_at_k": pct_tie_at_k,
        "median_boundary_tie_size": median_boundary_tie,
        "qnx_permutation_mean": float(np.mean(qnx_permutations)),
        "qnx_permutation_std": float(np.std(qnx_permutations)),
        "qnx_permutation_min": float(np.min(qnx_permutations)),
        "qnx_permutation_max": float(np.max(qnx_permutations)),
    }
