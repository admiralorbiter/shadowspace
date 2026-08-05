"""Fast Cached Numerical Cluster Bootstrap Inference (B=10,000 Resamples across base_sentence_id Clusters)."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def compute_cluster_aware_drift_metrics(
    pairs_data: List[Dict[str, Any]],
    evaluator,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Pre-evaluates predictions once, caches numeric arrays, and performs fast cluster percentile bootstrap resampling."""
    if not pairs_data:
        return {}

    # Pre-evaluate scores once
    all_scores_m = []
    all_scores_f = []
    all_deltas = []

    cluster_to_indices: Dict[str, List[int]] = {}
    for idx, p in enumerate(pairs_data):
        cid = p["base_sentence_id"]
        if cid not in cluster_to_indices:
            cluster_to_indices[cid] = []
        cluster_to_indices[cid].append(idx)

        s_m = evaluator.predict_score(p["text_masc"])
        s_f = evaluator.predict_score(p["text_fem"])
        d = s_m - s_f

        all_scores_m.append(s_m)
        all_scores_f.append(s_f)
        all_deltas.append(d)

    arr_deltas = np.array(all_deltas)
    arr_m = np.array(all_scores_m)
    arr_f = np.array(all_scores_f)

    cluster_ids = list(cluster_to_indices.keys())
    N_clusters = len(cluster_ids)

    msd = float(np.mean(arr_deltas))
    masd = float(np.mean(np.abs(arr_deltas)))
    max_abs = float(np.max(np.abs(arr_deltas)))

    th = evaluator.provenance.threshold
    flips = np.sum((arr_m >= th) != (arr_f >= th))
    cfr = float(flips / len(arr_deltas))

    # Fast Numerical Cluster Percentile Bootstrap Resampling (B=10,000)
    rng = np.random.default_rng(42)
    cluster_abs_deltas = [np.abs(arr_deltas[cluster_to_indices[cid]]) for cid in cluster_ids]

    boot_masds = np.zeros(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sampled_cids = rng.choice(N_clusters, size=N_clusters, replace=True)
        sample_sum = 0.0
        sample_count = 0
        for cid in sampled_cids:
            c_arr = cluster_abs_deltas[cid]
            sample_sum += np.sum(c_arr)
            sample_count += len(c_arr)
        boot_masds[b] = sample_sum / sample_count

    ci_lower = float(np.percentile(boot_masds, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_masds, 100 * (1.0 - alpha / 2.0)))

    return {
        "total_pairs_N": len(arr_deltas),
        "independent_clusters_count_N": N_clusters,
        "msd_mean_signed_drift": round(msd, 4),
        "masd_mean_absolute_score_difference": round(masd, 4),
        "masd_cluster_percentile_ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "cfr_counterfactual_flip_rate": round(cfr, 4),
        "flips_count": int(flips),
        "max_absolute_drift": round(max_abs, 4),
        "raw_deltas": arr_deltas,
        "pairs_data": pairs_data,
    }
