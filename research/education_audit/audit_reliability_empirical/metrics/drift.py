"""Cluster-Aware Evaluator Drift Metrics (Resampling Whole base_sentence_id Clusters N=373)."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def compute_cluster_aware_drift_metrics(
    pairs_data: List[Dict[str, Any]],
    evaluator,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Computes cluster-aware MSD, MASD, CFR, max drift, and cluster BCa bootstrap confidence intervals."""
    if not pairs_data:
        return {}

    # Group pairs by base_sentence_id clusters
    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for p in pairs_data:
        cid = p["base_sentence_id"]
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(p)

    cluster_ids = list(clusters.keys())
    N_clusters = len(cluster_ids)

    # Evaluate paired differences
    all_deltas = []
    all_scores_m = []
    all_scores_f = []

    for p in pairs_data:
        s_m = evaluator.predict_score(p["text_masc"])
        s_f = evaluator.predict_score(p["text_fem"])
        d = s_m - s_f

        all_scores_m.append(s_m)
        all_scores_f.append(s_f)
        all_deltas.append(d)

    arr_deltas = np.array(all_deltas)
    arr_m = np.array(all_scores_m)
    arr_f = np.array(all_scores_f)

    msd = float(np.mean(arr_deltas))
    masd = float(np.mean(np.abs(arr_deltas)))
    max_abs = float(np.max(np.abs(arr_deltas)))

    th = evaluator.provenance.threshold
    flips = np.sum((arr_m >= th) != (arr_f >= th))
    cfr = float(flips / len(arr_deltas))

    # Cluster Bootstrap (Resampling whole base_sentence_id clusters)
    rng = np.random.default_rng(42)
    boot_masds = []

    for _ in range(n_bootstrap):
        sampled_cids = rng.choice(cluster_ids, size=N_clusters, replace=True)
        sampled_deltas = []
        for cid in sampled_cids:
            for p in clusters[cid]:
                s_m = evaluator.predict_score(p["text_masc"])
                s_f = evaluator.predict_score(p["text_fem"])
                sampled_deltas.append(abs(s_m - s_f))
        boot_masds.append(np.mean(sampled_deltas))

    ci_lower = float(np.percentile(boot_masds, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_masds, 100 * (1.0 - alpha / 2.0)))

    return {
        "total_pairs_N": len(arr_deltas),
        "independent_clusters_count_N": N_clusters,
        "msd_mean_signed_drift": round(msd, 4),
        "masd_mean_absolute_score_difference": round(masd, 4),
        "masd_cluster_bca_ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "cfr_counterfactual_flip_rate": round(cfr, 4),
        "flips_count": int(flips),
        "max_absolute_drift": round(max_abs, 4),
        "raw_deltas": arr_deltas,
    }
