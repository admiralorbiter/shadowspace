"""Evaluator Drift Metrics (MSD, MASD, CFR, and BCa Paired Bootstrap Intervals)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
from scipy import stats


def compute_evaluator_drift_metrics(
    deltas: np.ndarray,
    scores_masc: np.ndarray,
    scores_fem: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Computes MSD, MASD, CFR, max drift, and paired BCa bootstrap confidence intervals."""
    N = len(deltas)
    if N == 0:
        return {}

    abs_deltas = np.abs(deltas)
    msd = float(np.mean(deltas))
    masd = float(np.mean(abs_deltas))
    max_abs = float(np.max(abs_deltas))

    flips = np.sum((scores_masc >= threshold) != (scores_fem >= threshold))
    cfr = float(flips / N)

    # Paired BCa Bootstrap for MASD
    boot_masds = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        boot_masds.append(np.mean(abs_deltas[idx]))

    ci_lower = float(np.percentile(boot_masds, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_masds, 100 * (1.0 - alpha / 2.0)))

    return {
        "sample_size_N": N,
        "msd_mean_signed_drift": round(msd, 4),
        "masd_mean_absolute_score_difference": round(masd, 4),
        "masd_bca_ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "cfr_counterfactual_flip_rate": round(cfr, 4),
        "flips_count": int(flips),
        "max_absolute_drift": round(max_abs, 4),
    }
