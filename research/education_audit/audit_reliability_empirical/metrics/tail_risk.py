"""Tail Risk Metrics: Quantile Q_.95 and Conditional Value-at-Risk (CVaR_.95)."""

from __future__ import annotations

from typing import Any, Dict
import numpy as np


def compute_tail_risk_metrics(
    deltas: np.ndarray,
    quantile_q: float = 0.95,
) -> Dict[str, Any]:
    """Computes Quantile Q_.95 and Conditional Value-at-Risk (CVaR_.95 / Expected Shortfall)."""
    abs_deltas = np.abs(deltas)
    if len(abs_deltas) == 0:
        return {}

    q_val = float(np.percentile(abs_deltas, 100 * quantile_q))
    tail_subset = abs_deltas[abs_deltas >= q_val]
    cvar_val = float(np.mean(tail_subset)) if len(tail_subset) > 0 else q_val

    return {
        "quantile_level": quantile_q,
        "quantile_q95": round(q_val, 4),
        "cvar_95_tail_risk": round(cvar_val, 4),
        "tail_excess_ratio": round(cvar_val / max(1e-6, np.mean(abs_deltas)), 3),
    }
