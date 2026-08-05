"""Two One-Sided Equivalence Testing (TOST) with Evaluator-Specific Margins."""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
from scipy import stats


def run_tost_equivalence_test(
    deltas: np.ndarray,
    evaluator_type: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Runs Two One-Sided Tests (TOST) against evaluator-specific scale bounds and relabels result."""
    N = len(deltas)
    if N < 2:
        return {"tost_passed": False}

    # Evaluator-specific margins
    if evaluator_type == "exact_lexicon":
        bound_delta = 2.0  # Terms per 100 words
    else:
        bound_delta = 0.02  # Probability points

    mean_d = float(np.mean(deltas))
    std_d = float(np.std(deltas, ddof=1))
    se_d = std_d / np.sqrt(N)

    t_lower = (mean_d - (-bound_delta)) / max(1e-9, se_d)
    p_lower = float(1.0 - stats.t.cdf(t_lower, df=N - 1))

    t_upper = (mean_d - bound_delta) / max(1e-9, se_d)
    p_upper = float(stats.t.cdf(t_upper, df=N - 1))

    p_tost = max(p_lower, p_upper)
    tost_passed = bool(p_tost < alpha)

    return {
        "evaluator_type": evaluator_type,
        "evaluator_specific_margin_delta": bound_delta,
        "mean_signed_drift": round(mean_d, 4),
        "standard_error": round(se_d, 4),
        "p_value_tost": round(p_tost, 4),
        "mean_signed_drift_equivalence_passed": tost_passed,
        "status_label": "Mean Signed Drift Equivalence Passed" if tost_passed else "Mean Signed Drift Equivalence Failed",
    }
