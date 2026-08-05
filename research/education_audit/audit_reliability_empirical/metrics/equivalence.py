"""Two One-Sided Equivalence Testing (TOST) against bound delta."""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
from scipy import stats


def run_tost_equivalence_test(
    deltas: np.ndarray,
    bound_delta: float = 0.02,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Runs Two One-Sided Tests (TOST) of equivalence for paired differences under bound delta."""
    N = len(deltas)
    if N < 2:
        return {"tost_passed": False}

    mean_d = float(np.mean(deltas))
    std_d = float(np.std(deltas, ddof=1))
    se_d = std_d / np.sqrt(N)

    # Lower bound test: H0_1: mu <= -bound_delta vs H1_1: mu > -bound_delta
    t_lower = (mean_d - (-bound_delta)) / max(1e-9, se_d)
    p_lower = float(1.0 - stats.t.cdf(t_lower, df=N - 1))

    # Upper bound test: H0_2: mu >= bound_delta vs H1_2: mu < bound_delta
    t_upper = (mean_d - bound_delta) / max(1e-9, se_d)
    p_upper = float(stats.t.cdf(t_upper, df=N - 1))

    p_tost = max(p_lower, p_upper)
    tost_passed = bool(p_tost < alpha)

    return {
        "equivalence_bound_delta": bound_delta,
        "mean_difference": round(mean_d, 4),
        "standard_error": round(se_d, 4),
        "p_value_lower_bound": round(p_lower, 4),
        "p_value_upper_bound": round(p_upper, 4),
        "p_value_tost": round(p_tost, 4),
        "tost_equivalence_passed": tost_passed,
    }
