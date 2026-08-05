"""Design Sensitivity Simulator for Educational Counterfactual Audit.

Calculates minimum detectable paired difference curves vs. number of profiles,
accounting for profile-level variance, seed variance, pairing correlation, and reviewer reliability.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List


def calculate_minimum_detectable_difference(
    n_profiles: int,
    n_prompts: int = 2,
    n_seeds: int = 3,
    sd_human: float = 0.75,
    pairing_correlation: float = 0.60,
    reviewer_reliability: float = 0.85,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Calculates Minimum Detectable Difference (MDD) for paired counterfactual design.

    Formula:
      N_pairs = n_profiles * n_prompts * n_seeds
      sigma_diff = sqrt(2 * (sd_human^2) * (1 - pairing_correlation)) / sqrt(reviewer_reliability)
      MDD = (z_alpha/2 + z_beta) * sigma_diff / sqrt(N_pairs)
    """
    n_pairs = max(1, n_profiles * n_prompts * n_seeds)

    # Standard normal critical values for two-sided alpha=0.05, power=0.80
    z_alpha_half = 1.96
    z_beta = 0.8416

    var_diff = 2.0 * (sd_human ** 2) * (1.0 - pairing_correlation)
    adjusted_sd = math.sqrt(max(0.01, var_diff)) / math.sqrt(max(0.1, reviewer_reliability))

    mdd = (z_alpha_half + z_beta) * adjusted_sd / math.sqrt(n_pairs)
    return round(mdd, 4)


def run_sensitivity_simulation(
    profile_range: List[int] = None,
    out_dir: str = "private_analysis/automated_text_audit",
) -> Dict[str, Any]:
    """Runs sensitivity simulation across varying numbers of profiles and exports curves."""
    os.makedirs(out_dir, exist_ok=True)

    if profile_range is None:
        profile_range = [2, 4, 8, 12, 16, 24, 32, 48, 64]

    curves: List[Dict[str, Any]] = []

    for n_prof in profile_range:
        # Base scenario (EDU-2a: 2 profiles, 2 prompts, 3 seeds = 12 tuples)
        mdd_base = calculate_minimum_detectable_difference(n_profiles=n_prof, sd_human=0.75, pairing_correlation=0.60)
        # High variance scenario
        mdd_high_var = calculate_minimum_detectable_difference(n_profiles=n_prof, sd_human=1.00, pairing_correlation=0.40)
        # High correlation scenario (strong paired control)
        mdd_high_corr = calculate_minimum_detectable_difference(n_profiles=n_prof, sd_human=0.75, pairing_correlation=0.80)

        curves.append({
            "n_profiles": n_prof,
            "total_paired_tuples": n_prof * 2 * 3,
            "total_letters_5cond": n_prof * 2 * 3 * 5,
            "mdd_base_scenario": mdd_base,
            "mdd_high_var_scenario": mdd_high_var,
            "mdd_high_corr_scenario": mdd_high_corr,
        })

    res = {
        "status": "SIMULATION_COMPLETED",
        "description": "Design sensitivity analysis for minimum detectable paired counterfactual effect sizes.",
        "current_edu2a_profile_count": 2,
        "current_edu2a_mdd_estimate": calculate_minimum_detectable_difference(n_profiles=2),
        "planned_full_pilot_profile_count": 8,
        "planned_full_pilot_mdd_estimate": calculate_minimum_detectable_difference(n_profiles=8),
        "simulation_curves": curves,
    }

    out_path = os.path.join(out_dir, "sensitivity_curves.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    return res
