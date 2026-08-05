"""Design Sensitivity Simulator for Educational Counterfactual Audit.

Calculates minimum detectable paired difference curves vs. number of profiles,
distinguishing Optimistic Independent-Pair Approximations from Hierarchical Profile-Level Simulations.
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
    hierarchical_profile_icc: float = 0.30,
) -> Dict[str, float]:
    """Calculates Minimum Detectable Difference (MDD) for paired counterfactual design.

    Returns:
    - optimistic_independent_pair_mdd: Assumes N_pairs = n_profiles * n_prompts * n_seeds (Optimistic).
    - hierarchical_profile_level_mdd: Treats n_profiles as the primary independent replication unit (Conservative).
    """
    n_pairs_opt = max(1, n_profiles * n_prompts * n_seeds)

    z_alpha_half = 1.96
    z_beta = 0.8416

    var_diff = 2.0 * (sd_human ** 2) * (1.0 - pairing_correlation)
    adjusted_sd = math.sqrt(max(0.01, var_diff)) / math.sqrt(max(0.1, reviewer_reliability))

    # Optimistic Independent-Pair MDD
    mdd_opt = (z_alpha_half + z_beta) * adjusted_sd / math.sqrt(n_pairs_opt)

    # Hierarchical Profile-Level MDD (Design Effect DE = 1 + (n_prompts*n_seeds - 1) * ICC)
    n_sub = n_prompts * n_seeds
    deff = 1.0 + (n_sub - 1) * hierarchical_profile_icc
    effective_n_profiles = max(1.0, (n_profiles * n_sub) / deff)
    mdd_hierarchical = (z_alpha_half + z_beta) * adjusted_sd / math.sqrt(effective_n_profiles)

    return {
        "optimistic_independent_pair_mdd": round(mdd_opt, 4),
        "hierarchical_profile_level_mdd": round(mdd_hierarchical, 4),
    }


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
        mdd_res = calculate_minimum_detectable_difference(n_profiles=n_prof, sd_human=0.75, pairing_correlation=0.60)
        curves.append({
            "n_profiles": n_prof,
            "total_paired_tuples": n_prof * 2 * 3,
            "total_letters_5cond": n_prof * 2 * 3 * 5,
            "optimistic_independent_pair_mdd": mdd_res["optimistic_independent_pair_mdd"],
            "hierarchical_profile_level_mdd": mdd_res["hierarchical_profile_level_mdd"],
        })

    mdd_edu2a = calculate_minimum_detectable_difference(n_profiles=2)
    mdd_pilot = calculate_minimum_detectable_difference(n_profiles=8)

    res = {
        "status": "SIMULATION_COMPLETED",
        "description": "Design sensitivity analysis distinguishing Optimistic Independent-Pair Approximations from Hierarchical Profile-Level Simulations.",
        "current_edu2a_profile_count": 2,
        "current_edu2a_mdd_estimate": mdd_edu2a["hierarchical_profile_level_mdd"],
        "current_edu2a_optimistic_mdd": mdd_edu2a["optimistic_independent_pair_mdd"],
        "planned_full_pilot_profile_count": 8,
        "planned_full_pilot_mdd_estimate": mdd_pilot["hierarchical_profile_level_mdd"],
        "planned_full_pilot_optimistic_mdd": mdd_pilot["optimistic_independent_pair_mdd"],
        "simulation_curves": curves,
    }

    out_path = os.path.join(out_dir, "sensitivity_curves.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    return res
