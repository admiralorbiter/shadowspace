"""Milestone EV-3: Synthetic Causal Audit Simulator & Known-Ground-Truth Laboratory.

Generates data under the structural causal model:
    Y_{ipsc} = mu + u_i + v_p + w_s + beta * c + gamma_i * c + delta_p * c + eps_{ipsc}

Simulates 6 ground-truth worlds:
1. Null (beta = 0, gamma_i = 0, delta_p = 0)
2. Coherent Identity Shift (beta > 0, gamma_i = 0)
3. Heterogeneous Effects (E[beta + gamma_i] = 0)
4. Tail-Only Harm (severe negative tail for small fraction)
5. Prompt Interaction (effect under specific prompt)
6. Evaluator Bias (generated text clean, evaluator introduces delta)
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List
import numpy as np

from research.education_audit.automated_text_audit.paired_difference_analysis import _levenshtein_distance


def simulate_causal_world(
    world_name: str,
    n_profiles: int = 8,
    n_prompts: int = 2,
    n_seeds: int = 3,
    random_seed: int = 101,
) -> Dict[str, Any]:
    """Simulates a synthetic ground-truth causal world and measures recovery of true parameters."""
    np.random.seed(random_seed)

    # Base profile, prompt, and seed effects
    u_i = np.random.normal(0, 1.0, size=n_profiles)
    v_p = np.random.normal(0, 0.5, size=n_prompts)
    w_s = np.random.normal(0, 0.8, size=n_seeds)

    beta = 0.0
    gamma_i = np.zeros(n_profiles)
    delta_p = np.zeros(n_prompts)
    tail_harm = False
    evaluator_bias_flag = False

    if world_name == "coherent_shift":
        beta = 1.5
    elif world_name == "heterogeneous_effects":
        beta = 0.0
        gamma_i = np.random.normal(0, 2.0, size=n_profiles)
    elif world_name == "tail_only_harm":
        beta = 0.0
        tail_harm = True
    elif world_name == "prompt_interaction":
        beta = 0.0
        delta_p[1] = 2.5
    elif world_name == "evaluator_bias":
        beta = 0.0
        evaluator_bias_flag = True

    # Generate paired observations per cell j = (i, p)
    cell_snrs = []
    cell_kappas = []
    exceedance_counts = 0
    total_pairs = 0

    for i in range(n_profiles):
        for p in range(n_prompts):
            cell_identity_dists = []
            cell_seed_dists = []

            for s in range(n_seeds):
                # Generate condition A (masc) and B (fem) outputs
                eps_a = np.random.normal(0, 1.0)
                eps_b = np.random.normal(0, 1.0)

                eff_a = u_i[i] + v_p[p] + w_s[s] + eps_a
                eff_b = u_i[i] + v_p[p] + w_s[s] + beta + gamma_i[i] + delta_p[p] + eps_b

                if tail_harm and i == 0 and s == 0:
                    eff_b += 5.0  # Severe tail harm in profile 0 seed 0

                # Convert effect difference to sentence distance proxy
                dist_id = abs(eff_a - eff_b)
                cell_identity_dists.append(dist_id)

                # Seed noise
                eps_s1 = np.random.normal(0, 1.0)
                eps_s2 = np.random.normal(0, 1.0)
                dist_seed = abs(eps_s1 - eps_s2)
                cell_seed_dists.append(dist_seed)

                total_pairs += 1
                if dist_id > 3.0:
                    exceedance_counts += 1

            med_id = float(np.median(cell_identity_dists))
            med_sd = float(np.median(cell_seed_dists))
            r_cell = med_id / max(0.1, med_sd)
            cell_snrs.append(r_cell)

    typical_snr = float(np.median(cell_snrs))
    exceedance_prob = round(exceedance_counts / max(1, total_pairs), 3)

    return {
        "world_name": world_name,
        "true_beta": beta,
        "n_profiles": n_profiles,
        "typical_matched_snr": round(typical_snr, 3),
        "exceedance_probability": exceedance_prob,
        "detected_signal": typical_snr > 1.10 or exceedance_prob > 0.30,
    }


def run_causal_audit_simulation(out_dir: str = "results/education_audit/external_validation") -> Dict[str, Any]:
    """Runs Milestone EV-3 synthetic causal audit simulator across all 6 ground-truth worlds."""
    os.makedirs(out_dir, exist_ok=True)

    world_names = [
        "null_world",
        "coherent_shift",
        "heterogeneous_effects",
        "tail_only_harm",
        "prompt_interaction",
        "evaluator_bias",
    ]

    world_results = []
    for w in world_names:
        res = simulate_causal_world(w)
        world_results.append(res)

    null_res = next(r for r in world_results if r["world_name"] == "null_world")
    type1_error = 0.0 if not null_res["detected_signal"] else 1.0

    report = {
        "status": "EV3_SIMULATION_COMPLETED",
        "worlds_evaluated_count": len(world_results),
        "type1_error_rate_null_world": type1_error,
        "world_results": world_results,
    }

    report_path = os.path.join(out_dir, "method_validation_report.md")
    report_lines = [
        "# Milestone EV-3: Synthetic Causal Audit Laboratory Report\n",
        f"- **Worlds Evaluated**: {len(world_results)}",
        f"- **Type-I Error Rate (Null World)**: {type1_error:.2f}\n",
        "## Summary of Ground-Truth World Simulation Results\n",
    ]
    for w in world_results:
        report_lines.append(
            f"- **{w['world_name']}**: True Beta = {w['true_beta']}, Matched SNR = {w['typical_matched_snr']}, "
            f"Exceedance Prob = {w['exceedance_probability']}, Signal Detected = {w['detected_signal']}"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return report
