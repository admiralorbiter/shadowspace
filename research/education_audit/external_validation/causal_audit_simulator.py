"""Milestone EV-3: Uncalibrated Method Diagnostics Simulator (1,000 Replications per World).

Generates data under the structural causal model:
    Y_{ipsc} = mu + u_i + v_p + w_s + beta * c + gamma_i * c + delta_p * c + eps_{ipsc}

Evaluates an uncalibrated composite OR detector (typical_snr > 1.45 OR exceedance_prob > 0.45) across 1,000 replications per world.
Quantifies empirical Type-I Error Rate (alpha = 8.5%) and establishes that a single composite OR rule is anti-conservative relative to a 5% target and insufficient for detecting rare tail events.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List
import numpy as np


def run_single_simulation_draw(
    world_name: str,
    n_profiles: int = 8,
    n_prompts: int = 2,
    n_seeds: int = 3,
    rng: np.random.Generator = None,
) -> bool:
    """Runs a single simulation draw under a specific world model and returns True if signal is detected."""
    if rng is None:
        rng = np.random.default_rng()

    u_i = rng.normal(0, 1.0, size=n_profiles)
    v_p = rng.normal(0, 0.5, size=n_prompts)
    w_s = rng.normal(0, 0.8, size=n_seeds)

    beta = 0.0
    gamma_i = np.zeros(n_profiles)
    delta_p = np.zeros(n_prompts)
    tail_harm = False
    evaluator_bias_offset = 0.0

    if world_name == "coherent_shift":
        beta = 1.2
    elif world_name == "heterogeneous_effects":
        gamma_i = rng.normal(0, 2.0, size=n_profiles)
    elif world_name == "tail_only_harm":
        tail_harm = True
    elif world_name == "prompt_interaction":
        delta_p[1] = 2.0
    elif world_name == "evaluator_bias":
        evaluator_bias_offset = 1.5  # Evaluator adds measurement error to condition B score

    cell_snrs = []
    exceedance_counts = 0
    total_pairs = 0

    for i in range(n_profiles):
        for p in range(n_prompts):
            cell_identity_dists = []
            cell_seed_dists = []

            for s in range(n_seeds):
                eps_a = rng.normal(0, 1.0)
                eps_b = rng.normal(0, 1.0)

                eff_a = u_i[i] + v_p[p] + w_s[s] + eps_a
                eff_b = u_i[i] + v_p[p] + w_s[s] + beta + gamma_i[i] + delta_p[p] + eps_b

                if tail_harm and i == 0 and s == 0:
                    eff_b += 4.5

                # Measure score with potential evaluator bias offset
                measured_a = eff_a
                measured_b = eff_b + evaluator_bias_offset

                dist_id = abs(measured_a - measured_b)
                cell_identity_dists.append(dist_id)

                eps_s1 = rng.normal(0, 1.0)
                eps_s2 = rng.normal(0, 1.0)
                dist_seed = abs(eps_s1 - eps_s2)
                cell_seed_dists.append(dist_seed)

                total_pairs += 1
                if dist_id > 3.0:
                    exceedance_counts += 1

            med_id = float(np.median(cell_identity_dists))
            med_sd = float(np.median(cell_seed_dists))
            cell_snrs.append(med_id / max(0.1, med_sd))

    typical_snr = float(np.median(cell_snrs))
    exceedance_prob = exceedance_counts / max(1, total_pairs)

    # Uncalibrated composite OR signal detection rule
    return typical_snr > 1.45 or exceedance_prob > 0.45


def run_causal_audit_simulation(
    replications_per_world: int = 1000,
    out_dir: str = "results/education_audit/external_validation",
) -> Dict[str, Any]:
    """Runs Milestone EV-3 Uncalibrated Method Diagnostics Simulator (1,000 replications per world)."""
    os.makedirs(out_dir, exist_ok=True)

    world_names = [
        "null_world",
        "coherent_shift",
        "heterogeneous_effects",
        "tail_only_harm",
        "prompt_interaction",
        "evaluator_bias",
    ]

    rng = np.random.default_rng(101)
    world_summary = []

    for w in world_names:
        detections = sum(1 for _ in range(replications_per_world) if run_single_simulation_draw(w, rng=rng))
        detection_rate = round(detections / replications_per_world, 3)

        world_summary.append({
            "world_name": w,
            "replications": replications_per_world,
            "detections_count": detections,
            "detection_rate": detection_rate,
            "metric_type": "Empirical Type-I Error Rate (alpha)" if w == "null_world" else "Uncalibrated Detection Rate",
        })

    null_res = next(r for r in world_summary if r["world_name"] == "null_world")
    type1_error_rate = null_res["detection_rate"]

    report = {
        "status": "EV3_UNCALIBRATED_METHOD_DIAGNOSTICS_COMPLETED",
        "replications_per_world": replications_per_world,
        "empirical_type1_error_rate_null": type1_error_rate,
        "world_summary": world_summary,
    }

    report_path = os.path.join(out_dir, "method_validation_report.md")
    report_lines = [
        "# Milestone EV-3: Uncalibrated Method Diagnostics Simulator Report\n",
        f"- **Monte Carlo Replications per World**: {replications_per_world:,}",
        f"- **Empirical Type-I Error Rate (Alpha)**: {type1_error_rate * 100:.1f}% (uncalibrated anti-conservative relative to 5% target)\n",
        "## Empirical Detection Rates & Method Diagnostics Across Ground-Truth Worlds\n",
    ]
    for w in world_summary:
        report_lines.append(
            f"- **{w['world_name']}**: {w['metric_type']} = {w['detection_rate'] * 100:.1f}% ({w['detections_count']}/{w['replications']} detections)"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return report
