"""Master Runner for External Evidence & Synthetic Validation Suite (Milestones EV-1, EV-2, EV-3).

Executes:
1. Milestone EV-1: External Benchmark Agency Metric Replication & Disagreement Atlas.
2. Milestone EV-2: Auditor Counterfactual Invariance Benchmark across Demographic Axes.
3. Milestone EV-3: Synthetic Causal Audit Simulator across 6 Ground-Truth Worlds.
4. O*NET 30.3 Profile Bank Construction.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from research.education_audit.external_validation.agency_metric_replication import run_agency_metric_replication
from research.education_audit.external_validation.evaluator_invariance import run_evaluator_invariance_benchmark
from research.education_audit.external_validation.causal_audit_simulator import run_causal_audit_simulation
from research.education_audit.external_validation.onet_profile_builder import generate_onet_grounded_profile_bank


def run_full_external_validation(out_dir: str = "results/education_audit/external_validation") -> Dict[str, Any]:
    """Orchestrates complete external evidence and synthetic validation program."""
    os.makedirs(out_dir, exist_ok=True)

    print("Step 1: Running Milestone EV-1 Real Reference-Letter Replication...")
    ev1_res = run_agency_metric_replication(out_dir=out_dir)

    print("Step 2: Running Milestone EV-2 Auditor Counterfactual Invariance Benchmark...")
    ev2_res = run_evaluator_invariance_benchmark(out_dir=out_dir)

    print("Step 3: Running Milestone EV-3 Synthetic Causal Audit Laboratory...")
    ev3_res = run_causal_audit_simulation(out_dir=out_dir)

    print("Step 4: Building O*NET 30.3 Grounded Synthetic Profile Bank...")
    onet_res = generate_onet_grounded_profile_bank(out_dir=out_dir)

    manifest = {
        "status": "EXTERNAL_VALIDATION_COMPLETED",
        "ev1_replication": ev1_res,
        "ev2_invariance": ev2_res,
        "ev3_simulation": ev3_res,
        "onet_profiles": onet_res,
    }

    manifest_path = os.path.join(out_dir, "validation_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nExternal Evidence & Synthetic Validation Suite Complete! Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    run_full_external_validation()
