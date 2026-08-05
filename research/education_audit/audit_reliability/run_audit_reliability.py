"""Master Runner for Audit Reliability & Counterfactual Meta-Evaluation Framework (AR-1 to AR-5)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from research.education_audit.audit_reliability.evaluator_panel import initialize_evaluator_panel
from research.education_audit.audit_reliability.counterfactual_evaluator_benchmark import run_counterfactual_evaluator_benchmark
from research.education_audit.audit_reliability.cross_domain_transfer import run_cross_domain_transfer_benchmark
from research.education_audit.audit_reliability.attribution_and_factuality_pilot import run_attribution_and_factuality_pilot


def run_full_audit_reliability_suite(out_dir: str = "results/education_audit/audit_reliability") -> Dict[str, Any]:
    """Runs full Audit Reliability Framework suite (AR-1 to AR-5) and generates master manifest."""
    os.makedirs(out_dir, exist_ok=True)

    print("Step 1: Initializing Evaluator Panel Architecture (AR-1)...")
    panel_res = initialize_evaluator_panel()

    print("Step 2: Running Large-Scale Counterfactual Evaluator Benchmark & Reliability Cards (AR-2)...")
    ar2_res = run_counterfactual_evaluator_benchmark(out_dir=out_dir)

    print("Step 3: Benchmarking Cross-Domain Evaluator Transfer (AR-3)...")
    ar3_res = run_cross_domain_transfer_benchmark(out_dir=out_dir)

    print("Step 4: Running Causal Attribution Bias & Factual Asymmetry Pilot (AR-5)...")
    ar5_res = run_attribution_and_factuality_pilot(out_dir=out_dir)

    manifest = {
        "status": "AUDIT_RELIABILITY_SUITE_COMPLETED",
        "ar1_panel_initialized": panel_res["evaluators_count"],
        "ar2_reliability_cards": ar2_res["reliability_cards"],
        "ar3_cross_domain_transfer": ar3_res,
        "ar5_attribution_and_factuality": ar5_res,
    }

    manifest_path = os.path.join(out_dir, "audit_reliability_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAudit Reliability Suite Execution Complete! Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    run_full_audit_reliability_suite()
