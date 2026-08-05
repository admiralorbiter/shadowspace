"""Master Runner for Audit Reliability & Counterfactual Meta-Evaluation Framework Scaffold (AR-1 to AR-4)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from research.education_audit.audit_reliability.evaluator_panel import initialize_evaluator_panel
from research.education_audit.audit_reliability.counterfactual_evaluator_benchmark import run_counterfactual_evaluator_benchmark
from research.education_audit.audit_reliability.cross_domain_transfer import run_cross_domain_transfer_benchmark
from research.education_audit.audit_reliability.attribution_and_factuality_pilot import run_attribution_and_factuality_pilot


def run_full_audit_reliability_suite(out_dir: str = "results/education_audit/audit_reliability") -> Dict[str, Any]:
    """Runs full Audit Reliability Framework prototype scaffold (AR-1 to AR-4) and generates master manifest."""
    os.makedirs(out_dir, exist_ok=True)

    print("Step 1: Initializing Evaluator Panel Architecture (AR-1)...")
    panel_res = initialize_evaluator_panel()

    print("Step 2: Running Controlled 16-Pair Counterfactual Evaluator Smoke Test (AR-2)...")
    ar2_res = run_counterfactual_evaluator_benchmark(out_dir=out_dir)

    print("Step 3: Running Cross-Context Score Agreement Contrast (AR-3)...")
    ar3_res = run_cross_domain_transfer_benchmark(out_dir=out_dir)

    print("Step 4: Running Synthetic Known-Answer Attribution & Coverage Fixture (AR-4)...")
    ar4_res = run_attribution_and_factuality_pilot(out_dir=out_dir)

    manifest = {
        "status": "AUDIT_RELIABILITY_FRAMEWORK_SCAFFOLD_VALIDATED",
        "ar1_panel_evaluators_count": panel_res["evaluators_count"],
        "ar1_independent_evaluators_count": panel_res["independent_evaluators_count"],
        "ar2_smoke_test": ar2_res["reliability_cards"],
        "ar3_cross_context_contrast": ar3_res,
        "ar4_synthetic_attribution_fixture": ar4_res,
    }

    manifest_path = os.path.join(out_dir, "audit_reliability_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAudit Reliability Framework Scaffold Complete! Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    run_full_audit_reliability_suite()
