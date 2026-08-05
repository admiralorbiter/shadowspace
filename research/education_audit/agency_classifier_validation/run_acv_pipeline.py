"""Master Runner for Agency Classifier Validation & Evaluator Invariance (Phases ACV-1, ACV-2, ACV-3)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from research.education_audit.agency_classifier_validation.labe_classifier_trainer import train_and_evaluate_labe_classifier
from research.education_audit.agency_classifier_validation.metric_disagreement_atlas import build_metric_disagreement_atlas
from research.education_audit.agency_classifier_validation.evaluator_invariance_benchmark import run_evaluator_invariance_benchmark


def run_full_acv_pipeline(out_dir: str = "results/education_audit/agency_classifier_validation") -> Dict[str, Any]:
    """Runs Phase ACV-1, ACV-2, and ACV-3 in sequence and outputs manifest."""
    os.makedirs(out_dir, exist_ok=True)

    print("Step 1: Running Phase ACV-1 (Training LABE Classifier & Locked-Test Evaluation)...")
    acv1_res, model_artifacts = train_and_evaluate_labe_classifier(out_dir=out_dir)

    print("Step 2: Running Phase ACV-2 (Building Metric Disagreement Atlas on 60 Wan Pairs)...")
    acv2_res = build_metric_disagreement_atlas(model_artifacts=model_artifacts, out_dir=out_dir)

    print("Step 3: Running Phase ACV-3 (Benchmarking Counterfactual Evaluator Invariance & Flips)...")
    acv3_res = run_evaluator_invariance_benchmark(model_artifacts=model_artifacts, out_dir=out_dir)

    manifest = {
        "status": "ACV_FULL_PIPELINE_COMPLETED",
        "phase_acv1_classifier": acv1_res,
        "phase_acv2_disagreement_atlas": acv2_res,
        "phase_acv3_invariance": acv3_res,
    }

    manifest_path = os.path.join(out_dir, "acv_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nACV Pipeline Execution Complete! Manifest written to: {manifest_path}")
    return manifest


if __name__ == "__main__":
    run_full_acv_pipeline()
