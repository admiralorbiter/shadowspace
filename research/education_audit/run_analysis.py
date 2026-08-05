"""Phase EDU-1 Analysis & Manifest Generator."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
from research.holonomy.experiments.run_phase_e0_summary import get_git_commit_sha


def run_edu1_analysis(data_dir: str = "results/education_audit/edu_1") -> Dict[str, Any]:
    """Analyzes EDU-1 evaluation records and exports case & analysis manifests."""
    eval_file = os.path.join(data_dir, "evaluations.jsonl")
    records = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Aggregate scores by condition and prompt
    cond_scores: Dict[str, List[float]] = {}
    cond_hallucinations: Dict[str, List[float]] = {}
    prompt_cond_scores: Dict[str, Dict[str, List[float]]] = {}

    for r in records:
        c = r["condition"]
        p = r["prompt_id"]
        s = r["recommendation_strength_score"]
        h = r["hallucinations_per_100_words"]

        cond_scores.setdefault(c, []).append(s)
        cond_hallucinations.setdefault(c, []).append(h)

        prompt_cond_scores.setdefault(p, {}).setdefault(c, []).append(s)

    mean_scores = {c: float(np.mean(vals)) for c, vals in cond_scores.items()}
    mean_hallucinations = {c: float(np.mean(vals)) for c, vals in cond_hallucinations.items()}

    # Verify planted signals
    masc_structured_score = np.mean(prompt_cond_scores.get("structured_prompt", {}).get("pronoun_masc", [0]))
    fem_structured_score = np.mean(prompt_cond_scores.get("structured_prompt", {}).get("pronoun_fem", [0]))
    anon_hallucinations = mean_hallucinations.get("anonymous", 1.0)
    fem_min_hallucinations = mean_hallucinations.get("pronoun_fem", 0.0)

    signal_recovered = bool((masc_structured_score > fem_structured_score + 0.3) and (fem_min_hallucinations > 0.0))
    null_verified = bool(anon_hallucinations == 0.0)

    mock_status = "PASSED" if (signal_recovered and null_verified) else "FAILED"


    summary = {
        "total_generations_evaluated": len(records),
        "total_cases_evaluated": 8,
        "identity_conditions_evaluated": ["anonymous", "pronoun_masc", "pronoun_fem", "name_masc", "name_fem"],
        "mean_recommendation_strength_by_condition": mean_scores,
        "mean_hallucinations_per_100_words_by_condition": mean_hallucinations,
        "planted_signal_recovered": signal_recovered,
        "planted_null_verified": null_verified,
        "mock_validation_status": mock_status,
        "status_labels": {
            "execution_status": "COMPLETED",
            "mock_validation_status": "PLANTED_SIGNAL_RECOVERED" if signal_recovered else "FAILED",
            "null_verification_status": "PLANTED_NULL_VERIFIED" if null_verified else "FAILED",
        },
    }

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": "edu_1_mock_validation",
        "phase": "EDU-1",
        "git_commit_sha": get_git_commit_sha(),
        "execution_status": "COMPLETED",
        "mock_validation_status": mock_status,
        "summary": summary,
    }

    manifest_path = os.path.join(data_dir, "analysis_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    case_manifest_path = os.path.join(data_dir, "case_manifest.json")
    with open(case_manifest_path, "w", encoding="utf-8") as f:
        json.dump({"total_cases": 8, "domains": ["technology", "math_data", "humanities", "leadership"]}, f, indent=2)

    print(f"\n================================================================================")
    print(f"EDU-1 MOCK PLANTED-BIAS VALIDATION REPORT:")
    print(f"================================================================================")
    print(f"    - Total Generations Evaluated: {len(records)}")
    print(f"    - Planted Signal Recovered: {signal_recovered}")
    print(f"    - Planted Null Verified: {null_verified}")
    print(f"    - Mean Strength (Anonymous): {mean_scores.get('anonymous', 0):.2f}")
    print(f"    - Mean Strength (Masculine Pronoun): {mean_scores.get('pronoun_masc', 0):.2f}")
    print(f"    - Mean Strength (Feminine Pronoun): {mean_scores.get('pronoun_fem', 0):.2f}")
    print(f"    - Mock Validation Status: {mock_status}")
    print(f"================================================================================")
    print(f"Manifest exported to: {manifest_path}")

    return manifest


if __name__ == "__main__":
    run_edu1_analysis()
