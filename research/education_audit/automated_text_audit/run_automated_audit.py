"""Master Runner for Automated Text Audit & Counterfactual Difference Atlas.

Orchestrates external benchmark calibration, fine-grained lexical feature extraction,
paired counterfactual divergence analysis, HTML Atlas generation, and sensitivity simulation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from research.education_audit.automated_text_audit.external_replication import run_external_calibration
from research.education_audit.automated_text_audit.paired_difference_analysis import analyze_paired_counterfactuals
from research.education_audit.automated_text_audit.visualize_counterfactuals import generate_counterfactual_atlas_html
from research.education_audit.automated_text_audit.sensitivity_simulator import run_sensitivity_simulation


def run_full_automated_audit(
    generations_file: str = "results/education_audit/edu_2a/generations.jsonl",
    private_out_dir: str = "private_analysis/automated_text_audit",
    public_out_dir: str = "results/education_audit/automated_text_audit",
) -> Dict[str, Any]:
    """Executes complete automated text audit workflow."""
    os.makedirs(private_out_dir, exist_ok=True)
    os.makedirs(public_out_dir, exist_ok=True)

    print("Step 1: Running External Benchmark Calibration...")
    ext_manifest, ext_results = run_external_calibration(out_dir=public_out_dir)

    print("Step 2: Extracting Features & Paired Differences on 60 Gemma Letters...")
    letter_features, paired_diffs = analyze_paired_counterfactuals(
        generations_file=generations_file,
        out_dir=private_out_dir,
    )

    print("Step 3: Generating HTML Counterfactual Difference Atlas...")
    html_path = generate_counterfactual_atlas_html(
        paired_diffs=paired_diffs,
        letter_features=letter_features,
        out_dir=private_out_dir,
    )

    print("Step 4: Running Design Sensitivity Simulation...")
    sim_res = run_sensitivity_simulation(out_dir=private_out_dir)

    # 5. Render Private Audit Report
    report_path = os.path.join(private_out_dir, "report.md")
    report_lines = [
        "# Phase EDU-2a Offline Analysis: Automated Text Audit Report\n",
        f"- **Generations Analyzed**: {len(letter_features)}",
        f"- **Counterfactual Pairs Evaluated**: {len(paired_diffs)}",
        f"- **HTML Difference Atlas Path**: `{html_path}`",
        f"- **Minimum Detectable Difference (EDU-2a 2 Profiles)**: {sim_res['current_edu2a_mdd_estimate']}",
        f"- **Minimum Detectable Difference (Planned 8 Profiles)**: {sim_res['planned_full_pilot_mdd_estimate']}\n",
        "## Summary of Findings\n",
        "1. **External Calibration**: Validated lexical feature extractors against reference-letter benchmarks.",
        "2. **Paired Divergence**: Quantified token edit distances, sentence alignments, and lexical deltas across fixed tuples.",
        "3. **HTML Atlas**: Generated interactive side-by-side pair explorer, difference heatmap, and tail-risk outlier plots.",
        "4. **Rater Protection**: All condition-linked outputs are safely stored in `private_analysis/automated_text_audit/` until manual reviews are locked.",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Export Public Calibration Registry Metadata
    feat_reg_manifest = {
        "status": "COMPLETED",
        "generations_count": len(letter_features),
        "counterfactual_pairs_count": len(paired_diffs),
        "calibration_status": ext_manifest.get("status"),
        "atlas_generated": True,
        "private_analysis_protected": True,
    }
    with open(os.path.join(public_out_dir, "feature_registry.json"), "w", encoding="utf-8") as f:
        json.dump(feat_reg_manifest, f, indent=2)

    print(f"\nAutomated Text Audit Complete! HTML Atlas: {html_path}")
    return feat_reg_manifest


if __name__ == "__main__":
    run_full_automated_audit()
