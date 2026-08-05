"""Master Runner for Automated Text Audit & Counterfactual Difference Atlas.

Orchestrates external benchmark calibration, fine-grained lexical feature extraction,
paired counterfactual divergence analysis, Matched SNR v2, HTML Atlas generation, and sensitivity simulation.
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

    snr_tail_file = os.path.join(private_out_dir, "counterfactual_snr_and_tail_risk.json")
    snr_metrics = {}
    if os.path.exists(snr_tail_file):
        with open(snr_tail_file, "r", encoding="utf-8") as f:
            snr_metrics = json.load(f)

    print("Step 3: Generating Private Study HTML Counterfactual Difference Atlas...")
    private_html_path = generate_counterfactual_atlas_html(
        paired_diffs=paired_diffs,
        letter_features=letter_features,
        out_dir=private_out_dir,
        snr_metrics=snr_metrics,
    )

    print("Step 4: Generating Public 100% Synthetic Demonstration HTML Atlas for UI Review...")
    synthetic_snr = {
        "overall_median_identity_perturbation": 2.0,
        "overall_median_seed_sampling_noise": 3.0,
        "matched_counterfactual_snr_ratio": 0.667,
        "log_snr_ratio": -0.405,
        "displacement_coherence_kappa": 0.124,
        "coherence_permutation_p_value": 0.452,
        "seed_null_exceedance_probability": 0.042,
        "snr_interpretation": "100% Synthetic Demonstration Only (Public UI Review)",
        "seed_null_exceedance_prob_gt_q95": 0.042,
        "quantile_effect_q90": 3.0,
        "cvar_90_worst_tail_average": 3.5,
        "directional_consistency_rate": 0.667,
    }

    synth_paired = []
    for idx in range(1, 25):
        is_p = idx <= 8
        synth_paired.append({
            "pair_id": f"DEMO_PAIR_{idx:03d}",
            "pair_label": "Pronoun: Masculine vs. Feminine (Demo)" if is_p else "Feminine vs. Anonymous (Demo)",
            "is_primary": is_p,
            "case_id": "DEMO_CASE_01",
            "case_label": "Humanities / Exceptional (Demo)",
            "prompt_id": "minimal_prompt",
            "prompt_label": "Minimal Prompt",
            "seed": 101,
            "condition_a": "pronoun_masc",
            "condition_b": "pronoun_fem",
            "gen_id_a": f"demo_gen_a_{idx}",
            "gen_id_b": f"demo_gen_b_{idx}",
            "signed_word_count_diff": 4 if idx % 2 == 0 else -6,
            "abs_word_count_diff": 6,
            "token_edit_distance": 12,
            "sentence_edit_distance": 2 if idx % 2 == 0 else 4,
            "verbatim_sentence_overlap_rate": 66.7,
            "signed_agentic_density_diff": 0.45 if idx % 2 == 0 else -0.30,
            "signed_communal_density_diff": -0.20 if idx % 2 == 0 else 0.40,
            "signed_warmth_density_diff": -0.15,
            "signed_leadership_density_diff": 0.35,
            "specificity_screening_flag_a": False,
            "specificity_screening_flag_b": False,
            "surfaced_reasons": ["Word-count difference: +4 words", "Leadership density difference: +0.35 per 100 words"],
        })

    synth_feats = [
        {
            "generation_id": f"demo_gen_a_{idx}",
            "case_id": "DEMO_CASE_01",
            "case_label": "Humanities / Exceptional (Demo)",
            "condition": "pronoun_masc",
            "prompt_id": "minimal_prompt",
            "prompt_label": "Minimal Prompt",
            "word_count": 210,
            "sentence_count": 8,
            "explicit_recommendation_flag": True,
            "specificity_screening_flag": False,
            "output_text": "Synthetic Demonstration Letter A prose for UI feedback inspection.",
        }
        for idx in range(1, 25)
    ]

    public_html_path = generate_counterfactual_atlas_html(
        paired_diffs=synth_paired,
        letter_features=synth_feats,
        out_dir=public_out_dir,
        snr_metrics=synthetic_snr,
    )

    print("Step 5: Running Design Sensitivity Simulation...")
    sim_res = run_sensitivity_simulation(out_dir=private_out_dir)

    # Render Private Audit Report
    report_path = os.path.join(private_out_dir, "report.md")
    report_lines = [
        "# Phase EDU-2a Offline Analysis: Automated Text Audit & Signal-to-Noise Report\n",
        f"- **Generations Analyzed**: {len(letter_features)}",
        f"- **Total Counterfactual Pairs Evaluated**: {len(paired_diffs)}",
        f"- **Primary Gender Comparisons**: {len([p for p in paired_diffs if p.get('is_primary')])}",
        f"- **Secondary Anonymous Baselines**: {len([p for p in paired_diffs if not p.get('is_primary')])}",
        f"- **Matched Counterfactual SNR Ratio (R)**: {snr_metrics.get('matched_counterfactual_snr_ratio')}",
        f"- **Log SNR Ratio (L)**: {snr_metrics.get('log_snr_ratio')}",
        f"- **Displacement Coherence (&kappa;)**: {snr_metrics.get('displacement_coherence_kappa')} (p = {snr_metrics.get('coherence_permutation_p_value')})",
        f"- **Seed-Null Exceedance Probability**: {snr_metrics.get('seed_null_exceedance_probability')}",
        f"- **SNR Baseline Interpretation**: {snr_metrics.get('snr_interpretation')}",
        f"- **Private Study HTML Atlas Path**: `{private_html_path}`",
        f"- **Public Synthetic Atlas Path**: `{public_html_path}`",
        f"- **Minimum Detectable Difference (Optimistic Independent-Pair)**: {sim_res['current_edu2a_optimistic_mdd']}",
        f"- **Minimum Detectable Difference (Hierarchical Profile-Level)**: {sim_res['current_edu2a_mdd_estimate']}\n",
        "## Summary of Findings\n",
        "1. **Matched Cell-Level SNR v2**: Calculated R_j = D_identity / D_seed per matched cell (profile, prompt, identity channel) with Log SNR L_j.",
        "2. **Counterfactual Displacement Geometry**: Measured multi-feature displacement directionality coherence (kappa) and sign-flip permutation p-value.",
        "3. **Seed-Null Exceedance**: Measured P(|D_identity| > Q_0.95(D_seed)) comparing identity perturbation against the 95th percentile of ordinary seed noise.",
        "4. **Profile-Aware Lexical Fact-Coverage Screen**: Evaluated profile fact coverage and unsupported specificity claims.",
        "5. **Rater Protection**: Real study atlas & real SNR are protected in `private_analysis/automated_text_audit/`; 100% synthetic demo is published in `results/`.",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    feat_reg_manifest = {
        "status": "COMPLETED",
        "generations_count": len(letter_features),
        "total_pairs_count": len(paired_diffs),
        "primary_gender_pairs_count": len([p for p in paired_diffs if p.get("is_primary")]),
        "secondary_anonymous_pairs_count": len([p for p in paired_diffs if not p.get("is_primary")]),
        "matched_counterfactual_snr_ratio": snr_metrics.get("matched_counterfactual_snr_ratio"),
        "log_snr_ratio": snr_metrics.get("log_snr_ratio"),
        "displacement_coherence_kappa": snr_metrics.get("displacement_coherence_kappa"),
        "snr_interpretation": snr_metrics.get("snr_interpretation"),
        "calibration_status": ext_manifest.get("status"),
        "public_synthetic_atlas_generated": True,
        "private_study_atlas_protected": True,
    }
    with open(os.path.join(public_out_dir, "feature_registry.json"), "w", encoding="utf-8") as f:
        json.dump(feat_reg_manifest, f, indent=2)

    print(f"\nAutomated Text Audit Complete! Public Synthetic Atlas: {public_html_path}")
    return feat_reg_manifest


if __name__ == "__main__":
    run_full_automated_audit()
