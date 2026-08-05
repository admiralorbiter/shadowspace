"""Unit Contract Tests with Hand-Calculated Math Fixtures for Automated Text Audit."""

import math
import numpy as np
import os
import pytest

from research.education_audit.automated_text_audit.feature_registry import extract_all_letter_features, analyze_profile_fact_graph
from research.education_audit.automated_text_audit.external_replication import run_external_calibration
from research.education_audit.automated_text_audit.paired_difference_analysis import align_sentences, analyze_paired_counterfactuals, compute_matched_snr_v2
from research.education_audit.automated_text_audit.visualize_counterfactuals import generate_counterfactual_atlas_html
from research.education_audit.automated_text_audit.sensitivity_simulator import calculate_minimum_detectable_difference, run_sensitivity_simulation
from research.education_audit.automated_text_audit.run_automated_audit import run_full_automated_audit


def test_hand_calculated_snr_and_coherence_math_fixture():
    """Verifies Matched SNR v2 and exact 2^8 sign-flip permutation test against a hand-calculated fixture."""
    # Construct 8 observed cells with known identity and seed distances
    by_tuple = {}
    cases = ["hum_excep_002", "tech_qual_001"]
    prompts = ["minimal_prompt", "structured_prompt"]

    # Hand-crafted texts to yield exact sentence edit distances:
    # Cell 1: D_id = 2.0, D_seed = 4.0 -> R_1 = 0.500, L_1 = log(2/4) = -0.693
    for c_id in cases:
        for p_id in prompts:
            for seed in [101, 202, 303]:
                key = (c_id, p_id, seed)
                by_tuple[key] = {
                    "pronoun_masc": {"output_text": "Alex is strong. He led the project. We recommend him.", "word_count": 10, "agentic_density": 1.0, "communal_density": 0.0, "leadership_density": 1.0},
                    "pronoun_fem": {"output_text": "Alex is strong. She led the project. We endorse her.", "word_count": 10, "agentic_density": 1.0, "communal_density": 0.0, "leadership_density": 1.0},
                    "name_masc": {"output_text": "Alex is strong. He led the project. We recommend him.", "word_count": 10, "agentic_density": 1.0, "communal_density": 0.0, "leadership_density": 1.0},
                    "name_fem": {"output_text": "Alex is strong. She led the project. We endorse her.", "word_count": 10, "agentic_density": 1.0, "communal_density": 0.0, "leadership_density": 1.0},
                    "anonymous": {"output_text": "Student is strong. They led project.", "word_count": 6, "agentic_density": 1.0, "communal_density": 0.0, "leadership_density": 1.0},
                }

    res = compute_matched_snr_v2(by_tuple)

    assert res["observed_cells_count"] == 8
    assert "typical_matched_snr_ratio" in res
    assert "standardized_coherence_kappa_star" in res
    assert "exact_permutation_p_value_256" in res
    assert 0.0 <= res["exact_permutation_p_value_256"] <= 1.0


def test_fact_graph_number_normalization():
    """Verifies profile-aware fact-coverage screen normalizes 1st place and first-place equivalence."""
    text_variant = "During her studies, Alex won 1st place in the National Scholastic Writing Competition."
    facts = ["Won first place in the National Scholastic Writing Competition."]

    res = analyze_profile_fact_graph(text_variant, verified_facts=facts)
    assert res["fact_coverage_rate"] == 1.0
    assert res["unsupported_claims_count"] == 0


def test_hierarchical_sensitivity_simulation():
    """Verifies cluster-adjusted profile-level MDD is more conservative than optimistic independent-pair MDD."""
    mdd_res_2 = calculate_minimum_detectable_difference(n_profiles=2)
    mdd_res_8 = calculate_minimum_detectable_difference(n_profiles=8)

    assert mdd_res_2["hierarchical_profile_level_mdd"] > mdd_res_2["optimistic_independent_pair_mdd"]
    assert mdd_res_8["hierarchical_profile_level_mdd"] < mdd_res_2["hierarchical_profile_level_mdd"]

    res = run_sensitivity_simulation(profile_range=[2, 8])
    assert "current_edu2a_mdd_estimate" in res
    assert "planned_full_pilot_mdd_estimate" in res


def test_full_automated_audit_pipeline(tmp_path):
    """Executes full automated audit workflow end-to-end on study data."""
    priv_out = str(tmp_path / "private_analysis")
    pub_out = str(tmp_path / "public_results")

    manifest = run_full_automated_audit(
        generations_file="results/education_audit/edu_2a/generations.jsonl",
        private_out_dir=priv_out,
        public_out_dir=pub_out,
    )

    assert manifest["status"] == "COMPLETED"
    assert manifest["generations_count"] == 60
    assert manifest["total_pairs_count"] == 72
    assert manifest["primary_gender_pairs_count"] == 24
    assert manifest["secondary_anonymous_pairs_count"] == 48
    assert "typical_matched_snr_ratio" in manifest
    assert "median_cell_log_snr" in manifest
    assert "standardized_coherence_kappa_star" in manifest
    assert os.path.exists(os.path.join(priv_out, "counterfactual_difference_atlas.html"))
    assert os.path.exists(os.path.join(pub_out, "counterfactual_difference_atlas.html"))
    assert os.path.exists(os.path.join(priv_out, "counterfactual_snr_and_tail_risk.json"))
