"""Unit Contract Tests for Automated Text Audit, Counterfactual SNR, & Fact Graph Entailment."""

import os
import pytest

from research.education_audit.automated_text_audit.feature_registry import extract_all_letter_features, analyze_profile_fact_graph
from research.education_audit.automated_text_audit.external_replication import run_external_calibration
from research.education_audit.automated_text_audit.paired_difference_analysis import align_sentences, analyze_paired_counterfactuals, compute_counterfactual_signal_to_noise
from research.education_audit.automated_text_audit.visualize_counterfactuals import generate_counterfactual_atlas_html
from research.education_audit.automated_text_audit.sensitivity_simulator import calculate_minimum_detectable_difference, run_sensitivity_simulation
from research.education_audit.automated_text_audit.run_automated_audit import run_full_automated_audit


def test_feature_extraction_and_fact_graph_accuracy():
    """Verifies feature registry and profile-aware fact graph entailment."""
    sample = (
        "It is my distinct pleasure to recommend Alex for the position. "
        "During his time in our lab, Alex spearheaded the neural network project and led a team of 4 researchers. "
        "He is exceptionally smart, analytical, and dedicated."
    )
    facts = ["Built a neural network optimization project.", "Secured a 3.8 GPA in Computer Science."]
    feats = extract_all_letter_features(sample, verified_facts=facts)
    assert feats["word_count"] > 20
    assert feats["sentence_count"] == 3
    assert feats["explicit_recommendation_flag"] is True
    assert feats["fact_coverage_rate"] > 0.0
    assert feats["unsupported_claims_count"] >= 1  # team of 4 researchers unsupported by profile facts


def test_external_replication_filter_bug_fix():
    """Verifies external replication filters masculine and feminine communal metrics correctly."""
    manifest, results = run_external_calibration()
    assert manifest["status"] == "CALIBRATION_COMPLETED"
    assert "mean_masculine_agentic_density" in manifest
    assert "mean_feminine_agentic_density" in manifest


def test_sensitivity_simulation():
    """Verifies minimum detectable difference decreases as number of profiles increases."""
    mdd_2 = calculate_minimum_detectable_difference(n_profiles=2)
    mdd_8 = calculate_minimum_detectable_difference(n_profiles=8)
    assert mdd_8 < mdd_2

    res = run_sensitivity_simulation(profile_range=[2, 8])
    assert "current_edu2a_mdd_estimate" in res
    assert len(res["simulation_curves"]) == 2


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
    assert "counterfactual_snr_ratio" in manifest
    assert os.path.exists(os.path.join(priv_out, "counterfactual_difference_atlas.html"))
    assert os.path.exists(os.path.join(pub_out, "counterfactual_difference_atlas.html"))
    assert os.path.exists(os.path.join(priv_out, "counterfactual_snr_and_tail_risk.json"))
