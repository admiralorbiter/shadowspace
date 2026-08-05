"""Unit Contract Tests for Automated Text Audit & Counterfactual Difference Atlas."""

import os
import pytest

from research.education_audit.automated_text_audit.feature_registry import extract_all_letter_features
from research.education_audit.automated_text_audit.external_replication import run_external_calibration
from research.education_audit.automated_text_audit.paired_difference_analysis import align_sentences, analyze_paired_counterfactuals
from research.education_audit.automated_text_audit.visualize_counterfactuals import generate_counterfactual_atlas_html
from research.education_audit.automated_text_audit.sensitivity_simulator import calculate_minimum_detectable_difference, run_sensitivity_simulation
from research.education_audit.automated_text_audit.run_automated_audit import run_full_automated_audit


def test_feature_extraction_accuracy():
    """Verifies feature registry extracts agentic, communal, ability, and structure metrics accurately."""
    sample = (
        "It is my distinct pleasure to recommend Alex for the position. "
        "During his time in our lab, Alex spearheaded the neural network project and led a team of 4 researchers. "
        "He is exceptionally smart, analytical, and dedicated."
    )
    feats = extract_all_letter_features(sample)
    assert feats["word_count"] > 20
    assert feats["sentence_count"] == 3
    assert feats["explicit_recommendation_flag"] is True
    assert feats["agentic_count"] >= 1  # spearheaded / led
    assert feats["ability_count"] >= 2  # smart / analytical
    assert feats["grindstone_count"] >= 1  # dedicated


def test_align_sentences_and_verbatim_overlap():
    """Verifies sentence alignment and verbatim sentence overlap rate."""
    t1 = "Alex is a strong researcher. He completed the neural net project. We strongly recommend him."
    t2 = "Sarah is a strong researcher. She completed the neural net project. We strongly recommend her."
    res = align_sentences(t1, t2)
    assert res["sentences_a_count"] == 3
    assert res["sentences_b_count"] == 3
    assert res["exact_matching_sentences_count"] == 0  # names/pronouns differ
    assert res["sentence_edit_distance"] == 3
    assert res["verbatim_sentence_overlap_rate"] == 0.0


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
    assert os.path.exists(os.path.join(priv_out, "counterfactual_difference_atlas.html"))
    assert os.path.exists(os.path.join(pub_out, "counterfactual_difference_atlas.html"))
    assert os.path.exists(os.path.join(priv_out, "paired_counterfactual_differences.csv"))
    assert os.path.exists(os.path.join(priv_out, "sensitivity_curves.json"))
    assert os.path.exists(os.path.join(pub_out, "external_replication_results.csv"))
