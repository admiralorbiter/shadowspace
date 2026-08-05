"""Unit Contract Tests for Agency Classifier Validation & Evaluator Invariance Suite (ACV-1, ACV-2, ACV-3)."""

import os
import pytest

from research.education_audit.agency_classifier_validation.labe_classifier_trainer import train_and_evaluate_labe_classifier
from research.education_audit.agency_classifier_validation.metric_disagreement_atlas import build_metric_disagreement_atlas
from research.education_audit.agency_classifier_validation.evaluator_invariance_benchmark import run_evaluator_invariance_benchmark
from research.education_audit.agency_classifier_validation.run_acv_pipeline import run_full_acv_pipeline


def test_acv1_labe_classifier_training_and_locked_test(tmp_path):
    """Verifies Phase ACV-1 classifier training and locked test evaluation metrics."""
    out_dir = str(tmp_path / "acv1")
    report, artifacts = train_and_evaluate_labe_classifier(out_dir=out_dir)
    assert report["status"] == "ACV1_CLASSIFIER_TRAINED_AND_EVALUATED"
    assert report["test_performance_locked"]["f1_score"] >= 0.55
    assert report["test_performance_locked"]["auroc"] >= 0.65
    assert "vectorizer" in artifacts
    assert "clf_lr" in artifacts
    assert "clf_gb" in artifacts
    assert os.path.exists(os.path.join(out_dir, "acv1_classifier_report.md"))


def test_acv2_metric_disagreement_atlas(tmp_path):
    """Verifies Phase ACV-2 Metric Disagreement Atlas generation on 60 Wan pairs."""
    out_dir = str(tmp_path / "acv2")
    _, artifacts = train_and_evaluate_labe_classifier(out_dir=str(tmp_path / "train"))
    report = build_metric_disagreement_atlas(model_artifacts=artifacts, out_dir=out_dir)
    assert report["status"] == "ACV2_DISAGREEMENT_ATLAS_BUILT"
    assert report["pairs_count"] == 60
    assert -1.0 <= report["pearson_correlation"] <= 1.0
    assert 0.0 <= report["sign_agreement_percentage"] <= 1.0
    assert os.path.exists(os.path.join(out_dir, "acv2_disagreement_atlas.md"))


def test_acv3_evaluator_invariance_benchmark(tmp_path):
    """Verifies Phase ACV-3 counterfactual identity-swap evaluator invariance benchmark."""
    out_dir = str(tmp_path / "acv3")
    _, artifacts = train_and_evaluate_labe_classifier(out_dir=str(tmp_path / "train"))
    report = run_evaluator_invariance_benchmark(model_artifacts=artifacts, out_dir=out_dir)
    assert report["status"] == "ACV3_INVARIANCE_BENCHMARK_COMPLETED"
    assert report["total_counterfactual_comparisons"] > 0
    assert report["lexicon_mean_drift_control"] == 0.0
    assert report["classifier_mean_drift"] >= 0.0
    assert os.path.exists(os.path.join(out_dir, "acv3_invariance_report.md"))


def test_full_acv_pipeline_execution(tmp_path):
    """Verifies end-to-end execution of full ACV master pipeline."""
    out_dir = str(tmp_path / "full_acv")
    manifest = run_full_acv_pipeline(out_dir=out_dir)
    assert manifest["status"] == "ACV_FULL_PIPELINE_COMPLETED"
    assert os.path.exists(os.path.join(out_dir, "acv_manifest.json"))
