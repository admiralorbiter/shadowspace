"""Unit Contract Tests for Agency Classifier Validation Suite (ACV-1 Baseline, ACV-2 Taxonomy, ACV-3 Invariance)."""

import os
import pytest

from research.education_audit.agency_classifier_validation.labe_classifier_trainer import train_and_evaluate_labe_classifier
from research.education_audit.agency_classifier_validation.metric_disagreement_atlas import build_metric_disagreement_atlas
from research.education_audit.agency_classifier_validation.evaluator_invariance_benchmark import run_evaluator_invariance_benchmark
from research.education_audit.agency_classifier_validation.run_acv_pipeline import run_full_acv_pipeline


def test_acv1_labe_sparse_ngram_baseline_training_and_locked_test(tmp_path):
    """Verifies Phase ACV-1 sparse n-gram classifier baseline training and locked test evaluation metrics."""
    out_dir = str(tmp_path / "acv1")
    report, artifacts = train_and_evaluate_labe_classifier(out_dir=out_dir)
    assert report["status"] == "ACV1_CLASSIFIER_BASELINE_EVALUATED"
    assert report["test_performance_locked"]["f1_score"] >= 0.55
    assert report["test_performance_locked"]["auroc"] >= 0.65
    assert "vectorizer" in artifacts
    assert "clf_lr" in artifacts
    assert "clf_gb" in artifacts
    assert os.path.exists(os.path.join(out_dir, "acv1_classifier_report.md"))


def test_acv2_metric_disagreement_atlas_mutually_exclusive_taxonomy(tmp_path):
    """Verifies Phase ACV-2 Metric Disagreement Atlas mutually exclusive taxonomy on 60 Wan pairs."""
    out_dir = str(tmp_path / "acv2")
    _, artifacts = train_and_evaluate_labe_classifier(out_dir=str(tmp_path / "train"))
    report = build_metric_disagreement_atlas(model_artifacts=artifacts, out_dir=out_dir)
    assert report["status"] == "ACV2_DISAGREEMENT_ATLAS_BUILT"
    assert report["pairs_count"] == 60
    assert "primary_taxonomy_counts" in report

    # Verify mutually exclusive sum equals total pairs count (60)
    counts = report["primary_taxonomy_counts"]
    total_tax = sum(counts.values())
    assert total_tax == 60, f"Taxonomy categories sum to {total_tax}, expected 60"
    assert os.path.exists(os.path.join(out_dir, "acv2_disagreement_atlas.md"))


def test_acv3_evaluator_invariance_benchmark_frame_separation(tmp_path):
    """Verifies Phase ACV-3 counterfactual identity-swap evaluator invariance with strict frame separation."""
    out_dir = str(tmp_path / "acv3")
    _, artifacts = train_and_evaluate_labe_classifier(out_dir=str(tmp_path / "train"))
    report = run_evaluator_invariance_benchmark(model_artifacts=artifacts, out_dir=out_dir)
    assert report["status"] == "ACV3_INVARIANCE_BENCHMARK_COMPLETED"
    assert report["total_counterfactual_comparisons"] == 9  # 6 name comparisons + 3 pronoun comparisons
    assert report["name_comparisons_count"] == 6
    assert report["pronoun_comparisons_count"] == 3
    assert report["lexicon_mean_abs_drift_control"] == 0.0
    assert -1.0 <= report["classifier_overall_mean_signed_drift"] <= 1.0
    assert 0.0 <= report["classifier_overall_mean_abs_drift"] <= 1.0
    assert os.path.exists(os.path.join(out_dir, "acv3_invariance_report.md"))


def test_full_acv_pipeline_execution(tmp_path):
    """Verifies end-to-end execution of full ACV master pipeline."""
    out_dir = str(tmp_path / "full_acv")
    manifest = run_full_acv_pipeline(out_dir=out_dir)
    assert manifest["status"] == "ACV_FULL_PIPELINE_COMPLETED"
    assert os.path.exists(os.path.join(out_dir, "acv_manifest.json"))
