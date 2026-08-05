"""Unit Contract Tests for Audit Reliability & Counterfactual Meta-Evaluation Framework Scaffold."""

import os
import pytest

from research.education_audit.audit_reliability.evaluator_panel import initialize_evaluator_panel
from research.education_audit.audit_reliability.counterfactual_evaluator_benchmark import run_counterfactual_evaluator_benchmark
from research.education_audit.audit_reliability.cross_domain_transfer import run_cross_domain_transfer_benchmark
from research.education_audit.audit_reliability.attribution_and_factuality_pilot import classify_sentence_attributions, detect_factual_claim_inflation, run_attribution_and_factuality_pilot
from research.education_audit.audit_reliability.run_audit_reliability import run_full_audit_reliability_suite


def test_ar1_evaluator_panel_initialization():
    """Verifies AR-1 evaluator panel binds 2 independent evaluators + 1 proxy evaluator."""
    res = initialize_evaluator_panel()
    assert res["status"] == "EVALUATOR_PANEL_INITIALIZED"
    assert res["evaluators_count"] == 3
    assert res["independent_evaluators_count"] == 2
    assert "exact_lexicon" in res["panel"]
    assert "sparse_ngram_ensemble" in res["panel"]
    assert "length_adjusted_ngram_proxy" in res["panel"]
    assert res["panel"]["length_adjusted_ngram_proxy"].is_independent is False


def test_ar2_counterfactual_evaluator_smoke_test(tmp_path):
    """Verifies AR-2 generates Reliability Cards for 16-pair smoke test."""
    out_dir = str(tmp_path / "ar2")
    res = run_counterfactual_evaluator_benchmark(out_dir=out_dir)
    assert res["status"] == "AR2_SMOKE_TEST_COMPLETED"
    assert res["evaluators_evaluated"] == 3

    cards = res["reliability_cards"]
    assert "exact_lexicon" in cards
    assert cards["exact_lexicon"]["masd_mean_absolute_score_difference"] == 0.0
    assert 0.0 <= cards["sparse_ngram_ensemble"]["masd_mean_absolute_score_difference"] <= 1.0
    assert os.path.exists(os.path.join(out_dir, "ar2_evaluator_reliability_cards.md"))


def test_ar3_cross_context_score_agreement_contrast(tmp_path):
    """Verifies AR-3 cross-context agreement contrast calculates correlation delta."""
    out_dir = str(tmp_path / "ar3")
    res = run_cross_domain_transfer_benchmark(out_dir=out_dir)
    assert res["status"] == "AR3_CROSS_CONTEXT_CONTRAST_COMPLETED"
    assert "correlation_contrast_delta_r" in res
    assert os.path.exists(os.path.join(out_dir, "ar3_cross_domain_transfer.md"))


def test_ar4_synthetic_attribution_and_coverage_fixture(tmp_path):
    """Verifies AR-4 synthetic known-answer attribution and coverage fixture."""
    out_dir = str(tmp_path / "ar4")

    attrs = classify_sentence_attributions("She is a brilliant researcher who worked hard.")
    assert attrs["ability"] >= 1
    assert attrs["effort"] >= 1

    facts = detect_factual_claim_inflation("Built a team of 12 enterprise developers", ["built software"])
    assert facts["inflation_triggers_count"] >= 1

    res = run_attribution_and_factuality_pilot(out_dir=out_dir)
    assert res["status"] == "AR4_SYNTHETIC_ATTRIBUTION_FIXTURE_VERIFIED"
    assert os.path.exists(os.path.join(out_dir, "ar4_attribution_and_factuality_report.md"))


def test_full_audit_reliability_scaffold_pipeline(tmp_path):
    """Verifies end-to-end execution of full master audit reliability scaffold pipeline."""
    out_dir = str(tmp_path / "full_ar")
    manifest = run_full_audit_reliability_suite(out_dir=out_dir)
    assert manifest["status"] == "AUDIT_RELIABILITY_FRAMEWORK_SCAFFOLD_VALIDATED"
    assert os.path.exists(os.path.join(out_dir, "audit_reliability_manifest.json"))
