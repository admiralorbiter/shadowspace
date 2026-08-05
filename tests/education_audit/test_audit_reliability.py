"""Unit Contract Tests for Audit Reliability & Counterfactual Meta-Evaluation Framework (AR-1 to AR-5)."""

import os
import pytest

from research.education_audit.audit_reliability.evaluator_panel import initialize_evaluator_panel
from research.education_audit.audit_reliability.counterfactual_evaluator_benchmark import run_counterfactual_evaluator_benchmark
from research.education_audit.audit_reliability.cross_domain_transfer import run_cross_domain_transfer_benchmark
from research.education_audit.audit_reliability.attribution_and_factuality_pilot import classify_sentence_attributions, detect_factual_claim_inflation, run_attribution_and_factuality_pilot
from research.education_audit.audit_reliability.run_audit_reliability import run_full_audit_reliability_suite


def test_ar1_evaluator_panel_initialization():
    """Verifies AR-1 evaluator panel binds all three evaluator instruments."""
    res = initialize_evaluator_panel()
    assert res["status"] == "EVALUATOR_PANEL_INITIALIZED"
    assert res["evaluators_count"] == 3
    assert "exact_lexicon" in res["panel"]
    assert "sparse_ngram_ensemble" in res["panel"]
    assert "transformer_contextual" in res["panel"]


def test_ar2_counterfactual_evaluator_reliability_cards(tmp_path):
    """Verifies AR-2 generates Reliability Cards with MASD, CFR, and signed drift."""
    out_dir = str(tmp_path / "ar2")
    res = run_counterfactual_evaluator_benchmark(out_dir=out_dir)
    assert res["status"] == "AR2_RELIABILITY_CARDS_GENERATED"
    assert res["evaluators_evaluated"] == 3

    cards = res["reliability_cards"]
    assert "eval_exact_lexicon" in cards
    assert cards["eval_exact_lexicon"]["masd_mean_absolute_score_difference"] == 0.0
    assert 0.0 <= cards["eval_sparse_ngram_ensemble"]["masd_mean_absolute_score_difference"] <= 1.0
    assert os.path.exists(os.path.join(out_dir, "ar2_evaluator_reliability_cards.md"))


def test_ar3_cross_domain_evaluator_transfer(tmp_path):
    """Verifies AR-3 cross-domain transfer benchmark calculates degradation delta."""
    out_dir = str(tmp_path / "ar3")
    res = run_cross_domain_transfer_benchmark(out_dir=out_dir)
    assert res["status"] == "AR3_CROSS_DOMAIN_TRANSFER_COMPLETED"
    assert "domain_transfer_degradation_delta_r" in res
    assert os.path.exists(os.path.join(out_dir, "ar3_cross_domain_transfer.md"))


def test_ar5_attribution_and_factuality_pilot(tmp_path):
    """Verifies AR-5 causal attribution taxonomy and factual claim inflation detection."""
    out_dir = str(tmp_path / "ar5")

    attrs = classify_sentence_attributions("She is a brilliant researcher who worked hard.")
    assert attrs["ability"] >= 1
    assert attrs["effort"] >= 1

    facts = detect_factual_claim_inflation("Built a team of 12 enterprise developers", ["built software"])
    assert facts["inflation_triggers_count"] >= 1

    res = run_attribution_and_factuality_pilot(out_dir=out_dir)
    assert res["status"] == "AR5_ATTRIBUTION_AND_FACTUALITY_COMPLETED"
    assert os.path.exists(os.path.join(out_dir, "ar4_attribution_and_factuality_report.md"))


def test_full_audit_reliability_suite_pipeline(tmp_path):
    """Verifies end-to-end execution of full master audit reliability suite."""
    out_dir = str(tmp_path / "full_ar")
    manifest = run_full_audit_reliability_suite(out_dir=out_dir)
    assert manifest["status"] == "AUDIT_RELIABILITY_SUITE_COMPLETED"
    assert os.path.exists(os.path.join(out_dir, "audit_reliability_manifest.json"))
