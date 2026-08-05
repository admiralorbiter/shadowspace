"""Unit Contract & Regression Tests for Empirical Repair Milestone (ER-1R, ER-2R, ER-2S)."""

import json
import os
import pytest
import numpy as np

from research.education_audit.audit_reliability_empirical.protocol import get_preregistered_protocol, validate_claim_gate
from research.education_audit.audit_reliability_empirical.provenance import EvaluatorProvenance, validate_evaluator_dependency_graph
from research.education_audit.audit_reliability_empirical.evaluators.panel import initialize_empirical_evaluator_panel
from research.education_audit.audit_reliability_empirical.evaluators.labe_transformer import LABETransformerAgencyEvaluator
from research.education_audit.audit_reliability_empirical.evaluators.sparse_ngram import SparseNgramEnsembleEvaluator
from research.education_audit.audit_reliability_empirical.counterfactuals.pair_builder import build_labe_test_counterfactual_corpus
from research.education_audit.audit_reliability_empirical.counterfactuals.pair_validator import validate_counterfactual_pair_purity
from research.education_audit.audit_reliability_empirical.metrics.equivalence import run_tost_equivalence_test
from research.education_audit.audit_reliability_empirical.run_empirical_pipeline import run_empirical_audit_reliability_pipeline


def test_er0_protocol_and_claim_gate():
    """Verifies ER-0 protocol parameters and Claim Gate enforcement."""
    protocol = get_preregistered_protocol()
    assert protocol["primary_epsilon"] == 0.01
    assert protocol["equivalence_bound_delta"] == 0.02
    assert validate_claim_gate("masd", is_independent=True, sample_size=1492) is True
    assert validate_claim_gate("masd", is_independent=False, sample_size=1492) is False
    assert validate_claim_gate("masd", is_independent=True, sample_size=16) is False


def test_er1r_transformer_loads_finetuned_checkpoint_with_zero_missing_keys():
    """Verifies BERT transformer loads fine-tuned checkpoint with zero missing keys."""
    evaluator = LABETransformerAgencyEvaluator()
    assert evaluator.provenance.is_independent is True
    assert len(evaluator.provenance.checkpoint_sha256) == 64
    score = evaluator.predict_score("He led the engineering team to deliver the project.")
    assert 0.0 <= score <= 1.0


def test_er1r_ngram_loaded_from_frozen_artifact():
    """Verifies sparse n-gram evaluator loads from serialized joblib artifacts."""
    evaluator = SparseNgramEnsembleEvaluator()
    assert evaluator.provenance.is_independent is True
    assert len(evaluator.provenance.checkpoint_sha256) == 64
    score = evaluator.predict_score("She managed the laboratory budget.")
    assert 0.0 <= score <= 1.0


def test_er1r_two_pipeline_runs_are_prediction_identical():
    """Verifies 100% two-run prediction bitwise identity across consecutive panel calls."""
    eval1 = SparseNgramEnsembleEvaluator()
    eval2 = SparseNgramEnsembleEvaluator()
    text = "Michael reorganized the departmental workflow."
    assert eval1.predict_score(text) == eval2.predict_score(text)


def test_er2r_separated_natural_and_injection_corpora():
    """Verifies ER-2R builds separated natural and controlled injection benchmark corpora."""
    corpora = build_labe_test_counterfactual_corpus()
    assert corpora["total_natural_pairs"] > 0
    assert corpora["total_injection_pairs"] > 0

    for pair in corpora["natural_corpus"]:
        assert validate_counterfactual_pair_purity(pair["text_masc"], pair["text_fem"], category="pronoun")

    for pair in corpora["injection_corpus"]:
        assert validate_counterfactual_pair_purity(pair["text_masc"], pair["text_fem"], category="name")


def test_er2s_evaluator_specific_margins_and_tost_relabeling():
    """Verifies evaluator-specific equivalence bounds and TOST result relabeling."""
    res_lex = run_tost_equivalence_test(np.array([0.0, 0.0, 0.0]), evaluator_type="exact_lexicon")
    assert res_lex["evaluator_specific_margin_delta"] == 2.0
    assert "Mean Signed Drift Equivalence" in res_lex["status_label"]

    res_model = run_tost_equivalence_test(np.array([0.001, -0.001, 0.002]), evaluator_type="labe_bert_transformer")
    assert res_model["evaluator_specific_margin_delta"] == 0.02
    assert "Mean Signed Drift Equivalence" in res_model["status_label"]


def test_full_empirical_repair_pipeline_execution(tmp_path):
    """Verifies end-to-end execution of full master empirical repair pipeline (ER-1R, ER-2R, ER-2S)."""
    out_dir = str(tmp_path / "empirical_repair")
    manifest = run_empirical_audit_reliability_pipeline(out_dir=out_dir)
    assert manifest["status"] == "EMPIRICAL_BENCHMARK_REPAIR_COMPLETED"
    assert manifest["independent_evaluators_count"] == 3
    assert manifest["natural_pairs_count"] > 0
    assert manifest["injection_pairs_count"] > 0
    assert os.path.exists(os.path.join(out_dir, "evaluator_reliability_cards.md"))
    assert os.path.exists(os.path.join(out_dir, "empirical_manifest.json"))
    assert os.path.exists(os.path.join(out_dir, "pair_rejection_log.json"))
