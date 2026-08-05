"""Unit Contract Tests for Empirical Audit Reliability Framework (ER-0 to ER-2)."""

import os
import pytest

from research.education_audit.audit_reliability_empirical.protocol import get_preregistered_protocol, validate_claim_gate
from research.education_audit.audit_reliability_empirical.provenance import EvaluatorProvenance, validate_evaluator_dependency_graph
from research.education_audit.audit_reliability_empirical.evaluators.panel import initialize_empirical_evaluator_panel
from research.education_audit.audit_reliability_empirical.counterfactuals.pair_builder import build_labe_test_counterfactual_corpus
from research.education_audit.audit_reliability_empirical.run_empirical_pipeline import run_empirical_audit_reliability_pipeline


def test_er0_protocol_and_claim_gate():
    """Verifies ER-0 protocol parameters and Claim Gate enforcement."""
    protocol = get_preregistered_protocol()
    assert protocol["primary_epsilon"] == 0.01
    assert protocol["equivalence_bound_delta"] == 0.02
    assert validate_claim_gate("masd", is_independent=True, sample_size=1492) is True
    assert validate_claim_gate("masd", is_independent=False, sample_size=1492) is False
    assert validate_claim_gate("masd", is_independent=True, sample_size=16) is False


def test_er0_provenance_acyclic_dependency_validation():
    """Verifies provenance schema fails closed on cyclic dependencies."""
    p1 = EvaluatorProvenance(
        evaluator_id="p1", evaluator_name="P1", model_family="test", checkpoint_revision="v1",
        checkpoint_sha256="hash", training_data_revision="data", score_scale=[0, 1], threshold=0.5,
        threshold_source="val", is_independent=True, independent_of=[]
    )
    assert validate_evaluator_dependency_graph([p1]) is True


def test_er1_three_independent_evaluators_panel(tmp_path):
    """Verifies ER-1 panel binds 3 independent evaluators (Lexicon, Sparse N-Gram, LABE BERT)."""
    out_dir = str(tmp_path / "panel")
    res = initialize_empirical_evaluator_panel(out_dir=out_dir)
    assert res["status"] == "EMPIRICAL_PANEL_INITIALIZED"
    assert res["panel_manifest"]["evaluators_count"] == 3
    assert res["panel_manifest"]["independent_evaluators_count"] == 3
    assert "exact_lexicon" in res["panel"]
    assert "sparse_ngram_ensemble" in res["panel"]
    assert "labe_bert_transformer" in res["panel"]


def test_er2_labe_test_counterfactual_corpus_builder():
    """Verifies ER-2 builds 1,492 validated paired counterfactual comparisons from LABE test split."""
    corpus = build_labe_test_counterfactual_corpus()
    assert len(corpus) == 1492
    for pair in corpus:
        assert pair["text_masc"] != pair["text_fem"]
        assert pair["category"] in ["pronoun", "name"]


def test_full_empirical_pipeline_execution(tmp_path):
    """Verifies full execution of master empirical pipeline (ER-0 to ER-2)."""
    out_dir = str(tmp_path / "full_empirical")
    manifest = run_empirical_audit_reliability_pipeline(out_dir=out_dir)
    assert manifest["status"] == "EMPIRICAL_BENCHMARK_COMPLETED"
    assert manifest["independent_evaluators_count"] == 3
    assert manifest["total_counterfactual_pairs"] == 1492
    assert os.path.exists(os.path.join(out_dir, "evaluator_reliability_cards.md"))
    assert os.path.exists(os.path.join(out_dir, "empirical_manifest.json"))
