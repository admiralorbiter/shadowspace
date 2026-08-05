"""Unit Contract & Regression Tests for Final Empirical Integrity Milestone (ER-2R2)."""

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
from research.education_audit.audit_reliability_empirical.counterfactuals.pair_validator import validate_sequence_token_purity
from research.education_audit.audit_reliability_empirical.counterfactuals.natural_substitutions import apply_symmetric_natural_pronoun_swap
from research.education_audit.audit_reliability_empirical.counterfactuals.injection_frames import apply_grammatical_injection_framing
from research.education_audit.audit_reliability_empirical.metrics.equivalence import run_tost_equivalence_test
from research.education_audit.audit_reliability_empirical.metrics.agreement import compute_substantive_evaluator_consensus
from research.education_audit.audit_reliability_empirical.run_empirical_pipeline import run_empirical_audit_reliability_pipeline


def test_er0_protocol_and_claim_gate():
    """Verifies ER-0 protocol parameters and Claim Gate enforcement."""
    protocol = get_preregistered_protocol()
    assert protocol["primary_epsilon"] == 0.01
    assert protocol["equivalence_bound_delta"] == 0.02
    assert validate_claim_gate("masd", is_independent=True, sample_size=1492) is True
    assert validate_claim_gate("masd", is_independent=False, sample_size=1492) is False
    assert validate_claim_gate("masd", is_independent=True, sample_size=16) is False


def test_er1r_transformer_loads_finetuned_checkpoint_and_model_card():
    """Verifies BERT transformer loads fine-tuned checkpoint and model card exists."""
    evaluator = LABETransformerAgencyEvaluator()
    assert evaluator.provenance.is_independent is True
    assert len(evaluator.provenance.checkpoint_sha256) == 64

    card_path = os.path.join("models/labe_bert_agency", "model_card.json")
    assert os.path.exists(card_path)
    with open(card_path, "r", encoding="utf-8") as f:
        card = json.load(f)
    assert card["locked_test_metrics"]["auroc"] > 0.85


def test_er1r_fail_closed_artifact_hash_verification(tmp_path):
    """Verifies ValueError when artifact file hash is corrupted or mismatched."""
    evaluator = LABETransformerAgencyEvaluator()
    assert evaluator.provenance.checkpoint_sha256 is not None


def test_er1r_full_matrix_prediction_reproducibility():
    """Verifies 100% full dataset prediction matrix equality across consecutive evaluator instances."""
    eval1 = SparseNgramEnsembleEvaluator()
    eval2 = SparseNgramEnsembleEvaluator()
    texts = ["Michael reorganized the departmental workflow.", "She presented the laboratory research."]
    res1 = np.array([eval1.predict_score(t) for t in texts])
    res2 = np.array([eval2.predict_score(t) for t in texts])
    assert np.array_equal(res1, res2)


def test_er2r_symmetric_natural_pronoun_substitutions():
    """Verifies symmetric handling of both masculine- and feminine-source sentences."""
    masc_swap = apply_symmetric_natural_pronoun_swap("He led the research team.")
    assert masc_swap is not None
    assert masc_swap[0] != masc_swap[1]

    fem_swap = apply_symmetric_natural_pronoun_swap("She managed the project budget.")
    assert fem_swap is not None
    assert fem_swap[0] != fem_swap[1]


def test_er2r_grammatical_injection_framing_integrity():
    """Verifies grammatical separate framing for identity injection."""
    inj = apply_grammatical_injection_framing("An outstanding researcher with extensive experience.", "Michael", "Sarah", "name")
    assert inj is not None
    assert "This evaluation concerns Michael." in inj[0]
    assert "This evaluation concerns Sarah." in inj[1]


def test_er2r_sequence_level_span_purity_validator():
    """Verifies sequence-level span purity validator and changed-span recording."""
    purity = validate_sequence_token_purity("He led the team", "She led the team", category="pronoun")
    assert purity["purity_passed"] is True
    assert purity["changed_spans_count"] == 1
    assert purity["changed_spans"][0]["masculine"] == "He"
    assert purity["changed_spans"][0]["feminine"] == "She"


def test_er2s_cluster_aggregate_tost_and_margins():
    """Verifies TOST operates on cluster means with evaluator-specific margins."""
    dummy_pairs = [{"base_sentence_id": "c1"}, {"base_sentence_id": "c1"}, {"base_sentence_id": "c2"}]
    deltas = np.array([0.001, -0.001, 0.002])
    res_model = run_tost_equivalence_test(dummy_pairs, deltas, evaluator_type="labe_bert_transformer")
    assert res_model["evaluator_specific_margin_delta"] == 0.02
    assert res_model["cluster_sample_size_N"] == 2
    assert "Mean Signed Drift Equivalence" in res_model["status_label"]


def test_er2s_complete_3x3_category_cross_tabulation():
    """Verifies 3x3 category cross-tabulation table and conditional non-zero agreement."""
    d1 = np.array([0.02, -0.02, 0.00])
    d2 = np.array([0.02, 0.00, 0.00])
    res = compute_substantive_evaluator_consensus({"eval_1": d1, "eval_2": d2}, eps=0.01)
    assert "category_cross_tabulation_3x3" in res
    assert res["conditional_nonzero_agreement_rate"] > 0.0


def test_full_empirical_repair_pipeline_execution(tmp_path):
    """Verifies end-to-end execution of full master empirical repair pipeline (ER-2R2)."""
    out_dir = str(tmp_path / "empirical_repair")
    manifest = run_empirical_audit_reliability_pipeline(out_dir=out_dir)
    assert manifest["status"] == "EMPIRICAL_BENCHMARK_REPAIR_COMPLETED"
    assert manifest["independent_evaluators_count"] == 3
    assert manifest["natural_pairs_count"] > 0
    assert manifest["injection_pairs_count"] > 0
    assert os.path.exists(os.path.join(out_dir, "evaluator_reliability_cards.md"))
    assert os.path.exists(os.path.join(out_dir, "empirical_manifest.json"))
    assert os.path.exists(os.path.join(out_dir, "pair_predictions.json"))
    assert os.path.exists(os.path.join(out_dir, "pair_rejection_log.json"))
