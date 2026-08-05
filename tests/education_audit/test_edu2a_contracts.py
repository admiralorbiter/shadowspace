"""Unit Contract Tests for Phase EDU-2a-R1.2b Submission Integrity, Typed Validation, & Reliability Gates."""

import json
import os
import pytest

from research.education_audit.adapters.ollama import OllamaEducationAdapter
from research.education_audit.case_builder import build_synthetic_audit_cases
from research.education_audit.variant_builder import build_variants_for_case
from research.education_audit.prompt_registry import PROMPT_TEMPLATES
from research.education_audit.evaluation.blinding import blind_generation_text
from research.education_audit.reporting.rating_packet import generate_blinded_rating_packet
from research.education_audit.analysis.edu_2a_analysis import run_edu2a_analysis
from research.education_audit.evaluation.validate_manual_ratings import validate_manual_ratings_file
from research.education_audit.evaluation.analyze_review_reliability import (
    quadratic_weighted_kappa,
    compute_intra_rater_reliability,
    compute_inter_rater_reliability,
)


def test_ollama_adapter_mock_fallback_mode(monkeypatch):
    """Verifies OllamaEducationAdapter fallback mode when ping_and_inspect fails and use_mock_fallback=True."""
    adapter = OllamaEducationAdapter(model_name="gemma3:12b", use_mock_fallback=True)
    monkeypatch.setattr(adapter, "ping_and_inspect", lambda: False)
    adapter.is_loaded = False
    case = build_synthetic_audit_cases()[0]
    variant = build_variants_for_case(case)[0]

    g = adapter.generate(case, variant, "minimal_prompt", PROMPT_TEMPLATES["minimal_prompt"], repeat_index=0, seed=101)
    assert g.model_id.startswith("mock-fallback")
    assert len(g.output_text) > 20


def test_non_destructive_blinding_and_leakage_detection():
    """Verifies non-destructive blinding preserves raw model prose while detecting explicit leakage."""
    raw_text_clean = "It is my pleasure to recommend [CANDIDATE]. They have demonstrated outstanding skill in their projects."
    blinded_c, leakage_c, _ = blind_generation_text(raw_text_clean, "Alexander", "he", "him", "his")
    assert leakage_c is False
    assert blinded_c == raw_text_clean

    raw_text_leaked = "It is my pleasure to recommend Alexander. He has demonstrated outstanding skill in his projects."
    blinded_l, leakage_l, details_l = blind_generation_text(raw_text_leaked, "Alexander", "he", "him", "his")
    assert leakage_l is True
    assert "Alexander" in details_l or "Gendered terms" in details_l


def test_provenance_shas_in_manifest(tmp_path):
    """Verifies analysis manifest exports exact provenance commit SHAs without circular self-references."""
    manifest = run_edu2a_analysis()
    assert "generation_code_commit_sha" in manifest
    assert "generation_artifact_commit_sha" in manifest
    assert "r1_1_analysis_code_commit_sha" in manifest
    assert "review_activation_code_commit_sha" in manifest
    assert "documentation_commit_sha" in manifest
    assert "parent_code_commit_sha" in manifest


def test_zero_variance_kappa_handling():
    """Verifies quadratic_weighted_kappa returns (None, 'UNIDENTIFIABLE_ZERO_VARIANCE') for identical ratings."""
    y1 = [4.0] * 20
    y2 = [4.0] * 20
    qwk, status = quadratic_weighted_kappa(y1, y2)
    assert qwk is None
    assert status == "UNIDENTIFIABLE_ZERO_VARIANCE"


def test_validator_fails_closed_when_manifest_missing(tmp_path):
    """Verifies validate_manual_ratings_file fails closed when design manifest is missing."""
    ratings_f = str(tmp_path / "manual_ratings.jsonl")
    valid_lids = [f"LTR_R1_{i:03d}" for i in range(1, 66)]

    with open(ratings_f, "w", encoding="utf-8") as f:
        f.write(json.dumps({"rating_id": "r1"}) + "\n")

    valid, errs = validate_manual_ratings_file(ratings_f, valid_lids, design_manifest_path=str(tmp_path / "nonexistent.json"))
    assert valid is False
    assert any("Required review-design manifest is missing" in e for e in errs)


def test_validator_rejects_string_booleans_and_boolean_integers(tmp_path):
    """Verifies validate_manual_ratings_file rejects string booleans ('False') and boolean integers (True for count)."""
    cases = build_synthetic_audit_cases()[:2]
    cases_map = {c.case_id: c for c in cases}
    var_map = {}
    adapter = OllamaEducationAdapter(use_mock_fallback=True)
    adapter.is_loaded = False
    gen_records = []
    for c in cases:
        for v in build_variants_for_case(c):
            var_map[v.variant_id] = v
            for p_id in ["minimal_prompt", "structured_prompt"]:
                for s in [101, 202, 303]:
                    gen_records.append(adapter.generate(c, v, p_id, PROMPT_TEMPLATES[p_id], repeat_index=0, seed=s))

    out_dir = str(tmp_path / "packet_test")
    priv_dir = str(tmp_path / "private_review")
    generate_blinded_rating_packet(gen_records, var_map, cases_map=cases_map, out_dir=out_dir, private_key_dir=priv_dir)

    valid_lids = [f"LTR_R1_{i:03d}" for i in range(1, 66)]
    design_p = os.path.join(priv_dir, "edu_2a_r1_review_design_manifest.json")

    # String boolean test
    ratings_f1 = str(tmp_path / "ratings_str_bool.jsonl")
    with open(ratings_f1, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "rating_id": "r1", "reviewer_id": "R1", "letter_id": "LTR_R1_001", "review_pass": 1,
            "recommendation_strength_score": 4.0, "opportunity_strength_score": 4.0, "leadership_language_score": 3.0,
            "competence_language_score": 4.0, "warmth_language_score": 3.0,
            "placeholder_or_template_artifact": "False", "incomplete_letter_flag": False
        }) + "\n")

    valid1, errs1 = validate_manual_ratings_file(ratings_f1, valid_lids, design_manifest_path=design_p, private_key_dir=priv_dir)
    assert valid1 is False
    assert any("must be JSON boolean" in e for e in errs1)

    # Boolean integer test (True as count)
    ratings_f2 = str(tmp_path / "ratings_bool_int.jsonl")
    with open(ratings_f2, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "rating_id": "r2", "reviewer_id": "R1", "letter_id": "LTR_R1_001", "review_pass": 2,
            "factual_fidelity_score": 5.0, "unsupported_positive_claims_count": True,
            "unsupported_negative_claims_count": 0, "major_accomplishment_omissions_count": 0, "adjudication_notes": ""
        }) + "\n")

    valid2, errs2 = validate_manual_ratings_file(ratings_f2, valid_lids, design_manifest_path=design_p, private_key_dir=priv_dir)
    assert valid2 is False
    assert any("must be non-negative integer" in e for e in errs2)


def test_positive_fixture_completes_review_and_opens_pilot(tmp_path):
    """Creates a full 60-generation mock packet with 170 valid rating records, asserting go_to_full_pilot = True."""
    cases = build_synthetic_audit_cases()[:2]
    cases_map = {c.case_id: c for c in cases}
    var_map = {}
    adapter = OllamaEducationAdapter(use_mock_fallback=True)
    adapter.is_loaded = False
    gen_records = []
    for c in cases:
        for v in build_variants_for_case(c):
            var_map[v.variant_id] = v
            for p_id in ["minimal_prompt", "structured_prompt"]:
                for s in [101, 202, 303]:
                    gen_records.append(adapter.generate(c, v, p_id, PROMPT_TEMPLATES[p_id], repeat_index=0, seed=s))

    out_dir = str(tmp_path / "full_positive_test")
    priv_dir = str(tmp_path / "private_review")
    generate_blinded_rating_packet(gen_records, var_map, cases_map=cases_map, out_dir=out_dir, private_key_dir=priv_dir)

    manifest_p = os.path.join(priv_dir, "edu_2a_r1_review_design_manifest.json")
    with open(manifest_p, "r", encoding="utf-8") as f:
        d_man = json.load(f)

    r2_allowed = d_man["reviewer2_allowed_letter_ids"]

    # Generate 170 valid ratings (65 R1 p1, 65 R1 p2, 20 R2 p1, 20 R2 p2)
    ratings = []
    for i in range(1, 66):
        lid = f"LTR_R1_{i:03d}"
        ratings.append({"rating_id": f"R1_p1_{lid}", "reviewer_id": "R1", "letter_id": lid, "review_pass": 1, "recommendation_strength_score": 4.0, "opportunity_strength_score": 4.0, "leadership_language_score": 3.0, "competence_language_score": 4.0, "warmth_language_score": 3.0, "placeholder_or_template_artifact": False, "incomplete_letter_flag": False})
        ratings.append({"rating_id": f"R1_p2_{lid}", "reviewer_id": "R1", "letter_id": lid, "review_pass": 2, "factual_fidelity_score": 5.0, "unsupported_positive_claims_count": 0, "unsupported_negative_claims_count": 0, "major_accomplishment_omissions_count": 0, "adjudication_notes": ""})

    for lid in r2_allowed:
        ratings.append({"rating_id": f"R2_p1_{lid}", "reviewer_id": "R2", "letter_id": lid, "review_pass": 1, "recommendation_strength_score": 4.0, "opportunity_strength_score": 4.0, "leadership_language_score": 3.0, "competence_language_score": 4.0, "warmth_language_score": 3.0, "placeholder_or_template_artifact": False, "incomplete_letter_flag": False})
        ratings.append({"rating_id": f"R2_p2_{lid}", "reviewer_id": "R2", "letter_id": lid, "review_pass": 2, "factual_fidelity_score": 5.0, "unsupported_positive_claims_count": 0, "unsupported_negative_claims_count": 0, "major_accomplishment_omissions_count": 0, "adjudication_notes": ""})

    eval_path = os.path.join(out_dir, "screening_evaluations.jsonl")
    with open(eval_path, "w", encoding="utf-8") as f:
        for g in gen_records:
            f.write(json.dumps({
                "generation_id": g.generation_id, "case_id": g.case_id, "variant_id": g.variant_id, "condition": g.condition,
                "prompt_id": g.prompt_id, "recommendation_strength_score": 3.0, "hallucinations_per_100_words": 0.0
            }) + "\n")

    gen_path = os.path.join(out_dir, "generations.jsonl")
    with open(gen_path, "w", encoding="utf-8") as f:
        for g in gen_records:
            f.write(json.dumps(g.__dict__) + "\n")

    ratings_f = os.path.join(out_dir, "manual_ratings.jsonl")
    with open(ratings_f, "w", encoding="utf-8") as f:
        for r in ratings:
            f.write(json.dumps(r) + "\n")

    manifest = run_edu2a_analysis(data_dir=out_dir, private_key_dir=priv_dir)
    assert manifest["manual_review_status"] == "COMPLETED"
    assert manifest["review_reliability_status"] == "PASSED_WITH_KAPPA_UNIDENTIFIABLE"
    assert manifest["go_to_full_pilot"] is True

