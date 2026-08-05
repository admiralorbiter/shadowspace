"""Unit Contract Tests for Phase EDU-2a-R1.2 Complete-Generation, Provenance & Review Contracts."""

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


def test_rating_packet_generation(tmp_path):
    """Verifies generate_blinded_rating_packet exports CSV, JSONL with rating fields, 5 duplicates, and private key."""
    cases = build_synthetic_audit_cases()
    cases_map = {c.case_id: c for c in cases}
    variants = build_variants_for_case(cases[0])
    var_map = {v.variant_id: v for v in variants}

    adapter = OllamaEducationAdapter(use_mock_fallback=True)
    adapter.is_loaded = False
    gen_records = [
        adapter.generate(cases[0], v, "minimal_prompt", PROMPT_TEMPLATES["minimal_prompt"], repeat_index=0)
        for v in variants
    ]

    out_dir = str(tmp_path / "packet_test")
    priv_key_dir = str(tmp_path / "private_review")
    csv_p, jsonl_p, key_p = generate_blinded_rating_packet(gen_records, var_map, cases_map=cases_map, out_dir=out_dir, private_key_dir=priv_key_dir)

    assert os.path.exists(csv_p)
    assert os.path.exists(jsonl_p)
    assert os.path.exists(key_p)

    with open(key_p, "r", encoding="utf-8") as f:
        key_data = json.load(f)

    assert len(key_data) == 10
    assert "LTR_R1_001" in key_data


def test_strict_completion_gate_on_truncation(tmp_path):
    """Verifies that truncation count > 0 causes completion_integrity_status = FAILED and go_to_full_pilot = False."""
    out_dir = tmp_path / "truncation_test"
    out_dir.mkdir()

    with open(out_dir / "screening_evaluations.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "generation_id": "g1", "case_id": "c1", "variant_id": "v1", "condition": "anonymous",
            "prompt_id": "minimal_prompt", "recommendation_strength_score": 3.0, "hallucinations_per_100_words": 0.0
        }) + "\n")

    with open(out_dir / "generations.jsonl", "w", encoding="utf-8") as f:
        for i in range(60):
            reason = "length" if i < 50 else "stop"
            f.write(json.dumps({
                "generation_id": f"g_{i}", "parameters": {"done_reason": reason}
            }) + "\n")

    manifest = run_edu2a_analysis(data_dir=str(out_dir))

    assert manifest["completion_integrity_status"] == "FAILED"
    assert manifest["truncation_count"] == 50
    assert manifest["go_to_full_pilot"] is False


def test_provenance_shas_in_manifest(tmp_path):
    """Verifies analysis manifest exports all 5 explicit commit SHAs."""
    manifest = run_edu2a_analysis()
    assert "generation_code_commit_sha" in manifest
    assert "generation_artifact_commit_sha" in manifest
    assert "analysis_code_commit_sha" in manifest
    assert "analysis_results_commit_sha" in manifest
    assert "documentation_commit_sha" in manifest


def test_manual_ratings_validator(tmp_path):
    """Verifies validate_manual_ratings_file strictly validates score bounds and quotas."""
    ratings_f = str(tmp_path / "manual_ratings.jsonl")
    valid_lids = [f"LTR_R1_{i:03d}" for i in range(1, 66)]
    r2_lids = valid_lids[:20]

    # Test invalid score range (6.0)
    with open(ratings_f, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "reviewer_id": "R1", "letter_id": "LTR_R1_001", "review_pass": 1,
            "recommendation_strength_score": 6.0
        }) + "\n")

    valid, errs = validate_manual_ratings_file(ratings_f, valid_lids, r2_lids)
    assert valid is False
    assert any("out of bounds" in e for e in errs)


def test_review_reliability_calculator():
    """Verifies quadratic_weighted_kappa and inter/intra-rater reliability functions."""
    y1 = [4.0, 3.0, 5.0, 2.0, 4.0]
    y2 = [4.0, 3.0, 5.0, 2.0, 4.0]
    qwk_perfect = quadratic_weighted_kappa(y1, y2)
    assert qwk_perfect == 1.0

    r1_records = [{"letter_id": f"LTR_{i}", "recommendation_strength_score": 4.0} for i in range(20)]
    r2_records = [{"letter_id": f"LTR_{i}", "recommendation_strength_score": 4.0} for i in range(20)]
    rel = compute_inter_rater_reliability(r1_records, r2_records)
    assert rel["reliability_gate_passed"] is True
    assert rel["weighted_kappa"] == 1.0
