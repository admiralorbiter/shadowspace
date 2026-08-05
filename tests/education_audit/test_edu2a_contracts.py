"""Unit Contract Tests for Phase EDU-2a-R1.2a Provenance, Validator, & Reliability Gates."""

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

    # Verify inter-rater reliability accepts zero-variance when agreement is 100%
    r1_records, r2_records = [], []
    for i in range(1, 21):
        lid = f"LTR_R1_{i:03d}"
        r1_records.append({"rating_id": f"R1_p1_{lid}", "reviewer_id": "R1", "letter_id": lid, "review_pass": 1, "recommendation_strength_score": 4.0, "opportunity_strength_score": 4.0, "leadership_language_score": 3.0, "competence_language_score": 4.0, "warmth_language_score": 3.0, "placeholder_or_template_artifact": False, "incomplete_letter_flag": False})
        r2_records.append({"rating_id": f"R2_p1_{lid}", "reviewer_id": "R2", "letter_id": lid, "review_pass": 1, "recommendation_strength_score": 4.0, "opportunity_strength_score": 4.0, "leadership_language_score": 3.0, "competence_language_score": 4.0, "warmth_language_score": 3.0, "placeholder_or_template_artifact": False, "incomplete_letter_flag": False})
        r1_records.append({"rating_id": f"R1_p2_{lid}", "reviewer_id": "R1", "letter_id": lid, "review_pass": 2, "factual_fidelity_score": 5.0, "unsupported_positive_claims_count": 0, "unsupported_negative_claims_count": 0, "major_accomplishment_omissions_count": 0, "adjudication_notes": ""})
        r2_records.append({"rating_id": f"R2_p2_{lid}", "reviewer_id": "R2", "letter_id": lid, "review_pass": 2, "factual_fidelity_score": 5.0, "unsupported_positive_claims_count": 0, "unsupported_negative_claims_count": 0, "major_accomplishment_omissions_count": 0, "adjudication_notes": ""})

    rel = compute_inter_rater_reliability(r1_records + r2_records)
    assert rel["status"] == "PASSED_WITH_KAPPA_UNIDENTIFIABLE"
    assert rel["reliability_gate_passed"] is True


def test_manual_ratings_validator_strict(tmp_path):
    """Verifies validate_manual_ratings_file rejects incomplete quotas, wrong R2 IDs, or missing required fields."""
    ratings_f = str(tmp_path / "manual_ratings.jsonl")
    valid_lids = [f"LTR_R1_{i:03d}" for i in range(1, 66)]

    # Test missing Pass 2 (only Pass 1 rows)
    with open(ratings_f, "w", encoding="utf-8") as f:
        for i in range(1, 66):
            f.write(json.dumps({
                "rating_id": f"r1_{i}", "reviewer_id": "R1", "letter_id": f"LTR_R1_{i:03d}", "review_pass": 1,
                "recommendation_strength_score": 4.0, "opportunity_strength_score": 4.0, "leadership_language_score": 3.0,
                "competence_language_score": 4.0, "warmth_language_score": 3.0, "placeholder_or_template_artifact": False, "incomplete_letter_flag": False
            }) + "\n")

    valid, errs = validate_manual_ratings_file(ratings_f, valid_lids)
    assert valid is False
    assert any("Incomplete Reviewer 1 Pass 2 quota" in e for e in errs)


def test_realistic_60_generation_mock_packet_and_reliability(tmp_path):
    """Creates a full 60-generation mock packet, verifies R1 Pass1 (65), R1 Pass2 (65), R2 Pass1 (20), R2 Pass2 (20)."""
    cases = build_synthetic_audit_cases()[:2]
    cases_map = {c.case_id: c for c in cases}

    gen_records = []
    var_map = {}
    adapter = OllamaEducationAdapter(use_mock_fallback=True)
    adapter.is_loaded = False

    for c in cases:
        vars_c = build_variants_for_case(c)
        for v in vars_c:
            var_map[v.variant_id] = v
            for p_id in ["minimal_prompt", "structured_prompt"]:
                for s in [101, 202, 303]:
                    g = adapter.generate(c, v, p_id, PROMPT_TEMPLATES[p_id], repeat_index=0, seed=s)
                    gen_records.append(g)

    assert len(gen_records) == 60

    out_dir = str(tmp_path / "packet_test_60")
    priv_dir = str(tmp_path / "private_review_60")
    generate_blinded_rating_packet(gen_records, var_map, cases_map=cases_map, out_dir=out_dir, private_key_dir=priv_dir)

    manifest_p = os.path.join(priv_dir, "edu_2a_r1_review_design_manifest.json")
    assert os.path.exists(manifest_p)

    with open(manifest_p, "r", encoding="utf-8") as f:
        d_man = json.load(f)

    assert len(d_man["reviewer2_allowed_letter_ids"]) == 20
    assert len(d_man["intra_rater_duplicate_pairs"]) == 5
