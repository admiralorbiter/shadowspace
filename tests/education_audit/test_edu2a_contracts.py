"""Unit Contract Tests for Phase EDU-2a-R1.2c Submission Compiler, Score Type Safety, & Dynamic SHA Capture."""

import csv
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
from research.education_audit.evaluation.compile_review_submissions import compile_review_submissions
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
    """Verifies analysis manifest exports exact provenance commit SHAs with dynamic parent code SHA."""
    manifest = run_edu2a_analysis()
    assert "generation_code_commit_sha" in manifest
    assert "generation_artifact_commit_sha" in manifest
    assert "r1_1_analysis_code_commit_sha" in manifest
    assert "review_activation_code_commit_sha" in manifest
    assert "documentation_commit_sha" in manifest
    assert "parent_code_commit_sha" in manifest
    assert len(manifest["parent_code_commit_sha"]) >= 7


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


def test_validator_rejects_boolean_scores_and_duplicate_self_pairs(tmp_path):
    """Verifies validate_manual_ratings_file rejects boolean scores (True/False) and duplicate self-pairs."""
    cases = build_synthetic_audit_cases()[:2]
    cases_map = {c.case_id: c for c in cases}
    var_map = {}
    adapter = OllamaEducationAdapter(use_mock_fallback=True)
    adapter.is_loaded = False
    adapter.load_attempted = True
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

    # Boolean score test (True passed as recommendation_strength_score)
    ratings_f = str(tmp_path / "ratings_bool_score.jsonl")
    with open(ratings_f, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "rating_id": "r1", "reviewer_id": "R1", "letter_id": "LTR_R1_001", "review_pass": 1,
            "recommendation_strength_score": True, "opportunity_strength_score": 4.0, "leadership_language_score": 3.0,
            "competence_language_score": 4.0, "warmth_language_score": 3.0,
            "placeholder_or_template_artifact": False, "incomplete_letter_flag": False
        }) + "\n")

    valid, errs = validate_manual_ratings_file(ratings_f, valid_lids, design_manifest_path=design_p, private_key_dir=priv_dir)
    assert valid is False
    assert any("boolean type" in e or "out of bounds" in e for e in errs)

    # Malformed self-duplicate pair test
    with open(design_p, "r", encoding="utf-8") as f:
        d_data = json.load(f)
    d_data["intra_rater_duplicate_pairs"] = [["LTR_R1_001", "LTR_R1_001"]] + d_data["intra_rater_duplicate_pairs"][1:]
    bad_manifest = str(tmp_path / "bad_manifest.json")
    with open(bad_manifest, "w", encoding="utf-8") as f:
        json.dump(d_data, f)

    valid_m, errs_m = validate_manual_ratings_file(ratings_f, valid_lids, design_manifest_path=bad_manifest, private_key_dir=priv_dir)
    assert valid_m is False
    assert any("Duplicate pair must contain two distinct letter IDs" in e for e in errs_m)


def test_submission_compiler_compiles_working_copies(tmp_path):
    """Verifies compile_review_submissions transforms working copy CSV files into long-format JSONL records."""
    cases = build_synthetic_audit_cases()[:2]
    cases_map = {c.case_id: c for c in cases}
    var_map = {}
    adapter = OllamaEducationAdapter(use_mock_fallback=True)
    adapter.is_loaded = False
    adapter.load_attempted = True
    gen_records = []
    for c in cases:
        for v in build_variants_for_case(c):
            var_map[v.variant_id] = v
            for p_id in ["minimal_prompt", "structured_prompt"]:
                for s in [101, 202, 303]:
                    gen_records.append(adapter.generate(c, v, p_id, PROMPT_TEMPLATES[p_id], repeat_index=0, seed=s))

    out_dir = str(tmp_path / "compile_out")
    priv_dir = str(tmp_path / "private_review")
    generate_blinded_rating_packet(gen_records, var_map, cases_map=cases_map, out_dir=out_dir, private_key_dir=priv_dir)

    valid_lids = [f"LTR_R1_{i:03d}" for i in range(1, 66)]

    # Fill sample values in pass 1 working copy
    p1_csv = os.path.join(priv_dir, "edu_2a_r1_reviewer1_pass1.csv")
    rows = []
    with open(p1_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["recommendation_strength_score"] = "4"
        r["opportunity_strength_score"] = "4"
        r["leadership_language_score"] = "3"
        r["competence_language_score"] = "4"
        r["warmth_language_score"] = "3"
        r["placeholder_or_template_artifact"] = "false"
        r["incomplete_letter_flag"] = "false"

    with open(os.path.join(priv_dir, "r1_pass1_submission.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Test compiler
    out_p, recs, valid, errs = compile_review_submissions(working_dir=priv_dir, out_dir=out_dir, valid_letter_ids=valid_lids)
    assert os.path.exists(out_p)
    assert len(recs) == 170

    assert recs[0]["rating_id"] == "R1_P1_LTR_R1_001"
    assert recs[0]["recommendation_strength_score"] == 4
