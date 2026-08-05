"""Unit Contract Tests for Phase EDU-2a-R1 Complete-Generation & Blind-Integrity Canary."""

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
from research.education_audit.schemas import GenerationRecord


def test_ollama_adapter_mock_fallback_mode(monkeypatch):
    """Verifies OllamaEducationAdapter fallback mode when ping_and_inspect fails and use_mock_fallback=True."""
    adapter = OllamaEducationAdapter(model_name="gemma3:12b", use_mock_fallback=True)
    monkeypatch.setattr(adapter, "ping_and_inspect", lambda: False)
    adapter.is_loaded = False  # Simulate offline service
    case = build_synthetic_audit_cases()[0]
    variant = build_variants_for_case(case)[0]

    g = adapter.generate(case, variant, "minimal_prompt", PROMPT_TEMPLATES["minimal_prompt"], repeat_index=0, seed=101)
    assert g.model_id.startswith("mock-fallback")


    assert len(g.output_text) > 20
    assert g.output_hash is not None


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
    case = build_synthetic_audit_cases()[0]
    variants = build_variants_for_case(case)
    var_map = {v.variant_id: v for v in variants}

    adapter = OllamaEducationAdapter(use_mock_fallback=True)
    gen_records = [
        adapter.generate(case, v, "minimal_prompt", PROMPT_TEMPLATES["minimal_prompt"], repeat_index=0)
        for v in variants
    ]

    out_dir = str(tmp_path / "packet_test")
    priv_key_dir = str(tmp_path / "private_review")
    csv_p, jsonl_p, key_p = generate_blinded_rating_packet(gen_records, var_map, out_dir=out_dir, private_key_dir=priv_key_dir)

    assert os.path.exists(csv_p)
    assert os.path.exists(jsonl_p)
    assert os.path.exists(key_p)

    with open(key_p, "r", encoding="utf-8") as f:
        key_data = json.load(f)

    # 5 base records + 5 duplicates = 10 total packet entries
    assert len(key_data) == 10
    assert "LTR_R1_001" in key_data


def test_strict_completion_gate_on_truncation(tmp_path):
    """Verifies that truncation count > 0 causes completion_integrity_status = FAILED and go_to_full_pilot = False."""
    out_dir = tmp_path / "truncation_test"
    out_dir.mkdir()

    # Create dummy screening evaluations
    with open(out_dir / "screening_evaluations.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "generation_id": "g1", "case_id": "c1", "variant_id": "v1", "condition": "anonymous",
            "prompt_id": "minimal_prompt", "recommendation_strength_score": 3.0, "hallucinations_per_100_words": 0.0
        }) + "\n")

    # Create dummy generations with truncation (done_reason == length)
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
    assert manifest["finding"] == "CANARY_PIPELINE_FAILURE_SURFACED"
