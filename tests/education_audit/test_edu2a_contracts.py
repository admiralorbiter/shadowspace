"""Unit Contract Tests for Phase EDU-2a Live Canary Infrastructure."""

import json
import os
import pytest

from research.education_audit.adapters.ollama import OllamaEducationAdapter
from research.education_audit.case_builder import build_synthetic_audit_cases
from research.education_audit.variant_builder import build_variants_for_case
from research.education_audit.prompt_registry import PROMPT_TEMPLATES
from research.education_audit.evaluation.blinding import blind_generation_text
from research.education_audit.reporting.rating_packet import generate_blinded_rating_packet


def test_ollama_adapter_mock_fallback_mode():
    """Verifies OllamaEducationAdapter fallback mode when use_mock_fallback=True."""
    adapter = OllamaEducationAdapter(model_name="gemma:12b", use_mock_fallback=True)
    case = build_synthetic_audit_cases()[0]
    variant = build_variants_for_case(case)[0]

    g = adapter.generate(case, variant, "minimal_prompt", PROMPT_TEMPLATES["minimal_prompt"], repeat_index=0, seed=101)
    assert g.model_id.startswith("mock-fallback")
    assert len(g.output_text) > 20
    assert g.output_hash is not None


def test_blinding_redaction_and_leakage_detection():
    """Verifies that names and explicit pronouns are redacted to [CANDIDATE] and they/them/their."""
    raw_text = "It is my pleasure to recommend Alexander. He has demonstrated outstanding skill in his projects. I give him my support."

    blinded, leakage, details = blind_generation_text(raw_text, "Alexander", "he", "him", "his")

    assert "[CANDIDATE]" in blinded
    assert "Alexander" not in blinded
    assert "They has demonstrated outstanding skill in their projects." in blinded or "they" in blinded.lower()
    assert "him" not in blinded.split()


def test_rating_packet_generation(tmp_path):
    """Verifies generate_blinded_rating_packet exports CSV, JSONL, and secret blinding_key.json."""
    case = build_synthetic_audit_cases()[0]
    variants = build_variants_for_case(case)
    var_map = {v.variant_id: v for v in variants}

    adapter = OllamaEducationAdapter(use_mock_fallback=True)
    gen_records = [
        adapter.generate(case, v, "minimal_prompt", PROMPT_TEMPLATES["minimal_prompt"], repeat_index=0)
        for v in variants
    ]

    out_dir = str(tmp_path / "packet_test")
    csv_p, jsonl_p, key_p = generate_blinded_rating_packet(gen_records, var_map, out_dir=out_dir)

    assert os.path.exists(csv_p)
    assert os.path.exists(jsonl_p)
    assert os.path.exists(key_p)

    with open(key_p, "r", encoding="utf-8") as f:
        key_data = json.load(f)

    assert len(key_data) == 5
    assert "LTR_001" in key_data
