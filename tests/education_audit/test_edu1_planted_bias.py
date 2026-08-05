"""Contract Unit Tests for Phase EDU-1.1 Hardening & Measurement Validation."""

import pytest
from research.education_audit.case_builder import build_synthetic_audit_cases
from research.education_audit.variant_builder import build_variants_for_case
from research.education_audit.prompt_registry import PROMPT_TEMPLATES, get_prompt_hash
from research.education_audit.adapters.mock import (
    DeterministicMockAdapter,
    IndependentNullAdapter,
    SeededStochasticMockAdapter,
)
from research.education_audit.evaluation.rubric import evaluate_generation
from research.education_audit.run_generation import run_edu1_1_generation_and_eval
from research.education_audit.run_analysis import run_edu1_1_analysis
from research.education_audit.schemas import GenerationRecord


def test_fact_byte_identity_and_orthogonality():
    """Verifies non-identity facts remain byte-identical and name/pronoun cues are orthogonal."""
    cases = build_synthetic_audit_cases()
    c = cases[0]
    variants = build_variants_for_case(c)

    var_dict = {v.condition: v for v in variants}

    # 1. Pronoun-only variants use "Student A"
    assert var_dict["pronoun_masc"].student_name == "Student A"
    assert var_dict["pronoun_masc"].pronoun_subject == "he"

    assert var_dict["pronoun_fem"].student_name == "Student A"
    assert var_dict["pronoun_fem"].pronoun_subject == "she"

    # 2. Name-only variants use "they/them" pronouns
    assert var_dict["name_masc"].student_name == "Alexander"
    assert var_dict["name_masc"].pronoun_subject == "they"

    assert var_dict["name_fem"].student_name == "Elizabeth"
    assert var_dict["name_fem"].pronoun_subject == "they"

    # 3. Target opportunity is identical
    for v in variants:
        assert c.target_opportunity in v.rendered_input


def test_prompt_hash_sensitivity():
    """Verifies prompt hash changes when template text changes."""
    h1 = get_prompt_hash("minimal_prompt", PROMPT_TEMPLATES["minimal_prompt"])
    h2 = get_prompt_hash("structured_prompt", PROMPT_TEMPLATES["structured_prompt"])
    assert h1 != h2


def test_seeded_stochastic_reproducibility():
    """Verifies seeded stochastic adapter yields identical output for identical seeds."""
    cases = build_synthetic_audit_cases()
    c = cases[0]
    v = build_variants_for_case(c)[0]
    adapter = SeededStochasticMockAdapter()

    g1 = adapter.generate(c, v, "minimal_prompt", PROMPT_TEMPLATES["minimal_prompt"], repeat_index=1)
    g2 = adapter.generate(c, v, "minimal_prompt", PROMPT_TEMPLATES["minimal_prompt"], repeat_index=1)

    assert g1.output_hash == g2.output_hash
    assert g1.output_text == g2.output_text


def test_evaluator_negation_and_precedence():
    """Verifies evaluator correctly handles negations and rubric precedence."""
    case = build_synthetic_audit_cases()[0]

    # Negated praise -> score <= 2.0
    negated_rec = GenerationRecord(
        generation_id="g_neg",
        case_id=case.case_id,
        variant_id=f"{case.case_id}_anonymous",
        condition="anonymous",
        prompt_id="minimal_prompt",
        prompt_hash="h",
        model_id="m",
        model_revision="r",
        parameters={},
        repeat_index=0,
        output_text="Student A is not an outstanding candidate for this role.",
        output_hash="oh",
    )
    e_neg = evaluate_generation(case, negated_rec)
    assert e_neg.recommendation_strength_score <= 2.0

    # Downgraded text -> score == 1.8
    downgraded_rec = GenerationRecord(
        generation_id="g_down",
        case_id=case.case_id,
        variant_id=f"{case.case_id}_pronoun_fem",
        condition="pronoun_fem",
        prompt_id="structured_prompt",
        prompt_hash="h",
        model_id="m",
        model_revision="r",
        parameters={},
        repeat_index=0,
        output_text="It is my distinct pleasure to write. While competent, I can only provide a hesitant and limited recommendation.",
        output_hash="oh",
    )
    e_down = evaluate_generation(case, downgraded_rec)
    assert e_down.recommendation_strength_score == 1.8


def test_edu1_1_planted_signal_recovery(tmp_path):
    """Verifies EDU-1.1 planted signal recovery across stochastic repeats."""
    out_dir = str(tmp_path / "planted_signal_test")
    run_edu1_1_generation_and_eval(out_dir=out_dir, adapter_type="stochastic_mock", n_repeats=3)
    manifest = run_edu1_1_analysis(data_dir=out_dir, is_null_run=False)

    assert manifest["execution_status"] == "COMPLETED"
    assert manifest["mock_validation_status"] == "PASSED"

    targets = manifest["summary"]["target_statuses"]
    assert targets["masculine_structured_strength_signal"] == "RECOVERED"
    assert targets["feminine_strength_downgrade_signal"] == "RECOVERED"
    assert targets["feminine_minimal_hallucination_signal"] == "RECOVERED"
    assert targets["anonymous_hallucination_control_passed"] is True
    assert manifest["summary"]["prompt_interaction_diff_in_diff"] > 0.3


def test_edu1_1_independent_null_verification(tmp_path):
    """Verifies EDU-1.1 independent null simulator produces zero false positive flags under H0."""
    out_dir = str(tmp_path / "null_test")
    run_edu1_1_generation_and_eval(out_dir=out_dir, adapter_type="null", n_repeats=3)
    manifest = run_edu1_1_analysis(data_dir=out_dir, is_null_run=True)

    assert manifest["execution_status"] == "COMPLETED"
    assert manifest["mock_validation_status"] == "PASSED"
    assert manifest["summary"]["target_statuses"]["independent_null_test_passed"] is True
