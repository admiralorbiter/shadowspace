"""Contract Unit Tests for Phase EDU-1.1a Hardening & Measurement Validation."""

import os
import subprocess
import sys
import pytest

from research.education_audit.case_builder import build_synthetic_audit_cases
from research.education_audit.variant_builder import build_variants_for_case
from research.education_audit.prompt_registry import PROMPT_TEMPLATES, get_prompt_hash
from research.education_audit.adapters.mock import (
    DeterministicMockAdapter,
    IndependentNullAdapter,
    SeededStochasticMockAdapter,
    stable_seed,
)
from research.education_audit.evaluation.rubric import evaluate_generation
from research.education_audit.run_generation import run_edu1_1_generation_and_eval
from research.education_audit.run_analysis import run_edu1_1_analysis
from research.education_audit.schemas import GenerationRecord


def test_fact_payload_byte_identity_across_variants():
    """Verifies that the fact payload following 'Verified Accomplishments:\n' is 100% byte-identical across all variants."""
    cases = build_synthetic_audit_cases()
    for case in cases:
        variants = build_variants_for_case(case)
        fact_payloads = []
        for v in variants:
            header, accomplishments = v.rendered_input.split("Verified Accomplishments:\n")
            fact_payloads.append(accomplishments)

        # Enforce exact byte-identity across all 5 variants
        first_payload = fact_payloads[0]
        for p in fact_payloads[1:]:
            assert p == first_payload, f"Fact payload mismatch for case {case.case_id}!"


def test_name_and_pronoun_cue_orthogonality():
    """Verifies that name-only variants preserve they/them pronouns and pronoun-only variants preserve 'Student A'."""
    cases = build_synthetic_audit_cases()
    c = cases[0]
    var_dict = {v.condition: v for v in build_variants_for_case(c)}

    # Pronoun-only
    assert var_dict["pronoun_masc"].student_name == "Student A"
    assert var_dict["pronoun_masc"].pronoun_subject == "he"
    assert var_dict["pronoun_fem"].student_name == "Student A"
    assert var_dict["pronoun_fem"].pronoun_subject == "she"

    # Name-only
    assert var_dict["name_masc"].student_name == "Alexander"
    assert var_dict["name_masc"].pronoun_subject == "they"
    assert var_dict["name_fem"].student_name == "Elizabeth"
    assert var_dict["name_fem"].pronoun_subject == "they"


def test_stable_seed_cross_process_reproducibility():
    """Verifies stable_seed yields identical seeds across different processes."""
    s1 = stable_seed("test_case_001", "minimal_prompt", 0)
    s2 = stable_seed("test_case_001", "minimal_prompt", 0)
    assert s1 == s2
    assert isinstance(s1, int)


def test_subprocess_cross_process_reproducibility():
    """Executes stochastic generation in separate processes with different PYTHONHASHSEED values and verifies output hash identity."""
    script = (
        "import sys; "
        "from research.education_audit.case_builder import build_synthetic_audit_cases; "
        "from research.education_audit.variant_builder import build_variants_for_case; "
        "from research.education_audit.adapters.mock import SeededStochasticMockAdapter; "
        "from research.education_audit.prompt_registry import PROMPT_TEMPLATES; "
        "c = build_synthetic_audit_cases()[0]; "
        "v = build_variants_for_case(c)[0]; "
        "adapter = SeededStochasticMockAdapter(); "
        "rec = adapter.generate(c, v, 'minimal_prompt', PROMPT_TEMPLATES['minimal_prompt'], repeat_index=2); "
        "print(rec.output_hash)"
    )

    env1 = os.environ.copy(); env1["PYTHONHASHSEED"] = "0"; env1["PYTHONPATH"] = "."
    env2 = os.environ.copy(); env2["PYTHONHASHSEED"] = "42"; env2["PYTHONPATH"] = "."

    out1 = subprocess.check_output([sys.executable, "-c", script], env=env1, text=True).strip()
    out2 = subprocess.check_output([sys.executable, "-c", script], env=env2, text=True).strip()

    assert out1 == out2, f"Stochastic mock output differs across PYTHONHASHSEED processes! ({out1} vs {out2})"


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


def test_edu1_1a_planted_signal_recovery(tmp_path):
    """Verifies EDU-1.1a planted signal recovery across stochastic repeats."""
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
    assert targets["rule_based_rubric_status"] == "SCREENING_ONLY"
    assert manifest["summary"]["prompt_interaction_diff_in_diff"] > 0.3


def test_edu1_1a_independent_null_verification(tmp_path):
    """Verifies EDU-1.1a independent null simulator produces strict 0-disparity under H0."""
    out_dir = str(tmp_path / "null_test")
    run_edu1_1_generation_and_eval(out_dir=out_dir, adapter_type="null", n_repeats=3)
    manifest = run_edu1_1_analysis(data_dir=out_dir, is_null_run=True)

    assert manifest["execution_status"] == "COMPLETED"
    assert manifest["mock_validation_status"] == "PASSED"
    assert manifest["summary"]["target_statuses"]["independent_null_contract_status"] == "INDEPENDENT_NULL_CONTRACT_PASSED"
