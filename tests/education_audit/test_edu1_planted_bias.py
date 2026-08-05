"""Unit test suite for EDU-1 Planted-Bias Validation."""

import os
import pytest
from research.education_audit.case_builder import build_synthetic_audit_cases
from research.education_audit.variant_builder import build_variants_for_case
from research.education_audit.adapters.mock import MockEducationAdapter
from research.education_audit.evaluation.rubric import evaluate_generation
from research.education_audit.run_generation import run_edu1_generation_and_eval
from research.education_audit.run_analysis import run_edu1_analysis


def test_build_synthetic_audit_cases():
    """Verifies 8 synthetic audit cases across 4 domains."""
    cases = build_synthetic_audit_cases()
    assert len(cases) == 8
    domains = {c.domain for c in cases}
    assert domains == {"technology", "math_data", "humanities", "leadership"}


def test_build_variants_for_case():
    """Verifies 5 counterfactual identity variants per case."""
    cases = build_synthetic_audit_cases()
    variants = build_variants_for_case(cases[0])
    assert len(variants) == 5
    conditions = {v.condition for v in variants}
    assert conditions == {"anonymous", "pronoun_masc", "pronoun_fem", "name_masc", "name_fem"}


def test_edu1_planted_bias_recovery(tmp_path):
    """Verifies EDU-1 pipeline generation, evaluation, and planted bias recovery."""
    out_dir = str(tmp_path / "edu_1_test")
    run_edu1_generation_and_eval(out_dir=out_dir, n_repeats=1)
    manifest = run_edu1_analysis(data_dir=out_dir)

    assert manifest["execution_status"] == "COMPLETED"
    assert manifest["mock_validation_status"] == "PASSED"
    assert manifest["summary"]["planted_signal_recovered"] is True
    assert manifest["summary"]["planted_null_verified"] is True
