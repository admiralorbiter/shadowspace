"""Unit Contract Tests for External Evidence & Synthetic Validation Suite (Milestones EV-1, EV-2, EV-3)."""

import os
import pytest

from research.education_audit.external_validation.wan2023_loader import load_wan2023_dataset
from research.education_audit.external_validation.labe_loader import load_labe_dataset
from research.education_audit.external_validation.agency_metric_replication import run_agency_metric_replication
from research.education_audit.external_validation.evaluator_invariance import run_evaluator_invariance_benchmark
from research.education_audit.external_validation.causal_audit_simulator import run_causal_audit_simulation, simulate_causal_world
from research.education_audit.external_validation.onet_profile_builder import generate_onet_grounded_profile_bank
from research.education_audit.external_validation.run_external_validation import run_full_external_validation


def test_wan2023_dataset_loader():
    """Verifies Wan 2023 benchmark loader imports 120 context-free and 6,028 context-based prompts."""
    res = load_wan2023_dataset()
    assert res["status"] == "LOADED"
    assert res["context_free_prompts_count"] == 120
    assert res["context_based_prompts_count"] == 6028


def test_labe_dataset_loader():
    """Verifies LABE benchmark loader imports labeled agency sentences."""
    res = load_labe_dataset()
    assert res["status"] == "LOADED"
    assert res["labeled_sentences_count"] == 4


def test_ev1_agency_metric_replication(tmp_path):
    """Verifies Milestone EV-1 calculates sign agreement and Spearman correlation."""
    out_dir = str(tmp_path / "ev1")
    res = run_agency_metric_replication(out_dir=out_dir)
    assert res["status"] == "EV1_REPL_COMPLETED"
    assert 0.0 <= res["sign_agreement_rate"] <= 1.0
    assert os.path.exists(os.path.join(out_dir, "replication_report.md"))


def test_ev2_evaluator_invariance_benchmark(tmp_path):
    """Verifies Milestone EV-2 measures evaluator drift across counterfactual swaps."""
    out_dir = str(tmp_path / "ev2")
    res = run_evaluator_invariance_benchmark(out_dir=out_dir)
    assert res["status"] == "EV2_INVARIANCE_COMPLETED"
    assert res["mean_absolute_agentic_drift"] == 0.0  # Dictionary evaluator is perfectly counterfactually invariant on identical text
    assert os.path.exists(os.path.join(out_dir, "auditor_invariance_report.md"))


def test_ev3_causal_audit_simulator(tmp_path):
    """Verifies Milestone EV-3 simulates 6 ground-truth worlds and checks Type-I error rate."""
    out_dir = str(tmp_path / "ev3")
    res = run_causal_audit_simulation(out_dir=out_dir)
    assert res["status"] == "EV3_SIMULATION_COMPLETED"
    assert res["type1_error_rate_null_world"] == 0.0
    assert os.path.exists(os.path.join(out_dir, "method_validation_report.md"))


def test_onet_profile_bank(tmp_path):
    """Verifies O*NET profile bank generates 12 grounded profiles across 6 domains."""
    out_dir = str(tmp_path / "onet")
    res = generate_onet_grounded_profile_bank(out_dir=out_dir)
    assert res["status"] == "ONET_PROFILES_GENERATED"
    assert res["profiles_count"] == 12
    assert os.path.exists(res["profile_bank_path"])


def test_full_external_validation_pipeline(tmp_path):
    """Verifies end-to-end execution of external validation master runner."""
    out_dir = str(tmp_path / "full_validation")
    manifest = run_full_external_validation(out_dir=out_dir)
    assert manifest["status"] == "EXTERNAL_VALIDATION_COMPLETED"
    assert os.path.exists(os.path.join(out_dir, "validation_manifest.json"))
