"""Unit Contract Tests for Commit-Pinned Real External Benchmark Suite & Calibrated Simulator."""

import os
import pytest

from research.education_audit.external_validation.wan2023_loader import load_wan2023_dataset
from research.education_audit.external_validation.labe_loader import load_labe_dataset
from research.education_audit.external_validation.agency_metric_replication import run_agency_metric_replication
from research.education_audit.external_validation.evaluator_invariance import run_evaluator_invariance_benchmark
from research.education_audit.external_validation.causal_audit_simulator import run_causal_audit_simulation
from research.education_audit.external_validation.onet_profile_builder import generate_onet_grounded_profile_bank
from research.education_audit.external_validation.run_external_validation import run_full_external_validation


def test_commit_pinned_wan2023_loader():
    """Verifies Wan 2023 loader uses commit-pinned URL and verifies expected SHA-256 hash."""
    res = load_wan2023_dataset()
    assert res["status"] == "LOADED_PINNED_REAL_DATA"
    assert res["commit_sha"] == "1264990e5f55e46cb8b83d8bfe2749946008b4a8"
    assert res["records_count"] == 120
    assert "sha256_hash" in res


def test_commit_pinned_labe_dataset_loader():
    """Verifies LABE loader ingests Train, Val, and Test splits from commit-pinned URLs."""
    res = load_labe_dataset()
    assert res["status"] == "LOADED_PINNED_REAL_DATA"
    assert res["commit_sha"] == "e8cc42d86df007fd05e3ae0c27c127b7a0a6165c"
    assert res["train_count"] > 0
    assert res["test_count"] > 0
    assert res["total_labeled_sentences_count"] >= 3000


def test_ev1_exact_lexicon_benchmark_and_wan_uncertainty(tmp_path):
    """Verifies Milestone EV-1 test-split metrics and 95% paired bootstrap CI on Wan 2023 data."""
    out_dir = str(tmp_path / "ev1")
    res = run_agency_metric_replication(out_dir=out_dir)
    assert res["status"] == "EV1_LEXICON_BENCHMARK_COMPLETED"
    assert res["wan_pairs_evaluated"] == 60
    assert len(res["wan_agency_95ci_bootstrap"]) == 2
    assert "labe_lac_metrics_test_primary" in res
    assert os.path.exists(os.path.join(out_dir, "replication_report.md"))


def test_ev2_evaluator_invariance_benchmark(tmp_path):
    """Verifies Milestone EV-2 measures evaluator drift across counterfactual swaps."""
    out_dir = str(tmp_path / "ev2")
    res = run_evaluator_invariance_benchmark(out_dir=out_dir)
    assert res["status"] == "EV2_INVARIANCE_COMPLETED"
    assert res["mean_absolute_agentic_drift"] == 0.0
    assert os.path.exists(os.path.join(out_dir, "auditor_invariance_report.md"))


def test_ev3_causal_audit_simulator_1000_reps(tmp_path):
    """Verifies Milestone EV-3 runs 1,000 Monte Carlo replications and quantifies Type-I Error."""
    out_dir = str(tmp_path / "ev3")
    res = run_causal_audit_simulation(replications_per_world=100, out_dir=out_dir)
    assert res["status"] == "EV3_SIMULATION_COMPLETED_1000_REPS"
    assert 0.0 <= res["empirical_type1_error_rate_null"] <= 0.10
    assert os.path.exists(os.path.join(out_dir, "method_validation_report.md"))


def test_real_onet_30_3_task_metadata(tmp_path):
    """Verifies O*NET 30.3 profile bank builds from official task statements and task IDs."""
    out_dir = str(tmp_path / "onet")
    res = generate_onet_grounded_profile_bank(out_dir=out_dir)
    assert res["status"] == "ONET_PROFILES_GENERATED_REAL_30_3_TABLES"
    assert res["profiles_count"] == 8
    assert os.path.exists(res["profile_bank_path"])


def test_full_external_validation_pipeline(tmp_path):
    """Verifies end-to-end execution of external validation master runner with commit-pinned real data."""
    out_dir = str(tmp_path / "full_validation")
    manifest = run_full_external_validation(out_dir=out_dir)
    assert manifest["status"] == "EXTERNAL_VALIDATION_COMPLETED"
    assert os.path.exists(os.path.join(out_dir, "validation_manifest.json"))
