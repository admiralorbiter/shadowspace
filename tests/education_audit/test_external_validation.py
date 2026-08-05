"""Unit Contract Tests for Real External Evidence & Synthetic Validation Suite."""

import os
import pytest

from research.education_audit.external_validation.wan2023_loader import load_wan2023_dataset
from research.education_audit.external_validation.labe_loader import load_labe_dataset
from research.education_audit.external_validation.agency_metric_replication import run_agency_metric_replication
from research.education_audit.external_validation.evaluator_invariance import run_evaluator_invariance_benchmark
from research.education_audit.external_validation.causal_audit_simulator import run_causal_audit_simulation
from research.education_audit.external_validation.onet_profile_builder import generate_onet_grounded_profile_bank
from research.education_audit.external_validation.run_external_validation import run_full_external_validation


def test_real_wan2023_dataset_loader():
    """Verifies Wan 2023 real dataset loader ingests published ChatGPT letters."""
    res = load_wan2023_dataset()
    assert res["status"] == "LOADED_REAL_DATA"
    assert res["records_count"] == 120
    assert "sha256_hash" in res


def test_real_labe_dataset_loader():
    """Verifies LABE real dataset loader ingests published LAC train and test CSVs."""
    res = load_labe_dataset()
    assert res["status"] == "LOADED_REAL_DATA"
    assert res["labeled_sentences_count"] >= 1000
    assert "train_sha256_hash" in res


def test_ev1_real_agency_metric_replication(tmp_path):
    """Verifies Milestone EV-1 calculates agency deltas and precision/recall on real datasets."""
    out_dir = str(tmp_path / "ev1")
    res = run_agency_metric_replication(out_dir=out_dir)
    assert res["status"] == "EV1_REPL_COMPLETED_REAL_DATA"
    assert res["wan_pairs_evaluated"] > 0
    assert 0.0 <= res["lexicon_precision_on_labe_lac"] <= 1.0
    assert 0.0 <= res["lexicon_recall_on_labe_lac"] <= 1.0
    assert os.path.exists(os.path.join(out_dir, "replication_report.md"))


def test_ev2_evaluator_invariance_benchmark(tmp_path):
    """Verifies Milestone EV-2 measures evaluator drift across counterfactual swaps."""
    out_dir = str(tmp_path / "ev2")
    res = run_evaluator_invariance_benchmark(out_dir=out_dir)
    assert res["status"] == "EV2_INVARIANCE_COMPLETED"
    assert res["mean_absolute_agentic_drift"] == 0.0
    assert os.path.exists(os.path.join(out_dir, "auditor_invariance_report.md"))


def test_ev3_causal_audit_simulator(tmp_path):
    """Verifies Milestone EV-3 simulates 6 ground-truth worlds and checks Type-I error rate."""
    out_dir = str(tmp_path / "ev3")
    res = run_causal_audit_simulation(out_dir=out_dir)
    assert res["status"] == "EV3_SIMULATION_COMPLETED"
    assert res["type1_error_rate_null_world"] == 0.0
    assert os.path.exists(os.path.join(out_dir, "method_validation_report.md"))


def test_real_onet_30_3_profile_bank(tmp_path):
    """Verifies O*NET 30.3 profile bank generates profiles from real SOC codes and task statements."""
    out_dir = str(tmp_path / "onet")
    res = generate_onet_grounded_profile_bank(out_dir=out_dir)
    assert res["status"] == "ONET_PROFILES_GENERATED_REAL_30_3"
    assert res["profiles_count"] == 8
    assert res["onet_release_version"] == "30.3"
    assert os.path.exists(res["profile_bank_path"])


def test_full_external_validation_pipeline(tmp_path):
    """Verifies end-to-end execution of external validation master runner with real data."""
    out_dir = str(tmp_path / "full_validation")
    manifest = run_full_external_validation(out_dir=out_dir)
    assert manifest["status"] == "EXTERNAL_VALIDATION_COMPLETED"
    assert os.path.exists(os.path.join(out_dir, "validation_manifest.json"))
