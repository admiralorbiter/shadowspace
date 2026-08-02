"""Unit tests for ChaosNLI acquisition and normalization modules."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from shadowspace.chaosnli.acquisition import acquire_sources, compute_sha256, verify_source_checksums
from shadowspace.chaosnli.normalize import compute_entropy_bits, normalize_dataset, normalize_record


def test_compute_entropy_bits() -> None:
    # Uniform 3-class distribution entropy = log2(3) approx 1.58496
    h_uniform = compute_entropy_bits(1 / 3, 1 / 3, 1 / 3)
    assert math.isclose(h_uniform, math.log2(3), rel_tol=1e-4)

    # Deterministic distribution entropy = 0.0
    h_deterministic = compute_entropy_bits(1.0, 0.0, 0.0)
    assert h_deterministic == 0.0


def test_normalize_record_valid() -> None:
    raw_item = {
        "uid": "snli_test_101",
        "premise": "A man is running.",
        "hypothesis": "A man is outdoors.",
        "genre": "caption",
        "old_label": "entailment",
        "labels": ["entailment"] * 60 + ["neutral"] * 40,
    }

    norm = normalize_record(raw_item, "chaosnli_snli")
    assert norm is not None
    assert norm["object_id"] == "chaosnli_snli_snli_test_101"
    assert norm["human_count_entailment"] == 60
    assert norm["human_count_neutral"] == 40
    assert norm["human_count_contradiction"] == 0
    assert norm["human_majority_label"] == "entailment"
    assert norm["human_majority_count"] == 60
    assert norm["human_agreement_rate"] == 0.60
    assert norm["has_zero_count"] is True


def test_acquisition_and_normalization_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw_dir = tmp_path / "raw"
        proc_dir = tmp_path / "processed"
        manifest_path = tmp_path / "sources.lock.yaml"

        raw_dir.mkdir(parents=True)

        # Create dummy jsonl raw source files
        snli_item = {
            "uid": "s001",
            "premise": "Two dogs play.",
            "hypothesis": "Animals are playing.",
            "old_label": "entailment",
            "labels": ["entailment"] * 70 + ["neutral"] * 30,
        }
        mnli_item = {
            "pairID": "m002",
            "premise": "It is raining.",
            "hypothesis": "It is dry outside.",
            "old_label": "contradiction",
            "labels": ["contradiction"] * 90 + ["neutral"] * 10,
        }

        with open(raw_dir / "chaosNLI_snli.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(snli_item) + "\n")

        with open(raw_dir / "chaosNLI_mnli_m.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(mnli_item) + "\n")

        with open(raw_dir / "chaosNLI_alphanli.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"uid": "a001"}) + "\n")

        # Test acquisition
        res = acquire_sources(raw_dir=raw_dir, manifest_path=manifest_path, force=False)
        assert manifest_path.exists()
        assert "snli" in res["files"]
        assert verify_source_checksums(raw_dir=raw_dir, manifest_path=manifest_path) is True

        # Test normalization
        norm_res = normalize_dataset(raw_dir=raw_dir, output_dir=proc_dir)
        assert norm_res["total_items"] == 2
        assert Path(norm_res["output_path"]).exists()
