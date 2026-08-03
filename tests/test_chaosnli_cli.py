"""Tests for ChaosNLI CLI sub-commands."""

from __future__ import annotations

import json
from unittest.mock import patch
import numpy as np
import polars as pl
from shadowspace.cli import main


def test_chaosnli_fetch_cli(capsys) -> None:
    fake_acquired = {
        "files": {
            "snli": {"path": "data/chaosnli/raw/chaosNLI_snli.jsonl"},
            "mnli": {"path": "data/chaosnli/raw/chaosNLI_mnli_m.jsonl"},
        }
    }
    with patch("shadowspace.chaosnli.acquisition.acquire_sources", return_value=fake_acquired):
        ret = main(["chaosnli", "fetch"])
        assert ret == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["task"] == "chaosnli.fetch"
        assert data["status"] == "success"


def test_chaosnli_verify_sources_cli(capsys) -> None:
    with patch("shadowspace.chaosnli.acquisition.verify_source_checksums", return_value=True):
        ret = main(["chaosnli", "verify-sources"])
        assert ret == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["task"] == "chaosnli.verify-sources"
        assert data["valid"] is True


def test_chaosnli_normalize_cli(capsys) -> None:
    fake_summary = {"total_items": 100, "mean_entropy_bits": 1.2}
    with patch("shadowspace.chaosnli.normalize.normalize_dataset", return_value=fake_summary):
        ret = main(["chaosnli", "normalize"])
        assert ret == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["task"] == "chaosnli.normalize"


def test_chaosnli_subcommands_stub(capsys) -> None:
    implemented = {
        "human-posterior",
        "build-spaces",
        "compute-neighbors",
        "compare-graphs",
        "analyze",
    }
    stubs = [
        "audit-joins",
        "select-cases",
        "render-packets",
        "build-bundle",
        "report",
    ]
    with patch("shadowspace.chaosnli.posterior.run_posterior_pipeline", return_value={"status": "success"}), \
         patch("polars.read_parquet") as mock_read_parquet, \
         patch("shadowspace.chaosnli.text_embeddings.build_text_distance_space", return_value={"embedding_dim": 384}), \
         patch("shadowspace.chaosnli.joint_spaces.compute_joint_distance_matrix") as mock_joint, \
         patch("shadowspace.chaosnli.graph_metrics.compute_human_split_half_reliability", return_value={"median_soft_qnx": 0.0426}), \
         patch("shadowspace.chaosnli.models.load_model_predictions", return_value={}), \
         patch("shadowspace.chaosnli.models.build_canonical_models_table") as mock_build_table, \
         patch("shadowspace.chaosnli.model_topology.evaluate_model_topology_recovery", return_value={}), \
         patch("numpy.save"), \
         patch("numpy.load", return_value=np.zeros((100, 100))):

        fake_df = pl.DataFrame({
            "object_id": [f"item_{i}" for i in range(100)],
            "human_p_entailment": [0.5] * 100,
            "human_p_neutral": [0.3] * 100,
            "human_p_contradiction": [0.2] * 100,
            "human_count_entailment": [50] * 100,
            "human_count_neutral": [30] * 100,
            "human_count_contradiction": [20] * 100,
        })
        mock_read_parquet.return_value = fake_df
        mock_joint.return_value = fake_df.select(["human_p_entailment", "human_p_neutral"]).to_numpy()

        for sub in implemented:
            ret = main(["chaosnli", sub])
            assert ret == 0, f"Implemented subcommand {sub} failed"

        for sub in stubs:
            ret = main(["chaosnli", sub])
            assert ret == 2, f"Stub subcommand {sub} should fail closed with code 2"


def test_chaosnli_subcommands_with_args(capsys) -> None:
    ret = main(["chaosnli", "predict", "--model", "roberta-large"])
    assert ret == 2
    ret = main(["chaosnli", "calibrate", "--model", "roberta-large"])
    assert ret == 2
    ret = main(["chaosnli", "import-codings", "dummy.csv"])
    assert ret == 2
    ret = main(["chaosnli", "verify-release", "artifacts/releases/v1"])
    assert ret == 2

