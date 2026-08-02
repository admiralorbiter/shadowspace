"""Tests for ChaosNLI CLI sub-commands."""

from __future__ import annotations

import json
from unittest.mock import patch
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
    subcommands = [
        "audit-joins",
        "human-posterior",
        "build-spaces",
        "compute-neighbors",
        "compare-graphs",
        "analyze",
        "select-cases",
        "render-packets",
        "build-bundle",
        "report",
    ]
    for sub in subcommands:
        ret = main(["chaosnli", sub])
        assert ret == 0, f"Subcommand {sub} failed"


def test_chaosnli_subcommands_with_args(capsys) -> None:
    ret = main(["chaosnli", "predict", "--model", "roberta-large"])
    assert ret == 0
    ret = main(["chaosnli", "calibrate", "--model", "roberta-large"])
    assert ret == 0
    ret = main(["chaosnli", "import-codings", "dummy.csv"])
    assert ret == 0
    ret = main(["chaosnli", "verify-release", "artifacts/releases/v1"])
    assert ret == 0
