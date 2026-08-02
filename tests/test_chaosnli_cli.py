"""Tests for ChaosNLI CLI sub-commands."""

from __future__ import annotations

import json
from shadowspace.cli import main


def test_chaosnli_fetch_cli(capsys) -> None:
    ret = main(["chaosnli", "fetch"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["task"] == "chaosnli.fetch"
    assert data["status"] == "success"


def test_chaosnli_subcommands_stub(capsys) -> None:
    subcommands = [
        "verify-sources",
        "normalize",
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
