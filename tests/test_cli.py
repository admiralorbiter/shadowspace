"""Tests for Shadowspace CLI commands."""

from __future__ import annotations

import tempfile
from pathlib import Path

from shadowspace.cli import main


def test_cli_generate_synthetic_and_validate() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = str(Path(tmp_dir) / "synth-v1")
        # 1. Generate synthetic bundle
        exit_code_gen = main(
            [
                "generate",
                "synthetic",
                "--classes",
                "4",
                "--seed",
                "20260801",
                "--samples",
                "200",
                "--output",
                out_dir,
            ]
        )
        assert exit_code_gen == 0

        # 2. Validate bundle
        exit_code_val = main(["validate-bundle", out_dir])
        assert exit_code_val == 0


def test_cli_generate_calibration_and_validate() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = str(Path(tmp_dir) / "calib-v1")
        # 1. Generate calibration bundle
        exit_code_gen = main(["generate", "calibration", "--output", out_dir])
        assert exit_code_gen == 0

        # 2. Validate bundle
        exit_code_val = main(["validate-bundle", out_dir])
        assert exit_code_val == 0


def test_cli_validate_nonexistent_bundle_returns_nonzero() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        nonexistent = str(Path(tmp_dir) / "does-not-exist")
        exit_code_val = main(["validate-bundle", nonexistent])
        assert exit_code_val == 1
