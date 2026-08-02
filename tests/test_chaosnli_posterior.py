"""Unit tests for ChaosNLI posterior uncertainty and split-half estimation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from shadowspace.chaosnli.posterior import (
    compute_dirichlet_posteriors,
    compute_split_half_distributions,
    run_posterior_pipeline,
)


def test_compute_dirichlet_posteriors_shapes() -> None:
    # 5 items with known counts
    counts = np.array([
        [50, 30, 20],
        [100, 0, 0],
        [33, 33, 34],
        [0, 50, 50],
        [10, 80, 10],
    ])

    draws, summaries = compute_dirichlet_posteriors(counts, n_draws=500, seed=42)

    assert draws.shape == (5, 500, 3)
    # Probabilities sum to 1.0 per draw
    np.testing.assert_allclose(draws.sum(axis=-1), 1.0, atol=1e-6)

    assert len(summaries["posterior_entropy_mean"]) == 5
    assert len(summaries["p_max_majority"]) == 5

    # For item 1 [100, 0, 0], majority probability for entailment should be ~1.0
    assert summaries["p_majority_entailment"][1] > 0.99
    # For item 2 [33, 33, 34], no single majority dominates strongly
    assert summaries["p_max_majority"][2] < 0.60


def test_compute_split_half_distributions() -> None:
    counts = np.array([
        [60, 40, 0],
        [10, 10, 80],
    ])

    p_h1, p_h2 = compute_split_half_distributions(counts, seed=42)

    assert p_h1.shape == (2, 3)
    assert p_h2.shape == (2, 3)
    np.testing.assert_allclose(p_h1.sum(axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(p_h2.sum(axis=-1), 1.0, atol=1e-5)


def test_run_posterior_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        canon_file = tmp_path / "canonical_items.parquet"

        # Create dummy canonical parquet
        df = pl.DataFrame({
            "object_id": ["item_1", "item_2"],
            "human_count_entailment": [70, 20],
            "human_count_neutral": [20, 70],
            "human_count_contradiction": [10, 10],
        })
        df.write_parquet(canon_file)

        res = run_posterior_pipeline(
            canonical_parquet=canon_file,
            output_dir=tmp_path,
            n_draws=100,
            seed=42,
        )

        assert res["n_items"] == 2
        assert Path(res["output_path"]).exists()

        res_df = pl.read_parquet(res["output_path"])
        assert "posterior_entropy_mean" in res_df.columns
        assert "p_max_majority" in res_df.columns
