"""Unit tests for ChaosNLI model loading, temperature scaling, and topology evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from shadowspace.chaosnli.model_topology import (
    evaluate_hypothesis2_temperature_scaling,
    evaluate_model_topology_recovery,
)
from shadowspace.chaosnli.models import compute_model_probabilities, load_model_predictions


def test_temperature_scaling_softmax() -> None:
    logits = np.array([[2.0, 1.0, 0.0]])
    p_1 = compute_model_probabilities(logits, temperature=1.0)
    p_2 = compute_model_probabilities(logits, temperature=2.0)

    # Temperature 2.0 softens the distribution (higher entropy)
    ent_1 = -np.sum(p_1 * np.log2(p_1))
    ent_2 = -np.sum(p_2 * np.log2(p_2))

    assert ent_2 > ent_1
    np.testing.assert_allclose(p_1.sum(axis=1), 1.0)
    np.testing.assert_allclose(p_2.sum(axis=1), 1.0)


def test_model_topology_evaluation_mock() -> None:
    df = pl.DataFrame(
        {
            "object_id": [f"item_{i}" for i in range(10)],
            "human_p_entailment": [0.8, 0.7, 0.1, 0.2, 0.5, 0.9, 0.1, 0.3, 0.4, 0.6],
            "human_p_neutral": [0.1, 0.2, 0.8, 0.7, 0.5, 0.0, 0.1, 0.3, 0.4, 0.3],
            "human_p_contradiction": [0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.8, 0.4, 0.2, 0.1],
        }
    )

    model_results = {
        "mock-model": {
            "object_ids": np.array([f"item_{i}" for i in range(10)]),
            "logits": np.random.default_rng(42).normal(size=(10, 3)).astype(np.float32),
        }
    }

    evals = evaluate_model_topology_recovery(
        model_results, canonical_items_path=df, k=3, qnx_hh_soft=0.8
    )
    assert "mock-model" in evals
    assert "qnx_soft_hm" in evals["mock-model"]
    assert evals["mock-model"]["all_point_estimates_below_human"] is True

    h2_curves = evaluate_hypothesis2_temperature_scaling(
        model_results, df, temperatures=[0.5, 1.0, 2.0], k=3
    )
    assert "mock-model" in h2_curves
    assert len(h2_curves["mock-model"]) == 3


def _write_canonical_items(path: Path) -> None:
    pl.DataFrame(
        {
            "object_id": ["chaosnli_snli_item-a", "chaosnli_mnli_item-b"],
            "human_p_entailment": [0.7, 0.1],
            "human_p_neutral": [0.2, 0.2],
            "human_p_contradiction": [0.1, 0.7],
        }
    ).write_parquet(path)


def test_model_loader_fails_closed_when_real_artifact_is_missing(tmp_path) -> None:
    canonical_path = tmp_path / "items.parquet"
    _write_canonical_items(canonical_path)

    with pytest.raises(FileNotFoundError, match="synthetic fallback is disabled"):
        load_model_predictions(
            tmp_path / "missing.json",
            canonical_path,
            expected_model_names=["mock-model"],
            expected_sha256=None,
        )


def test_model_loader_allows_only_explicit_synthetic_fallback(tmp_path) -> None:
    canonical_path = tmp_path / "items.parquet"
    _write_canonical_items(canonical_path)

    models = load_model_predictions(
        tmp_path / "missing.json",
        canonical_path,
        allow_synthetic=True,
        expected_model_names=["mock-model"],
        expected_sha256=None,
    )

    assert set(models) == {"mock-model"}
    assert models["mock-model"]["logits"].shape == (2, 3)


def test_model_loader_rejects_incomplete_real_artifact(tmp_path) -> None:
    canonical_path = tmp_path / "items.parquet"
    _write_canonical_items(canonical_path)
    model_path = tmp_path / "models.json"
    model_path.write_text(
        json.dumps(
            {
                "mock-model": {
                    "item-a": {"logits": [1.0, 0.0, -1.0]},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing 1 canonical item IDs"):
        load_model_predictions(
            model_path,
            canonical_path,
            expected_model_names=["mock-model"],
            expected_sha256=None,
        )
