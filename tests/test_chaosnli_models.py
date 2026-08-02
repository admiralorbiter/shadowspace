"""Unit tests for ChaosNLI model loading, temperature scaling, and topology evaluation."""

from __future__ import annotations

import numpy as np
import polars as pl

from shadowspace.chaosnli.models import compute_model_probabilities
from shadowspace.chaosnli.model_topology import evaluate_hypothesis2_temperature_scaling, evaluate_model_topology_recovery


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
    df = pl.DataFrame({
        "object_id": [f"item_{i}" for i in range(10)],
        "human_p_entailment": [0.8, 0.7, 0.1, 0.2, 0.5, 0.9, 0.1, 0.3, 0.4, 0.6],
        "human_p_neutral": [0.1, 0.2, 0.8, 0.7, 0.5, 0.0, 0.1, 0.3, 0.4, 0.3],
        "human_p_contradiction": [0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.8, 0.4, 0.2, 0.1],
    })

    model_results = {
        "mock-model": {
            "object_ids": np.array([f"item_{i}" for i in range(10)]),
            "logits": np.random.default_rng(42).normal(size=(10, 3)).astype(np.float32),
        }
    }

    evals = evaluate_model_topology_recovery(model_results, canonical_items_path=df, k=3, qnx_hh_soft=0.8)
    assert "mock-model" in evals
    assert "qnx_soft_hm" in evals["mock-model"]
    assert evals["mock-model"]["h1_confirmed"] is True

    h2_curves = evaluate_hypothesis2_temperature_scaling(model_results, df, temperatures=[0.5, 1.0, 2.0], k=3)
    assert "mock-model" in h2_curves
    assert len(h2_curves["mock-model"]) == 3
