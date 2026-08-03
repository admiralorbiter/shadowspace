"""Unit tests for ChaosNLI text embeddings and joint space modules."""

from __future__ import annotations

import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import compute_hellinger_matrix
from shadowspace.chaosnli.joint_spaces import compute_joint_distance_matrix, evaluate_hypothesis7_joint_space
from shadowspace.chaosnli.text_embeddings import compute_text_cosine_distance_matrix, extract_text_embeddings


def test_text_embeddings_and_cosine_distance() -> None:
    df = pl.DataFrame({
        "premise": ["A dog is running in the park.", "Two cats are sleeping on the couch."],
        "hypothesis": ["An animal is outdoors.", "Felines are resting indoors."],
    })

    emb, method_name = extract_text_embeddings(df, method="tfidf-svd")
    assert emb.shape[0] == 2
    assert emb.shape[1] > 0

    d_txt = compute_text_cosine_distance_matrix(emb)
    assert d_txt.shape == (2, 2)
    np.testing.assert_allclose(np.diag(d_txt), 0.0, atol=1e-5)
    assert d_txt[0, 1] > 0.0


def test_joint_distance_matrix_blending() -> None:
    p = np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.1, 0.8, 0.1],
    ])
    d_op = compute_hellinger_matrix(p)

    emb = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    d_txt = compute_text_cosine_distance_matrix(emb)

    d_joint = compute_joint_distance_matrix(d_op, d_txt, lambda_weight=0.5)

    assert d_joint.shape == (3, 3)
    np.testing.assert_allclose(np.diag(d_joint), 0.0, atol=1e-5)
    # Nodes 0 and 1 had d_op = 0.0, but d_txt > 0.0, so d_joint > 0.0
    assert d_joint[0, 1] > 0.0


def test_hypothesis7_evaluation_mock() -> None:
    df = pl.DataFrame({
        "object_id": ["item_1", "item_2", "item_3", "item_4"],
        "human_p_entailment": [0.5, 0.5, 0.1, 0.1],
        "human_p_neutral": [0.5, 0.5, 0.8, 0.8],
        "human_p_contradiction": [0.0, 0.0, 0.1, 0.1],
    })

    d_op = compute_hellinger_matrix(df.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy())
    emb = np.random.default_rng(42).normal(size=(4, 16))
    d_txt = compute_text_cosine_distance_matrix(emb)

    res = evaluate_hypothesis7_joint_space(df, d_op, d_txt, lambdas=[0.0, 0.1, 0.5, 1.0], k=2)

    assert res["n_items"] == 4
    assert "lambda_evaluations" in res
    assert len(res["lambda_evaluations"]) == 4
