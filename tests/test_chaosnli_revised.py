"""Unit tests for revised tie audit, soft Q_NX, and profile graph modules."""

from __future__ import annotations

import numpy as np
import polars as pl

from shadowspace.chaosnli.audit_ties import run_multiplicity_and_tie_audit
from shadowspace.chaosnli.distances import compute_hellinger_matrix
from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights, compute_soft_qnx
from shadowspace.chaosnli.profile_graph import analyze_level2_profile_heterogeneity, build_level1_profile_graph


def test_neighbors_soft_properties() -> None:
    p = np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.1, 0.8, 0.1],
    ])
    d = compute_hellinger_matrix(p)
    w = compute_soft_neighborhood_weights(d, k=2)

    assert w.shape == (4, 4)
    # Sum of weights for each node must equal k=2
    np.testing.assert_allclose(w.sum(axis=1), 2.0, atol=1e-5)

    # For node 0, nodes 1 and 2 are tied at distance 0.0. Each gets 1.0 weight
    np.testing.assert_allclose(w[0, 1], 1.0)
    np.testing.assert_allclose(w[0, 2], 1.0)

    # Soft Q_NX with identical weights must be 1.0
    qnx_soft, local_o = compute_soft_qnx(w, w, k=2)
    assert qnx_soft == 1.0
    assert len(local_o) == 4


def test_multiplicity_and_tie_audit() -> None:
    df = pl.DataFrame({
        "object_id": ["item_1", "item_2", "item_3", "item_4"],
        "source_dataset": ["chaosnli_snli", "chaosnli_snli", "chaosnli_mnli", "chaosnli_mnli"],
        "human_count_entailment": [50, 50, 50, 10],
        "human_count_neutral": [50, 50, 50, 80],
        "human_count_contradiction": [0, 0, 0, 10],
        "human_p_entailment": [0.5, 0.5, 0.5, 0.1],
        "human_p_neutral": [0.5, 0.5, 0.5, 0.8],
        "human_p_contradiction": [0.0, 0.0, 0.0, 0.1],
        "human_entropy_bits": [1.0, 1.0, 1.0, 0.92],
    })

    p = df.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    d = compute_hellinger_matrix(p)

    res = run_multiplicity_and_tie_audit(df, d, k=2, n_permutations=3)

    assert res["n_items"] == 4
    assert res["unique_profiles"] == 2
    assert res["items_in_non_singleton_profiles"] == 3
    assert res["max_profile_multiplicity"] == 3


def test_level1_profile_graph() -> None:
    df = pl.DataFrame({
        "object_id": ["item_1", "item_2", "item_3", "item_4"],
        "source_dataset": ["chaosnli_snli", "chaosnli_snli", "chaosnli_mnli", "chaosnli_mnli"],
        "human_count_entailment": [50, 50, 50, 10],
        "human_count_neutral": [50, 50, 50, 80],
        "human_count_contradiction": [0, 0, 0, 10],
        "human_p_entailment": [0.5, 0.5, 0.5, 0.1],
        "human_p_neutral": [0.5, 0.5, 0.5, 0.8],
        "human_p_contradiction": [0.0, 0.0, 0.0, 0.1],
        "human_entropy_bits": [1.0, 1.0, 1.0, 0.92],
    })

    level1_res = build_level1_profile_graph(df, metric="hellinger", k=1)
    assert level1_res["n_profiles"] == 2
    assert level1_res["n_total_items"] == 4

    level2_df = analyze_level2_profile_heterogeneity(df, level1_res["profile_df"])
    assert len(level2_df) == 1
    assert level2_df["frequency"][0] == 3
    assert level2_df["n_snli"][0] == 2
    assert level2_df["n_mnli"][0] == 1
