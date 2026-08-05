"""Regression unit tests covering audit review findings."""

import pytest
import polars as pl
import numpy as np
from shadowspace.ambiguity_atlas.pair_index import (
    source_family,
    compute_source_splits,
    find_approximate_doppelgaenger_pairs,
)
from shadowspace.ambiguity_atlas.posterior import audit_pair_posterior_stability
from shadowspace.ambiguity_atlas.schemas import validate_canonical_df, validate_oof_df


def test_source_family_normalization_and_mutually_exclusive_splits():
    """Verify source family normalization handles chaosnli_snli and chaosnli_mnli without overlapping substrings."""
    assert source_family("snli") == "snli"
    assert source_family("chaosnli_snli") == "snli"
    assert source_family("chaosnli-snli") == "snli"
    assert source_family("mnli") == "mnli"
    assert source_family("chaosnli_mnli") == "mnli"
    assert source_family("chaosnli-mnli") == "mnli"
    
    with pytest.raises(ValueError):
        source_family("invalid_dataset")

    ds_a = ["chaosnli_snli", "chaosnli_mnli", "chaosnli_snli"]
    ds_b = ["chaosnli_snli", "chaosnli_mnli", "chaosnli_mnli"]
    
    splits = compute_source_splits(ds_a, ds_b)
    assert splits["within_snli"] == 1
    assert splits["within_mnli"] == 1
    assert splits["cross_source"] == 1
    assert sum(splits.values()) == 3


def test_approximate_pair_metadata_alignment_after_swap():
    """Verify entropy_a and entropy_b stay aligned with item_a and item_b after object_id swap."""
    data = [
        {
            "object_id": "z_item",
            "source_dataset": "chaosnli_snli",
            "premise": "Premise Z",
            "hypothesis": "Hypo Z",
            "human_count_entailment": 60,
            "human_count_neutral": 30,
            "human_count_contradiction": 10,
            "human_p_entailment": 0.6,
            "human_p_neutral": 0.3,
            "human_p_contradiction": 0.1,
            "human_entropy_bits": 1.295,
            "human_majority_label": "entailment",
        },
        {
            "object_id": "a_item",
            "source_dataset": "chaosnli_snli",
            "premise": "Premise A",
            "hypothesis": "Hypo A",
            "human_count_entailment": 60,
            "human_count_neutral": 10,
            "human_count_contradiction": 30,
            "human_p_entailment": 0.6,
            "human_p_neutral": 0.1,
            "human_p_contradiction": 0.3,
            "human_entropy_bits": 1.295,
            "human_majority_label": "entailment",
        },
    ]
    df = pl.DataFrame(data)
    df_approx = find_approximate_doppelgaenger_pairs(df, max_conf_diff=0.05, max_entropy_diff=0.05)
    
    assert df_approx.height == 1
    row = df_approx.to_dicts()[0]
    
    # After swap, object_id_a must be 'a_item' and object_id_b must be 'z_item'
    assert row["object_id_a"] == "a_item"
    assert row["object_id_b"] == "z_item"
    assert row["entropy_a"] == 1.295
    assert row["entropy_b"] == 1.295
    assert "is_pareto_optimal" in row


def test_posterior_fixed_majority_coordinate_system():
    """Verify posterior stability uses original fixed majority_idx M0."""
    counts_a = np.array([60, 30, 10])
    counts_b = np.array([60, 10, 30])
    
    res = audit_pair_posterior_stability(
        counts_a, counts_b,
        majority_idx=0,
        pair_id="test_pair",
        n_draws=200,
    )
    
    assert "prob_both_retain_original_majority" in res
    assert "prob_joint_collision" in res
    assert 0.0 <= res["prob_joint_collision"] <= 1.0


def test_preflight_schema_validation_invariants():
    """Verify preflight schema validator detects invalid bounds and non-summing probabilities."""
    invalid_data = [
        {
            "object_id": "item_1",
            "source_dataset": "chaosnli_snli",
            "premise": "P1",
            "hypothesis": "H1",
            "human_count_entailment": 50,
            "human_count_neutral": 30,
            "human_count_contradiction": 20,
            "human_p_entailment": 0.7,  # Invalid sum (0.7 + 0.3 + 0.2 = 1.2)
            "human_p_neutral": 0.3,
            "human_p_contradiction": 0.2,
            "human_entropy_bits": 1.485,
            "human_majority_label": "entailment",
        }
    ]
    df = pl.DataFrame(invalid_data)
    with pytest.raises(ValueError, match="do not sum to 1.0"):
        validate_canonical_df(df)
