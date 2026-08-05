"""Unit tests for pair_index module."""

import pytest
import polars as pl
from shadowspace.ambiguity_atlas.pair_index import find_strict_doppelgaenger_pairs


def test_find_strict_doppelgaenger_pairs_synthetic():
    """Test exact doppelgänger pair discovery on synthetic dataframe."""
    data = [
        # Pair 1: Majority Entailment (60), Neutral 30, Contradiction 10 vs Neutral 10, Contradiction 30
        {
            "object_id": "item_1",
            "source_dataset": "snli",
            "premise": "A man is running.",
            "hypothesis": "A person is moving.",
            "human_count_entailment": 60,
            "human_count_neutral": 30,
            "human_count_contradiction": 10,
            "human_p_entailment": 0.6,
            "human_p_neutral": 0.3,
            "human_p_contradiction": 0.1,
            "human_entropy_bits": 1.2954,
            "human_majority_label": "entailment",
        },
        {
            "object_id": "item_2",
            "source_dataset": "mnli",
            "premise": "A dog barks.",
            "hypothesis": "A dog is noisy.",
            "human_count_entailment": 60,
            "human_count_neutral": 10,
            "human_count_contradiction": 30,
            "human_p_entailment": 0.6,
            "human_p_neutral": 0.1,
            "human_p_contradiction": 0.3,
            "human_entropy_bits": 1.2954,
            "human_majority_label": "entailment",
        },
        # Symmetric item (minority counts equal, 20 vs 20) -> should be excluded
        {
            "object_id": "item_3",
            "source_dataset": "snli",
            "premise": "Cat sleeps.",
            "hypothesis": "Animal rests.",
            "human_count_entailment": 60,
            "human_count_neutral": 20,
            "human_count_contradiction": 20,
            "human_p_entailment": 0.6,
            "human_p_neutral": 0.2,
            "human_p_contradiction": 0.2,
            "human_entropy_bits": 1.371,
            "human_majority_label": "entailment",
        },
    ]
    df = pl.DataFrame(data)
    pairs_df, summary = find_strict_doppelgaenger_pairs(df)
    
    assert summary["exact_pairs_count"] == 1
    assert pairs_df.height == 1
    row = pairs_df.to_dicts()[0]
    assert row["object_id_a"] == "item_1"
    assert row["object_id_b"] == "item_2"
    assert row["minority_label_high_a"] == "neutral"
    assert row["minority_label_high_b"] == "contradiction"
    assert row["d_hellinger"] > 0.0
