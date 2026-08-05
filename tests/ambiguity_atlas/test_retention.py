"""Unit tests for retention module using synthetic models."""

import pytest
import numpy as np
from shadowspace.ambiguity_atlas.retention import compute_pair_model_retention


def test_synthetic_perfect_preservation_model():
    """Verify synthetic perfect preservation model produces retention_ratio == 1.0 and PRESERVED category."""
    pair = {
        "majority_label": "entailment",
        "minority_orientation_a": 0.5,
        "minority_orientation_b": -0.5,
        "d_hellinger": 0.4,
    }
    # q_a = (0.6, 0.3, 0.1), q_b = (0.6, 0.1, 0.3)
    model_preds_a = {"q_raw_e": 0.6, "q_raw_n": 0.3, "q_raw_c": 0.1}
    model_preds_b = {"q_raw_e": 0.6, "q_raw_n": 0.1, "q_raw_c": 0.3}
    
    res = compute_pair_model_retention(pair, model_preds_a, model_preds_b, tier="raw")
    
    assert np.isclose(res["retention_ratio"], 1.0, atol=1e-10)
    assert res["sign_accurate"] is True
    assert res["retention_category"] == "PRESERVED"


def test_synthetic_collapse_model():
    """Verify synthetic collapse model produces retention_ratio == 0.0 and COLLAPSED category."""
    pair = {
        "majority_label": "entailment",
        "minority_orientation_a": 0.5,
        "minority_orientation_b": -0.5,
        "d_hellinger": 0.4,
    }
    # q_a = (0.6, 0.2, 0.2), q_b = (0.6, 0.2, 0.2)
    model_preds_a = {"q_raw_e": 0.6, "q_raw_n": 0.2, "q_raw_c": 0.2}
    model_preds_b = {"q_raw_e": 0.6, "q_raw_n": 0.2, "q_raw_c": 0.2}
    
    res = compute_pair_model_retention(pair, model_preds_a, model_preds_b, tier="raw")
    
    assert np.isclose(res["retention_ratio"], 0.0, atol=1e-10)
    assert res["retention_category"] == "COLLAPSED"


def test_synthetic_inversion_model():
    """Verify synthetic inversion model produces retention_ratio == -1.0 and INVERTED category."""
    pair = {
        "majority_label": "entailment",
        "minority_orientation_a": 0.5,
        "minority_orientation_b": -0.5,
        "d_hellinger": 0.4,
    }
    # q_a = (0.6, 0.1, 0.3), q_b = (0.6, 0.3, 0.1)  (swapped!)
    model_preds_a = {"q_raw_e": 0.6, "q_raw_n": 0.1, "q_raw_c": 0.3}
    model_preds_b = {"q_raw_e": 0.6, "q_raw_n": 0.3, "q_raw_c": 0.1}
    
    res = compute_pair_model_retention(pair, model_preds_a, model_preds_b, tier="raw")
    
    assert np.isclose(res["retention_ratio"], -1.0, atol=1e-10)
    assert res["sign_accurate"] is False
    assert res["retention_category"] == "INVERTED"
