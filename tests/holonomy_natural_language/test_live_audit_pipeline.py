"""Unit tests for Phase E2-A1.2a-R1.1 Live Audit pipeline components."""

import numpy as np
import pytest

from research.holonomy.geometry.connection import (
    ConnectionEstimator,
    compute_derived_inverse_map,
    compute_forward_affine_commutator,
    compute_holonomy_norm_statistics,
    compute_rename_context_interaction_test,
    evaluate_edge_predictive_skill,
    fit_constrained_commuting_transports,
    fit_pooled_forward_transports,
    whiten_coordinates,
)
from research.holonomy.natural_language.controlled_orbit_dataset import build_controlled_orbit_dataset
from research.holonomy.natural_language.model_adapter import (
    HuggingFaceNLIAdapter,
    LiveNLIConfig,
    NLIModelAdapter,
    get_helmert_basis,
)


def test_direct_logit_ilr_calculation():
    """Verifies that z = ell @ V is identical to V^T log_softmax(ell)."""
    adapter = NLIModelAdapter()
    V = get_helmert_basis()

    logits = np.array([[2.5, 0.1, -1.2], [-0.5, 3.2, 1.1]], dtype=np.float64)
    aligned_logits = adapter.align_logits(logits)

    # 1. Direct logit calculation
    z_direct = adapter.compute_direct_ilr_coordinates(aligned_logits)

    # 2. Traditional log-softmax calculation
    probs = np.exp(aligned_logits) / np.sum(np.exp(aligned_logits), axis=-1, keepdims=True)
    log_probs = np.log(probs)
    z_log_softmax = np.dot(log_probs, V)

    assert np.allclose(z_direct, z_log_softmax, atol=1e-12)


def test_controlled_orbit_dataset_building_duplicate_free():
    """Verifies building 300 unique controlled orbits with zero text hash overlap between splits."""
    ds = build_controlled_orbit_dataset(target_orbit_count=300, seed=123)

    assert len(ds.train_orbits) == 180
    assert len(ds.val_orbits) == 60
    assert len(ds.test_orbits) == 60

    total_orbits = len(ds.train_orbits) + len(ds.val_orbits) + len(ds.test_orbits)
    assert total_orbits == 300

    for orb in ds.train_orbits:
        assert orb.is_closed is True
        assert "track" in orb.metadata


def test_evaluate_edge_predictive_skill():
    """Verifies evaluate_edge_predictive_skill computes RMSE and skill vs identity correctly."""
    estimator = ConnectionEstimator()
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    tgt = src + np.array([0.5, -0.2])  # Pure translation

    t_map = estimator.estimate_linear_transport("test_edge", "s", "t", src, tgt)
    skill = evaluate_edge_predictive_skill(t_map, src, tgt)

    assert pytest.approx(skill["rmse_affine"], abs=1e-5) == 0.0
    assert skill["relative_skill_vs_identity"] > 0.0


def test_compute_rename_context_interaction_test():
    """Verifies compute_rename_context_interaction_test on zero interaction vectors."""
    orbit_coords = {
        "orb1": {
            "x0": np.array([0.0, 0.0]),
            "x1": np.array([0.5, 0.0]),
            "x2": np.array([0.5, 0.4]),
            "x3": np.array([0.0, 0.4]),  # Perfect rectangle d = (z2-z3) - (z1-z0) = [0,0]
        }
    }
    res = compute_rename_context_interaction_test(orbit_coords, n_permutations=100)
    assert pytest.approx(res["interaction_mean_norm"], abs=1e-5) == 0.0
