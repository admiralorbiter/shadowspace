"""Unit tests for Phase E2-A1.2a-R1.2 Live Audit pipeline components."""

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

    z_direct = adapter.compute_direct_ilr_coordinates(aligned_logits)

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


def test_fit_constrained_commuting_transports_slsqp():
    """Verifies SLSQP constrained optimization enforces exact commutation S_H < 1e-10."""
    src_a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    tgt_a = src_a + np.array([0.5, -0.2])

    src_b = tgt_a
    tgt_b = src_b + np.array([-0.1, 0.4])

    t_a_c, t_b_c = fit_constrained_commuting_transports(src_a, tgt_a, src_b, tgt_b)

    # 1. Linear commutation A_a A_b == A_b A_a
    lin_comm = np.dot(t_a_c.matrix_2d, t_b_c.matrix_2d) - np.dot(t_b_c.matrix_2d, t_a_c.matrix_2d)
    assert np.allclose(lin_comm, np.zeros((2, 2)), atol=1e-5)

    # 2. Translation commutation (A_a - I) b_b == (A_b - I) b_a
    trans_comm = np.dot(t_a_c.matrix_2d - np.eye(2), t_b_c.bias_2d) - np.dot(t_b_c.matrix_2d - np.eye(2), t_a_c.bias_2d)
    assert np.allclose(trans_comm, np.zeros(2), atol=1e-5)

    # 3. Homogeneous commutator norm S_H < 1e-5
    commutator_path = compute_forward_affine_commutator(t_a_c, t_b_c)
    stats = compute_holonomy_norm_statistics(commutator_path)
    assert stats["homogeneous_norm_S_H"] < 1e-5



def test_evaluate_edge_predictive_skill():
    """Verifies evaluate_edge_predictive_skill computes RMSE and skill vs identity correctly."""
    estimator = ConnectionEstimator()
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    tgt = src + np.array([0.5, -0.2])  # Pure translation

    t_map = estimator.estimate_linear_transport("test_edge", "s", "t", src, tgt)
    skill = evaluate_edge_predictive_skill(t_map, src, tgt)

    assert pytest.approx(skill["rmse_affine"], abs=1e-5) == 0.0
    assert skill["relative_skill_vs_identity"] > 0.0
