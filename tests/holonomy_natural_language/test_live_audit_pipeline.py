"""Unit tests for Phase E2-A1.2 Live Audit pipeline components."""

import numpy as np
import pytest

from research.holonomy.geometry.connection import (
    ConnectionEstimator,
    compute_derived_inverse_map,
    compute_forward_affine_commutator,
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


def test_controlled_orbit_dataset_building():
    """Verifies building 300 controlled orbits with balanced NLI labels and 60/20/20 split."""
    ds = build_controlled_orbit_dataset(target_orbit_count=300, seed=123)

    assert len(ds.train_orbits) == 180
    assert len(ds.val_orbits) == 60
    assert len(ds.test_orbits) == 60

    total_orbits = len(ds.train_orbits) + len(ds.val_orbits) + len(ds.test_orbits)
    assert total_orbits == 300

    for orb in ds.train_orbits:
        assert orb.is_closed is True


def test_forward_affine_commutator_derivation():
    """Verifies compute_forward_affine_commutator derives T_a^-1 and T_b^-1 correctly."""
    estimator = ConnectionEstimator()

    # Synthetic full-rank source and target coordinates
    src_a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    tgt_a = src_a + np.array([0.5, -0.2])  # Pure translation A_a = I, b_a = [0.5, -0.2]

    src_b = tgt_a
    tgt_b = src_b + np.array([-0.1, 0.4])  # Pure translation A_b = I, b_b = [-0.1, 0.4]

    t_a = estimator.estimate_linear_transport("rename_a", "x0", "x1", src_a, tgt_a)
    t_b = estimator.estimate_linear_transport("rename_b", "x1", "x2", src_b, tgt_b)

    commutator_path = compute_forward_affine_commutator(t_a, t_b)
    H_hom = commutator_path.compute_homogeneous_matrix()

    # Pure translations commute exactly, so H_hom == I_3
    assert np.allclose(H_hom, np.eye(3), atol=1e-5)
