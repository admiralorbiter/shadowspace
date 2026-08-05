"""Unit tests for Minority-Swap Collision Theorem and geometry module."""

import pytest
import numpy as np
from shadowspace.ambiguity_atlas.geometry import (
    mirror_distribution,
    summary_entropy,
    hellinger_mirror_distance,
    fisher_rao_mirror_distance,
    js_mirror_distance,
    aitchison_mirror_distance,
    hellinger_distance,
    fisher_rao_distance,
    js_distance,
    aitchison_distance,
    binary_entropy,
)
from shadowspace.ambiguity_atlas.summaries import (
    compute_shannon_entropy,
    compute_minority_orientation,
)


def test_mirror_sum_to_one():
    """Verify p+ and p- sum to 1.0."""
    m_values = [0.4, 0.5, 0.6, 0.8, 0.95]
    delta_values = [-0.8, -0.3, 0.0, 0.3, 0.8]
    
    for m in m_values:
        for delta in delta_values:
            p_plus, p_minus = mirror_distribution(m, delta)
            np.testing.assert_allclose(np.sum(p_plus), 1.0, atol=1e-12)
            np.testing.assert_allclose(np.sum(p_minus), 1.0, atol=1e-12)


def test_majority_probability_invariant():
    """Verify p+ and p- share the exact same majority probability."""
    m = 0.65
    delta = 0.4
    p_plus, p_minus = mirror_distribution(m, delta)
    assert p_plus[0] == m
    assert p_minus[0] == m
    assert np.max(p_plus) == m
    assert np.max(p_minus) == m


def test_entropy_symmetry_identity():
    """Verify analytical summary entropy H(m, delta) == H(m, -delta) and equals raw vector entropy."""
    m_grid = np.linspace(0.4, 0.9, 10)
    delta_grid = np.linspace(-0.9, 0.9, 10)
    
    for m in m_grid:
        for delta in delta_grid:
            h_analytical_pos = summary_entropy(m, delta)
            h_analytical_neg = summary_entropy(m, -delta)
            
            # Verify symmetry
            assert np.isclose(h_analytical_pos, h_analytical_neg, atol=1e-12)
            
            # Verify match with raw vector Shannon entropy
            p_plus, p_minus = mirror_distribution(m, delta)
            h_raw_plus = compute_shannon_entropy(p_plus)
            h_raw_minus = compute_shannon_entropy(p_minus)
            
            assert np.isclose(h_analytical_pos, h_raw_plus, atol=1e-10)
            assert np.isclose(h_analytical_neg, h_raw_minus, atol=1e-10)


def test_minority_orientation_opposite_signs():
    """Verify minority orientation delta of p+ and p- are exact opposites."""
    m = 0.6
    delta = 0.5
    p_plus, p_minus = mirror_distribution(m, delta)
    
    ori_plus = compute_minority_orientation(p_plus, majority_idx=0)
    ori_minus = compute_minority_orientation(p_minus, majority_idx=0)
    
    assert np.isclose(ori_plus, delta, atol=1e-12)
    assert np.isclose(ori_minus, -delta, atol=1e-12)


def test_closed_form_vs_numerical_distances():
    """Verify closed-form mirror distance formulas equal direct numerical distances."""
    m_list = [0.45, 0.6, 0.75]
    delta_list = [0.1, 0.4, 0.7]
    
    for m in m_list:
        for delta in delta_list:
            p_plus, p_minus = mirror_distribution(m, delta)
            
            # Hellinger
            dh_analytical = hellinger_mirror_distance(m, delta)
            dh_numerical = hellinger_distance(p_plus, p_minus)
            assert np.isclose(dh_analytical, dh_numerical, atol=1e-10)
            
            # Fisher-Rao
            dfr_analytical = fisher_rao_mirror_distance(m, delta)
            dfr_numerical = fisher_rao_distance(p_plus, p_minus)
            assert np.isclose(dfr_analytical, dfr_numerical, atol=1e-10)
            
            # Jensen-Shannon
            djs_analytical = js_mirror_distance(m, delta)
            djs_numerical = js_distance(p_plus, p_minus)
            assert np.isclose(djs_analytical, djs_numerical, atol=1e-10)
            
            # Aitchison
            da_analytical = aitchison_mirror_distance(m, delta)
            da_numerical = aitchison_distance(p_plus, p_minus, alpha=0.0)
            assert np.isclose(da_analytical, da_numerical, atol=1e-8)


def test_zero_distance_at_delta_zero():
    """Verify distance is zero when delta = 0."""
    m = 0.6
    delta = 0.0
    assert hellinger_mirror_distance(m, delta) == 0.0
    assert fisher_rao_mirror_distance(m, delta) == 0.0
    assert js_mirror_distance(m, delta) == 0.0
    assert aitchison_mirror_distance(m, delta) == 0.0


def test_monotonicity_with_delta():
    """Verify distance increases strictly monotonically with |delta|."""
    m = 0.55
    deltas = np.linspace(0.01, 0.8, 20)
    dh_list = [hellinger_mirror_distance(m, d) for d in deltas]
    
    # Check strictly increasing
    diffs = np.diff(dh_list)
    assert np.all(diffs > 0)


def test_valid_domain_majority_preservation():
    """Verify designated class remains majority across valid parameter domain."""
    # Case m >= 0.5: delta in [-1, 1]
    m = 0.5
    for delta in np.linspace(-1, 1, 21):
        p_plus, _ = mirror_distribution(m, delta)
        assert np.argmax(p_plus) == 0
        
    # Case 1/3 <= m < 0.5: |delta| <= (3m - 1) / (1 - m)
    m = 0.4
    max_delta = (3 * m - 1.0) / (1.0 - m)  # 0.2 / 0.6 = 1/3
    # Interior: strict majority at index 0
    for delta in np.linspace(-max_delta + 1e-9, max_delta - 1e-9, 21):
        p_plus, _ = mirror_distribution(m, delta)
        assert np.argmax(p_plus) == 0

    # Boundary: p_plus[0] equals p_plus[1] or p_plus[2], remaining max
    p_boundary, _ = mirror_distribution(m, max_delta)
    assert np.isclose(p_boundary[0], np.max(p_boundary))

