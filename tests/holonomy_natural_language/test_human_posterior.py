"""Unit tests for Dirichlet human posterior sampler."""

import numpy as np
import pytest
from research.holonomy.natural_language.human_posterior import sample_dirichlet_human_posterior


def test_human_posterior_sampling():
    counts = [60, 30, 10]  # [n_E=60, n_N=30, n_C=10]
    p_sample = sample_dirichlet_human_posterior(counts, alpha=0.5, seed=42)

    assert p_sample.shape == (3,)
    assert np.isclose(p_sample.sum(), 1.0)
    assert p_sample[0] > p_sample[1] > p_sample[2]
