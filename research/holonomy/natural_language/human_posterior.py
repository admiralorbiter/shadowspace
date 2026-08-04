"""Dirichlet Posterior Sampler for Human Annotation Distributions (Phase E2)."""

from __future__ import annotations

from typing import Sequence
import numpy as np
from numpy.typing import NDArray


def sample_dirichlet_human_posterior(
    counts: NDArray[np.int64] | Sequence[int],
    alpha: float = 0.5,
    num_samples: int = 1,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Samples Dirichlet posterior p^H ~ Dirichlet(n_E + alpha, n_N + alpha, n_C + alpha).

    Standard distribution order: [Entailment, Neutral, Contradiction].
    """
    counts_arr = np.array(counts, dtype=np.float64)

    if counts_arr.shape != (3,):
        raise ValueError(f"Human annotation counts must have shape (3,), got {counts_arr.shape}")
    if np.any(counts_arr < 0):
        raise ValueError(f"Human annotation counts must be non-negative, got {counts}")
    if alpha <= 0:
        raise ValueError(f"Alpha prior parameter must be positive, got {alpha}")
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}")

    rng = np.random.default_rng(seed)
    alpha_vec = counts_arr + alpha
    samples = rng.dirichlet(alpha_vec, size=num_samples)
    return samples[0] if num_samples == 1 else samples
