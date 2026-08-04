"""Dirichlet Posterior Sampler for Human Annotation Distributions (Phase E2)."""

from __future__ import annotations

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
    if seed is not None:
        np.random.seed(seed)

    counts_arr = np.array(counts, dtype=np.float64)
    alpha_vec = counts_arr + alpha
    samples = np.random.dirichlet(alpha_vec, size=num_samples)
    return samples[0] if num_samples == 1 else samples
