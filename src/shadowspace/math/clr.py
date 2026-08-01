"""Centered log-ratio (CLR) transform for probability vectors.

Implements the multiplicative zero-replacement policy defined in ADR-014.
All implementations must use this module — never inline _clr_transform.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from shadowspace.conventions import CLR_ZERO_DELTA


def clr_transform(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the centered log-ratio (CLR) transform with multiplicative zero replacement.

    For an (N, K) input matrix:
      1. Count exact zeros per row: m[i] = sum(p[i] == 0)
      2. Replace each zero with CLR_ZERO_DELTA (from conventions).
      3. Scale each positive entry by (1 - m * CLR_ZERO_DELTA) to preserve the simplex.
      4. Apply log and subtract per-row geometric mean.

    This follows ADR-014 exactly. Inputs with m * CLR_ZERO_DELTA >= 1 for any
    row raise ValueError.

    Args:
        probabilities: Shape (N, K), float64, finite, non-negative.
            Rows should sum to 1 (simplex constraint), but this is not enforced here
            — callers are responsible for validating simplex membership.

    Returns:
        Shape (N, K) CLR-transformed matrix.

    Raises:
        ValueError: If any row has so many zeros that m * CLR_ZERO_DELTA >= 1.
    """
    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must be 2-D, got shape {probabilities.shape}")

    mat = probabilities.copy()
    m = (mat == 0.0).sum(axis=1, keepdims=True)

    # Guard: multiplicative replacement requires m * delta < 1
    max_m = int(m.max())
    if max_m * CLR_ZERO_DELTA >= 1.0:
        raise ValueError(
            f"Row has {max_m} zeros; m * CLR_ZERO_DELTA = {max_m * CLR_ZERO_DELTA:.6f} >= 1. "
            "Multiplicative replacement is undefined for this input."
        )

    scale = 1.0 - m * CLR_ZERO_DELTA  # shape (N, 1)

    mask_zero = mat == 0.0
    mat[mask_zero] = CLR_ZERO_DELTA
    mat[~mask_zero] = (scale * probabilities)[~mask_zero]

    log_mat = np.log(mat)
    gm = log_mat.mean(axis=1, keepdims=True)
    return np.asarray(log_mat - gm, dtype=np.float64)
