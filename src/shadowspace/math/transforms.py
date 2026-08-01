"""shadowspace.math.transforms — Pure coordinate transformation functions for probability spaces.

Implementations follow Shadowspace representation contracts:
- Raw probabilities (identity)
- Square-root embedding (unit sphere coordinates)
- Centered log-ratio (CLR, via shadowspace.math.clr)
- Logit (log-odds representation with clipping)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from shadowspace.math.clr import clr_transform

__all__ = ["clr_transform", "logit_transform", "sqrt_transform"]


def sqrt_transform(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply the square-root transformation p -> sqrt(p).

    Maps points from the probability simplex onto the positive orthant of the
    unit sphere in L2 norm (since sum(p_i) = 1 implies sum(sqrt(p_i)^2) = 1).

    Args:
        probabilities: Shape (N, K), float64, non-negative probability vectors.

    Returns:
        Shape (N, K) square-root embedded matrix with unit L2 row norms.

    Raises:
        ValueError: If probabilities is not 2-D or contains negative values.
    """
    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must be 2-D, got shape {probabilities.shape}")
    # Reject genuinely negative values; allow tiny floating-point artifacts (< 1e-9)
    if np.any(probabilities < -1e-9):
        raise ValueError(
            f"probabilities must be non-negative for sqrt_transform "
            f"(min value: {probabilities.min():.4g})."
        )

    # Clip tiny floating-point artifacts to 0 before sqrt to prevent NaN
    clipped = np.clip(probabilities, 0.0, None)
    return np.asarray(np.sqrt(clipped), dtype=np.float64)


def logit_transform(probabilities: NDArray[np.float64], eps: float = 1e-6) -> NDArray[np.float64]:
    """Apply the logit (log-odds) transformation log(p / (1 - p)).

    Clips probabilities to [eps, 1 - eps] to handle exact 0 and 1 values safely.

    Args:
        probabilities: Shape (N, K), float64. Values should lie in (0, 1); exact
            boundary values 0 and 1 are clipped to ``[eps, 1 - eps]`` before
            the transform is applied.
        eps: Clipping threshold for boundary values. Defaults to 1e-6.

    Returns:
        Shape (N, K) logit-transformed coordinate matrix.

    Raises:
        ValueError: If probabilities is not 2-D.
    """
    if probabilities.ndim != 2:
        raise ValueError(f"probabilities must be 2-D, got shape {probabilities.shape}")

    clipped = np.clip(probabilities, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped))
    return np.asarray(logits, dtype=np.float64)
