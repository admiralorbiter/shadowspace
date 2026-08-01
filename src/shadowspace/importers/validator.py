from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


class ImportValidationError(ValueError):
    """Raised when an imported CSV or Parquet matrix fails Shadowspace validity constraints."""

    pass


def validate_import_matrix(
    matrix: NDArray[np.float64] | np.ndarray[Any, Any],
    n_classes: int | None = None,
    normalize: bool = False,
    sum_atol: float = 1e-3,
) -> NDArray[np.float64]:
    """Validates and processes a candidate probability/logit matrix."""
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=np.float64)

    if matrix.ndim != 2:
        raise ImportValidationError(
            f"Expected a 2D matrix (N samples x K classes), but got shape {matrix.shape}"
        )

    n_samples, k_dim = matrix.shape
    if n_samples < 1:
        raise ImportValidationError("Dataset must contain at least 1 row.")

    if k_dim < 2:
        raise ImportValidationError(f"Dataset must have at least 2 class columns, but got {k_dim}.")

    if n_classes is not None and k_dim != n_classes:
        raise ImportValidationError(
            f"Expected {n_classes} class columns, but found {k_dim} feature columns."
        )

    if not np.isfinite(matrix).all():
        nan_rows = np.where(~np.isfinite(matrix).all(axis=1))[0]
        raise ImportValidationError(
            f"Found NaN or Infinite values at row index(es): {nan_rows[:5].tolist()}"
        )

    if normalize:
        # Apply softmax row-wise: exp(x - max(x)) / sum(exp(x - max(x)))
        shifted = matrix - np.max(matrix, axis=1, keepdims=True)
        exp_matrix = np.exp(shifted)
        probs = exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)
        return cast(NDArray[np.float64], probs.astype(np.float64))

    # Validate probabilities directly
    if (matrix < 0.0).any():
        neg_rows = np.where((matrix < 0.0).any(axis=1))[0]
        raise ImportValidationError(
            f"Probabilities cannot be negative. Invalid values found at row index(es): {neg_rows[:5].tolist()}. "
            "If your file contains unnormalized logit scores, enable the 'normalize' option."
        )

    row_sums = np.sum(matrix, axis=1)
    invalid_sums = np.where(np.abs(row_sums - 1.0) > sum_atol)[0]
    if len(invalid_sums) > 0:
        idx = invalid_sums[0]
        raise ImportValidationError(
            f"Row {idx} probabilities sum to {row_sums[idx]:.5f}, expected 1.0 (±{sum_atol}). "
            "If your file contains raw logits, enable the 'normalize' option."
        )

    # Normalize small floating point roundoff errors to sum exactly to 1.0
    probs = matrix / row_sums[:, np.newaxis]
    return cast(NDArray[np.float64], probs.astype(np.float64))
