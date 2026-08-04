"""Sheaf-Based Out-of-Distribution Metric (GlueOOD Solver).

Computes exact optimal consensus fiber vector v* via least-squares solver and returns normalized GlueOOD score.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from numpy.typing import NDArray

from research.holonomy.geometry.connection import ParallelTransportMap


def compute_glue_ood_score(
    predicted_sections: Sequence[NDArray[np.float64]],
    restriction_maps: Sequence[ParallelTransportMap] | None = None,
    ridge_lambda: float = 1e-6,
) -> float:
    """Computes exact least-squares GlueOOD score over m neighboring contextual predictions.

    Solves v* = argmin_v sum_j || A_j v + b_j - s_j ||^2
    Returns normalized residual score.

    Args:
        predicted_sections: List of m ILR ambiguity vectors s_j.
        restriction_maps: List of m affine transport maps (A_j, b_j).
        ridge_lambda: Ridge regularization parameter.

    Returns:
        GlueOOD score >= 0. High values indicate mutually incompatible contextual predictions.
    """
    if not predicted_sections:
        return 0.0

    sections = [np.atleast_1d(s) for s in predicted_sections]
    m = len(sections)
    d = sections[0].shape[0]

    if restriction_maps is None or len(restriction_maps) != m:
        # Fallback: identity restriction maps A_j = I, b_j = 0 -> v* is centroid
        stacked = np.row_stack(sections)
        v_star = stacked.mean(axis=0)
        residuals = stacked - v_star
        return float(np.sum(residuals ** 2)) / float(m)

    # Solve exact system (sum A_j^T A_j + lambda I) v* = sum A_j^T (s_j - b_j)
    lhs = ridge_lambda * np.eye(d)
    rhs = np.zeros(d, dtype=np.float64)

    for rmap, s_j in zip(restriction_maps, sections):
        A_j = rmap.matrix_2d
        b_j = rmap.bias_2d

        lhs += np.dot(A_j.T, A_j)
        rhs += np.dot(A_j.T, (s_j - b_j))

    v_star = np.linalg.solve(lhs, rhs)

    # Compute residual sum of squares: sum_j || A_j v* + b_j - s_j ||^2
    total_residual = 0.0
    for rmap, s_j in zip(restriction_maps, sections):
        pred = np.dot(rmap.matrix_2d, v_star) + rmap.bias_2d
        total_residual += float(np.sum((pred - s_j) ** 2))

    return total_residual / float(m)
