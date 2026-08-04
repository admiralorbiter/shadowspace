"""Sheaf-Based Out-of-Distribution Metric (GlueOOD).

Evaluates whether a novel observation x* can be coherently attached to the existing transport sheaf.
"""

from __future__ import annotations

from typing import List, Sequence
import numpy as np
from numpy.typing import NDArray

from research.holonomy.geometry.connection import ParallelTransportMap


def compute_glue_ood_score(
    predicted_sections: Sequence[NDArray[np.float64]],
    restriction_maps: Sequence[ParallelTransportMap] | None = None,
) -> float:
    """Computes GlueOOD score over m neighboring contextual predictions.

    Args:
        predicted_sections: List of m ILR ambiguity vectors predicted by contexts U_1, ..., U_m.
        restriction_maps: Optional list of m transport operators R_{x* -> U_j}.

    Returns:
        GlueOOD score >= 0. High values indicate mutually incompatible contextual predictions.
    """
    if not predicted_sections:
        return 0.0

    sections = [np.atleast_1d(s) for s in predicted_sections]

    if restriction_maps is not None and len(restriction_maps) == len(sections):
        # Transform sections into local fiber coordinate frame
        transformed = [rmap.transform(s) for rmap, s in zip(restriction_maps, sections)]
    else:
        transformed = sections

    # Optimal consensus vector v* is the centroid of transformed section predictions
    stacked = np.row_stack(transformed)  # (m, 2)
    consensus_v = stacked.mean(axis=0)

    # Incoherence residual energy = sum_j || v* - transformed_j ||^2
    residuals = stacked - consensus_v
    glue_ood = float(np.sum(residuals ** 2))

    return glue_ood
