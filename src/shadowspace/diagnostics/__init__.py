"""shadowspace.diagnostics — Local integrity diagnostics, k-NN neighbor classification, and view trustworthiness metrics."""

from shadowspace.diagnostics.knn import (
    NeighborClassification,
    classify_point_neighbors,
    compute_knn,
    compute_point_diagnostics,
)
from shadowspace.diagnostics.trustworthiness import (
    compute_kruskal_stress,
    compute_view_continuity,
    compute_view_trustworthiness,
)

__all__ = [
    "NeighborClassification",
    "classify_point_neighbors",
    "compute_knn",
    "compute_kruskal_stress",
    "compute_point_diagnostics",
    "compute_view_continuity",
    "compute_view_trustworthiness",
]
