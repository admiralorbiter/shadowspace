"""shadowspace.math — Mathematical transforms, metrics, and registries for probability spaces."""

from shadowspace.math.clr import clr_transform
from shadowspace.math.metrics import (
    aitchison_distance,
    euclidean_distance,
    fisher_rao_distance,
    hellinger_distance,
    jensen_shannon_distance,
    pairwise_aitchison,
    pairwise_euclidean,
    pairwise_fisher_rao,
    pairwise_hellinger,
    pairwise_jensen_shannon,
)
from shadowspace.math.registry import MetricRegistry
from shadowspace.math.subspace_angles import compute_canonical_angles, compute_grassmannian_distance
from shadowspace.math.transforms import logit_transform, sqrt_transform

__all__ = [
    "MetricRegistry",
    "aitchison_distance",
    "clr_transform",
    "compute_canonical_angles",
    "compute_grassmannian_distance",
    "euclidean_distance",
    "fisher_rao_distance",
    "hellinger_distance",
    "jensen_shannon_distance",
    "logit_transform",
    "pairwise_aitchison",
    "pairwise_euclidean",
    "pairwise_fisher_rao",
    "pairwise_hellinger",
    "pairwise_jensen_shannon",
    "sqrt_transform",
]
