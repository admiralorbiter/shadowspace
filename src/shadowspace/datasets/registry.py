"""Registry for automated benchmark datasets."""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class DatasetSpec:
    """Specification metadata for a benchmark dataset."""

    key: str
    display_name: str
    n_classes: int
    description: str
    source_fn: str
    requires_download: bool = False
    estimated_size_mb: float = 0.0


REGISTRY: Final[dict[str, DatasetSpec]] = {
    "iris_3class": DatasetSpec(
        key="iris_3class",
        display_name="Iris (Fisher's Iris, 3-class)",
        n_classes=3,
        description="Classic 3-class Iris dataset (150 samples, 4 features).",
        source_fn="fetch_iris",
        requires_download=False,
        estimated_size_mb=0.0,
    ),
    "digits_10class": DatasetSpec(
        key="digits_10class",
        display_name="Handwritten Digits (10-class)",
        n_classes=10,
        description="Optical recognition of handwritten digits (1,797 8x8 images).",
        source_fn="fetch_digits",
        requires_download=False,
        estimated_size_mb=0.0,
    ),
    "wine_3class": DatasetSpec(
        key="wine_3class",
        display_name="Wine Recognition (3-class)",
        n_classes=3,
        description="Chemical analysis of wines grown in the same region in Italy (178 samples).",
        source_fn="fetch_wine",
        requires_download=False,
        estimated_size_mb=0.0,
    ),
    "covertype_7class": DatasetSpec(
        key="covertype_7class",
        display_name="Forest Cover Type (7-class, 10k subset)",
        n_classes=7,
        description="Forest cover type from cartographic variables (10,000 stratified subset).",
        source_fn="fetch_covertype",
        requires_download=True,
        estimated_size_mb=11.0,
    ),
}
