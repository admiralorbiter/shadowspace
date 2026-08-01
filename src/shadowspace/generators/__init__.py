"""shadowspace.generators — Synthetic belief world and Fashion-MNIST generators."""

from shadowspace.generators.calibration import generate_calibration_bundle
from shadowspace.generators.fashion_mnist import FASHION_CLASSES, generate_fashion_mnist_bundle
from shadowspace.generators.synthetic import generate_synthetic_bundle

__all__ = [
    "FASHION_CLASSES",
    "generate_calibration_bundle",
    "generate_fashion_mnist_bundle",
    "generate_synthetic_bundle",
]
