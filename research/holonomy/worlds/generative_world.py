"""Generative Ambiguity World Simulator (Phase E0).

Generates exact human and model probability distributions p(x) in Delta^2 over
formal Cayley complex states by mixing latent interpreters with context-dependent weights.
"""

from __future__ import annotations

from typing import List, Sequence
import numpy as np
from numpy.typing import NDArray

from research.holonomy.worlds.formulas import FormalState
from research.holonomy.worlds.interpreters import STANDARD_INTERPRETERS, LatentInterpreter


class GenerativeWorld:
    """Generates synthetic probability distributions over 3-class NLI space."""

    def __init__(
        self,
        interpreters: Sequence[LatentInterpreter] | None = None,
        num_classes: int = 3,
        epsilon: float = 1e-4,
    ) -> None:
        self.interpreters = list(interpreters or STANDARD_INTERPRETERS)
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.num_interpreters = len(self.interpreters)

    def evaluate_interpreters(self, state: FormalState) -> NDArray[np.int64]:
        """Returns labels y_r(state) for each latent interpreter."""
        return np.array([interp(state) for interp in self.interpreters], dtype=np.int64)

    def generate_distribution(
        self, state: FormalState, weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Mixes interpreter outcomes according to weights w_r (summing to 1)."""
        labels = self.evaluate_interpreters(state)
        p = np.zeros(self.num_classes, dtype=np.float64)
        for w, lbl in zip(weights, labels):
            p[lbl] += w

        # Smooth slightly to keep strictly inside interior of simplex Delta^2
        p = (1.0 - self.epsilon) * p + (self.epsilon / self.num_classes)
        return p / p.sum()


class FlatWorld(GenerativeWorld):
    """Flat World: Interpreter weights w_r are invariant under all semantic transformations."""

    def __init__(self, base_weights: NDArray[np.float64] | None = None) -> None:
        super().__init__()
        if base_weights is None:
            w = np.ones(self.num_interpreters, dtype=np.float64)
            self.base_weights = w / w.sum()
        else:
            self.base_weights = base_weights / base_weights.sum()

    def get_weights(self, state: FormalState) -> NDArray[np.float64]:
        return self.base_weights.copy()


class CurvedWorld(GenerativeWorld):
    """Curved World: Interpreter weights transform under K_g(x), inducing planted holonomy."""

    def __init__(self, rotation_angle: float = np.pi / 4) -> None:
        super().__init__()
        self.rotation_angle = rotation_angle
        # Create non-commuting transition matrices for transformation generators
        c, s = np.cos(rotation_angle), np.sin(rotation_angle)
        self.K_g1 = np.array([
            [c, -s, 0.0, 0.0],
            [s,  c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def get_weights(self, state: FormalState) -> NDArray[np.float64]:
        # Context-dependent weight modulation based on state metadata
        base_w = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        if "swap" in state.id:
            w = np.dot(self.K_g1, base_w)
            w = np.abs(w)
            return w / w.sum()
        return base_w
