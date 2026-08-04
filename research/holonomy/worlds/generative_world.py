"""Generative Ambiguity World Simulator (Phase E0.7).

Generates exact human and model probability distributions p(x) in Delta^2 over
canonical formal states by transforming interpreter weights in ILR space:
u(x) = ilr(w(x)), u(gx) = K_g u(x) + c_g, w(gx) = ilr_inverse(u(gx)).
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from numpy.typing import NDArray

from research.holonomy.algebra.formulas import FormalState
from research.holonomy.geometry.simplex_bundle import HELMERT_V3, ilr_inverse, ilr_transform
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

        p = (1.0 - self.epsilon) * p + (self.epsilon / self.num_classes)
        return p / p.sum()


class FlatWorld(GenerativeWorld):
    """Flat World: Interpreter weights w_r are invariant under semantic transformations."""

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
    """Curved World: Interpreter weights transform in ILR space: u(gx) = K_g u(x) + c_g."""

    def __init__(self, rotation_angle: float = np.pi / 4) -> None:
        super().__init__()
        self.rotation_angle = rotation_angle
        c, s = np.cos(rotation_angle), np.sin(rotation_angle)
        # K_g operates in 2D ILR weight space for 3-interpreter simplex Delta^2
        self.K_g1 = np.array([[c, -s], [s, c]], dtype=np.float64)
        self.base_weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)

    def get_weights(self, state: FormalState) -> NDArray[np.float64]:
        p_id = state.premise.canonical_id()
        if "Q(x)" in p_id and p_id.find("Q(x)") < p_id.find("P(x)"):
            # Map base weights into ILR weight space
            u_base = ilr_transform(self.base_weights)
            u_transformed = np.dot(self.K_g1, u_base)
            # Map back to probability simplex Delta^2 via inverse ILR
            return ilr_inverse(u_transformed)
        return self.base_weights.copy()
