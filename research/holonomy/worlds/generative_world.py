"""Generative Ambiguity World Simulator (Phase E0.7.1).

Generates human and model probability distributions p(x) in Delta^2 over canonical formal states
by transforming interpreter weights in ILR weight space: u(gx) = K_g u(x) + c_g.
Uses an explicit full-rank observation matrix L_x in R^{3x3} so interpreter weight dynamics map cleanly to label distributions.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from numpy.typing import NDArray

from research.holonomy.algebra.formulas import FormalState
from research.holonomy.geometry.simplex_bundle import ilr_inverse, ilr_transform
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
        # Explicit full-rank observation matrix L_x mapping weight vector w in Delta^2 to class probabilities p in Delta^2
        self.observation_matrix_L = np.array(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=np.float64
        )

    def generate_distribution_from_weights(
        self, weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Maps weight vector w in Delta^2 to class probability vector p in Delta^2 via L_x."""
        w_clean = weights / weights.sum()
        p = np.dot(self.observation_matrix_L, w_clean)
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

    def generate_distribution(
        self, state: FormalState, weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.generate_distribution_from_weights(weights)


class CurvedWorld(GenerativeWorld):
    """Curved World: Interpreter weights transform via non-commuting K_a and S_b in ILR weight space."""

    def __init__(self, rotation_angle: float = np.pi / 4) -> None:
        super().__init__()
        self.rotation_angle = rotation_angle
        c, s = np.cos(rotation_angle), np.sin(rotation_angle)
        # K_a: Rotation matrix in ILR weight space
        self.K_a = np.array([[c, -s], [s, c]], dtype=np.float64)
        self.c_a = np.array([0.1, -0.05], dtype=np.float64)

        # S_b: Shear matrix in ILR weight space (non-commuting with K_a)
        self.S_b = np.array([[1.0, 0.5], [0.0, 1.0]], dtype=np.float64)
        self.c_b = np.array([-0.05, 0.1], dtype=np.float64)

        self.base_weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)

    def transform_weights_along_edge(
        self, generator_name: str, u_source: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Transforms ILR weight coordinates u_source along generator edge g: u_target = K_g u_source + c_g."""
        if generator_name == "swap":
            return np.dot(self.K_a, u_source) + self.c_a
        elif generator_name == "neg":
            return np.dot(self.S_b, u_source) + self.c_b
        elif generator_name == "swap_inv":
            return np.dot(np.linalg.inv(self.K_a), u_source - self.c_a)
        elif generator_name == "neg_inv":
            return np.dot(np.linalg.inv(self.S_b), u_source - self.c_b)
        elif generator_name == "flat_close":
            # Nuisance-matched flat null closure map: (R_a^(-1) S_b^(-1) R_a)^(-1)
            K_close = np.dot(np.linalg.inv(self.K_a), np.dot(np.linalg.inv(self.S_b), self.K_a))
            return np.dot(np.linalg.inv(K_close), u_source)
        return u_source.copy()

    def get_weights(self, state: FormalState) -> NDArray[np.float64]:
        p_id = state.premise.canonical_id()
        u_base = ilr_transform(self.base_weights)

        if "Q(x)" in p_id or "SWAP" in state.id:
            u_transformed = self.transform_weights_along_edge("swap", u_base)
            return ilr_inverse(u_transformed)
        elif "NOT" in state.hypothesis.canonical_id():
            u_transformed = self.transform_weights_along_edge("neg", u_base)
            return ilr_inverse(u_transformed)

        return self.base_weights.copy()

    def generate_distribution(
        self, state: FormalState, weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.generate_distribution_from_weights(weights)
