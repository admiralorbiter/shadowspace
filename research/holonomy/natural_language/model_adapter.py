"""Model Adapter & Label Alignment for Phase E2 Classifier Audits.

Maps raw HuggingFace NLI logits/softmax predictions to standard [Entailment, Neutral, Contradiction] order
by inspecting model.config.id2label string mappings.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray


class NLIModelAdapter:
    """Safely aligns NLI model outputs to standard [E, N, C] probability order."""

    def __init__(self, id2label: Dict[int, str] | None = None) -> None:
        # Default RoBERTa / DeBERTa MNLI mapping: 0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT
        if id2label is None:
            self.id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
        else:
            self.id2label = id2label

        self._build_permutation()

    def _build_permutation(self) -> None:
        """Builds index permutation array to reorder probabilities into [E, N, C]."""
        e_idx, n_idx, c_idx = -1, -1, -1
        for idx, lbl in self.id2label.items():
            l_str = str(lbl).upper()
            if "ENTAIL" in l_str or l_str == "E":
                e_idx = int(idx)
            elif "NEUTRAL" in l_str or l_str == "N":
                n_idx = int(idx)
            elif "CONTRADICT" in l_str or l_str == "C":
                c_idx = int(idx)

        if e_idx == -1 or n_idx == -1 or c_idx == -1:
            raise ValueError(f"Could not map all 3 NLI labels from id2label: {self.id2label}")

        # Permutation mapping: standard_probs = raw_probs[[e_idx, n_idx, c_idx]]
        self.perm_indices = np.array([e_idx, n_idx, c_idx], dtype=np.int64)

    def align_probabilities(self, raw_probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reorders raw model probability array to standard [Entailment, Neutral, Contradiction]."""
        raw_arr = np.atleast_1d(raw_probs)
        aligned = raw_arr[self.perm_indices]
        return aligned / aligned.sum()

    def predict_mock_orbit_vertices(
        self, premise: str, hypothesis: str
    ) -> NDArray[np.float64]:
        """Mock prediction for testing pipeline without GPU dependencies."""
        # Simple deterministic hash-based distribution
        h_val = hash(premise + "||" + hypothesis) % 100
        p_raw = np.array([0.5 + 0.001 * (h_val % 10), 0.3, 0.2], dtype=np.float64)
        return p_raw / p_raw.sum()
