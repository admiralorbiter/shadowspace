"""Unit tests for NLI Model Adapter label alignment."""

import numpy as np
import pytest
from research.holonomy.natural_language.model_adapter import NLIModelAdapter


def test_label_mapping_alignment():
    # RoBERTa / DeBERTa id2label mapping: 0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT
    id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
    adapter = NLIModelAdapter(id2label)

    # Raw model output: [p_C=0.1, p_N=0.3, p_E=0.6]
    raw_probs = np.array([0.1, 0.3, 0.6], dtype=np.float64)
    aligned = adapter.align_probabilities(raw_probs)

    # Expected standard [E, N, C] output: [0.6, 0.3, 0.1]
    assert np.allclose(aligned, [0.6, 0.3, 0.1])
