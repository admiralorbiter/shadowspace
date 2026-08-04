"""Unit tests for NLI Model Adapter label alignment and batch shapes."""

import numpy as np
import pytest
from research.holonomy.natural_language.model_adapter import HuggingFaceNLIAdapter, NLIModelAdapter


def test_label_mapping_alignment_1d():
    # RoBERTa / DeBERTa id2label mapping: 0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT
    id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
    adapter = NLIModelAdapter(id2label)

    # Raw model output: [p_C=0.1, p_N=0.3, p_E=0.6]
    raw_probs = np.array([0.1, 0.3, 0.6], dtype=np.float64)
    aligned = adapter.align_probabilities(raw_probs)

    # Expected standard [E, N, C] output: [0.6, 0.3, 0.1]
    assert aligned.shape == (3,)
    assert np.allclose(aligned, [0.6, 0.3, 0.1])


def test_label_mapping_alignment_multidimensional():
    id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
    adapter = NLIModelAdapter(id2label)

    # 2D shape (N=2, 3)
    raw_2d = np.array([[0.1, 0.3, 0.6], [0.2, 0.7, 0.1]], dtype=np.float64)
    aligned_2d = adapter.align_probabilities(raw_2d)
    assert aligned_2d.shape == (2, 3)
    assert np.allclose(aligned_2d[0], [0.6, 0.3, 0.1])
    assert np.allclose(aligned_2d[1], [0.1, 0.7, 0.2])

    # 3D shape (B=2, N=2, 3)
    raw_3d = np.array([[[0.1, 0.3, 0.6], [0.2, 0.7, 0.1]], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    aligned_3d = adapter.align_probabilities(raw_3d)
    assert aligned_3d.shape == (2, 2, 3)
    assert np.allclose(aligned_3d[1, 0], [1.0, 0.0, 0.0])
    assert np.allclose(aligned_3d[1, 1], [0.0, 0.0, 1.0])


def test_huggingface_adapter_mock_fallback():
    hf_adapter = HuggingFaceNLIAdapter(model_name="mock-model", use_mock_fallback=True)
    p_pred = hf_adapter.predict("Alice walked.", "Alice moved.")
    assert p_pred.shape == (3,)
    assert np.isclose(p_pred.sum(), 1.0)
