"""Phase E2-A1.1 Hardening Test Suite.

Verifies:
- ConnectionEstimator rank deficiency & condition number identifiability gates
- HuggingFaceNLIAdapter strict mock fallback refusal (use_mock_fallback=False)
- Complete runtime provenance tracking
- E002 experiment handling rank deficiency without false-positive curvature claims
"""

import numpy as np
import pytest

from research.holonomy.experiments.e002_classifier_holonomy import run_e002_classifier_holonomy_experiment
from research.holonomy.geometry.connection import ConnectionEstimator, EstimatorIdentifiabilityError
from research.holonomy.natural_language.model_adapter import HuggingFaceNLIAdapter, NLIModelAdapter


def test_estimator_identifiability_error_on_rank_1_data():
    """Verifies that 1D collinear observation vectors raise EstimatorIdentifiabilityError."""
    estimator = ConnectionEstimator()

    # Create collinear rank-1 source observations in ILR space (2D)
    x = np.linspace(-1.0, 1.0, 10)
    Z_src = np.column_stack([x, 2.0 * x])  # All points fall on line y = 2x
    Z_tgt = Z_src + 0.1

    with pytest.raises(EstimatorIdentifiabilityError) as exc_info:
        estimator.estimate_total_least_squares_transport("test_edge", "x0", "x1", Z_src, Z_tgt)

    assert "rank-deficient" in str(exc_info.value) or "unidentifiable" in str(exc_info.value)


def test_estimator_strict_identifiability_passed_on_full_rank_data():
    """Verifies that 2D full-rank data passes transport estimation with metadata."""
    estimator = ConnectionEstimator()

    # Full-rank 2D synthetic square data
    Z_src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    Z_tgt = Z_src + np.array([0.5, -0.2])  # Pure translation

    t_map = estimator.estimate_total_least_squares_transport("test_edge", "x0", "x1", Z_src, Z_tgt)
    assert t_map.metadata["design_rank"] == 2
    assert t_map.metadata["condition_number"] < 1e3
    assert np.allclose(t_map.matrix_2d, np.eye(2), atol=1e-5)
    assert np.allclose(t_map.bias_2d, [0.5, -0.2], atol=1e-5)


def test_huggingface_adapter_strict_mock_fallback_refusal():
    """Verifies that use_mock_fallback=False raises RuntimeError on invalid model loading."""
    adapter = HuggingFaceNLIAdapter(model_name="non-existent-nli-model-xyz-123", use_mock_fallback=False)

    with pytest.raises(RuntimeError) as exc_info:
        adapter.load()

    assert "Failed to load requested live model" in str(exc_info.value)


def test_huggingface_adapter_provenance_metadata():
    """Verifies that HuggingFaceNLIAdapter returns valid runtime provenance fields."""
    adapter = HuggingFaceNLIAdapter(model_name="roberta-large-mnli", use_mock_fallback=True)
    meta = adapter.get_provenance_metadata()

    assert meta["model_requested"] == "roberta-large-mnli"
    assert meta["use_mock_fallback"] is True
    assert "adapter_mode" in meta


def test_estimator_identifiability_error_structured_fields():
    """Verifies that EstimatorIdentifiabilityError exposes structured failure attributes."""
    estimator = ConnectionEstimator()
    x = np.linspace(-1.0, 1.0, 10)
    Z_src = np.column_stack([x, 2.0 * x])
    Z_tgt = Z_src + 0.1

    with pytest.raises(EstimatorIdentifiabilityError) as exc_info:
        estimator.estimate_total_least_squares_transport("rename_a", "x0", "x1", Z_src, Z_tgt)

    err = exc_info.value
    assert err.generator_name == "rename_a"
    assert err.reason == "rank_deficient"
    assert err.design_rank == 1
    assert err.required_rank == 2
    assert len(err.singular_values) == 2


def test_estimator_input_validation_checks():
    """Verifies input validation for non-finite values, shape mismatch, and insufficient observations."""
    estimator = ConnectionEstimator()
    valid_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # 1. Non-finite values
    nan_coords = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    with pytest.raises(EstimatorIdentifiabilityError) as exc1:
        estimator.estimate_total_least_squares_transport("edge_nan", "s", "t", nan_coords, valid_2d)
    assert exc1.value.reason == "non_finite_values"

    # 2. Shape mismatch
    mismatched = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(EstimatorIdentifiabilityError) as exc2:
        estimator.estimate_total_least_squares_transport("edge_shape", "s", "t", valid_2d, mismatched)
    assert exc2.value.reason == "shape_mismatch"

    # 3. Insufficient observations (N=2 < d+1=3)
    with pytest.raises(EstimatorIdentifiabilityError) as exc3:
        estimator.estimate_total_least_squares_transport("edge_few", "s", "t", mismatched, mismatched)
    assert exc3.value.reason == "insufficient_observations"


def test_huggingface_adapter_mocked_fast_failure(monkeypatch):
    """Verifies fast offline adapter failure using monkeypatched loader."""
    def mock_from_pretrained(*args, **kwargs):
        raise OSError("Offline environment: model checkpoint not found")

    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", mock_from_pretrained)

    adapter = HuggingFaceNLIAdapter(model_name="roberta-large-mnli", use_mock_fallback=False)
    with pytest.raises(RuntimeError) as exc_info:
        adapter.load()

    assert "Failed to load requested live model" in str(exc_info.value)


def test_e002_experiment_mock_mode_identifiability():
    """Verifies that E002 in mock mode identifies rank deficiency, sets audit_status='not_estimable', and uses None for non-estimable fields."""
    results = run_e002_classifier_holonomy_experiment(use_live_model=False)
    res = results[0]

    assert res.is_live_model is False
    assert res.adapter_mode == "deterministic_sha256_mock"
    assert res.estimator_identifiable is False
    assert res.audit_status == "not_estimable"
    assert res.min_edge_design_rank == 1
    assert res.artificial_curvature_detected is False
    assert res.linear_is_flat is None
    assert res.affine_is_flat is None
    assert res.curvature_magnitude is None
    assert res.mean_held_out_return_residual is None
    assert set(res.edge_diagnostics.keys()) == {"rename_a", "rename_b", "rename_a_inv", "rename_b_inv"}


