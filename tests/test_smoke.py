"""Sprint 0 smoke tests.

Covers:
- Package import
- conventions module constants
- DtourAdapter contract (valid inputs, all rejection cases)
- Calibration fixture shape, probability invariants, ID uniqueness
- One lightweight Hypothesis property (generated rows sum to 1)
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from shadowspace import __version__
from shadowspace.adapters.dtour import DtourAdapter, RendererAdapter
from shadowspace.conventions import (
    CLR_ZERO_DELTA,
    CLR_ZERO_MATCH,
    CLR_ZERO_POLICY,
    DTOUR_LICENSE,
    DTOUR_PINNED_VERSION,
    FISHER_RAO_CONVENTION,
    FISHER_RAO_SCALE,
    PROB_SUM_ATOL,
)
from shadowspace.data.calibration import (
    calibration_fixture,
    calibration_ids,
    calibration_matrix,
)

# ---------------------------------------------------------------------------
# Package smoke
# ---------------------------------------------------------------------------


def test_package_version_is_string() -> None:
    assert isinstance(__version__, str)
    assert __version__  # not empty


# ---------------------------------------------------------------------------
# Conventions constants
# ---------------------------------------------------------------------------


def test_fisher_rao_convention() -> None:
    assert FISHER_RAO_CONVENTION == "canonical_fisher_information"


def test_fisher_rao_scale() -> None:
    assert FISHER_RAO_SCALE == 2.0


def test_clr_zero_policy() -> None:
    assert CLR_ZERO_POLICY == "multiplicative_replacement"


def test_clr_zero_delta() -> None:
    assert CLR_ZERO_DELTA == 1e-6


def test_clr_zero_match() -> None:
    assert CLR_ZERO_MATCH == "exact_zero_only"


def test_dtour_pinned_version() -> None:
    assert DTOUR_PINNED_VERSION == "0.4.4"


def test_dtour_license() -> None:
    assert DTOUR_LICENSE == "MIT"


# ---------------------------------------------------------------------------
# RendererAdapter protocol runtime check
# ---------------------------------------------------------------------------


def test_dtour_adapter_satisfies_protocol() -> None:
    adapter = DtourAdapter()
    assert isinstance(adapter, RendererAdapter)


# ---------------------------------------------------------------------------
# DtourAdapter — valid input
# ---------------------------------------------------------------------------


def test_adapter_load_valid(
    calib_matrix: NDArray[np.float64], calib_ids: list[str]
) -> None:
    adapter = DtourAdapter()
    adapter.load(calib_matrix, calib_ids)  # must not raise


def test_adapter_load_minimal_valid() -> None:
    adapter = DtourAdapter()
    matrix = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float64)
    adapter.load(matrix, ["a", "b"])


def test_adapter_set_selection_valid(
    calib_matrix: NDArray[np.float64], calib_ids: list[str]
) -> None:
    adapter = DtourAdapter()
    adapter.load(calib_matrix, calib_ids)
    adapter.set_selection({"corner_0", "corner_1"})  # must not raise


def test_adapter_set_selection_empty(
    calib_matrix: NDArray[np.float64], calib_ids: list[str]
) -> None:
    adapter = DtourAdapter()
    adapter.load(calib_matrix, calib_ids)
    adapter.set_selection(set())  # empty selection is valid


def test_adapter_current_view_basis_returns_none(
    calib_matrix: NDArray[np.float64], calib_ids: list[str]
) -> None:
    adapter = DtourAdapter()
    adapter.load(calib_matrix, calib_ids)
    assert adapter.current_view_basis() is None


def test_adapter_set_basis_valid(
    calib_matrix: NDArray[np.float64], calib_ids: list[str]
) -> None:
    adapter = DtourAdapter()
    adapter.load(calib_matrix, calib_ids)
    basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    adapter.set_basis(basis)
    res_basis = adapter.current_view_basis()
    assert res_basis is not None
    np.testing.assert_array_equal(res_basis, basis)


def test_adapter_set_basis_rejects_before_load() -> None:
    adapter = DtourAdapter()
    basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="load\\(\\) must be called before set_basis"):
        adapter.set_basis(basis)


# ---------------------------------------------------------------------------
# DtourAdapter — rejection cases
# ---------------------------------------------------------------------------


def test_adapter_rejects_1d() -> None:
    adapter = DtourAdapter()
    with pytest.raises(ValueError, match="2-D"):
        adapter.load(np.array([0.5, 0.5], dtype=np.float64), ["a"])


def test_adapter_rejects_fewer_than_2_columns() -> None:
    adapter = DtourAdapter()
    with pytest.raises(ValueError, match="at least 2 columns"):
        adapter.load(np.array([[0.5], [0.3]], dtype=np.float64), ["a", "b"])


def test_adapter_rejects_nonnumeric_matrix() -> None:
    adapter = DtourAdapter()
    with pytest.raises(TypeError, match="floating-point"):
        adapter.load(np.array([[1, 0], [0, 1]], dtype=np.int32), ["a", "b"])  # type: ignore[arg-type]


def test_adapter_rejects_non_string_ids() -> None:
    adapter = DtourAdapter()
    with pytest.raises(TypeError, match="strings"):
        adapter.load(np.ones((2, 2), dtype=np.float64), [1, 2])  # type: ignore[list-item]


def test_adapter_rejects_id_count_mismatch() -> None:
    adapter = DtourAdapter()
    with pytest.raises(ValueError, match="rows"):
        adapter.load(np.ones((3, 2), dtype=np.float64), ["a", "b"])


def test_adapter_rejects_nonfinite_values() -> None:
    adapter = DtourAdapter()
    matrix = np.ones((2, 2), dtype=np.float64)
    matrix[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        adapter.load(matrix, ["a", "b"])


def test_adapter_rejects_duplicate_ids() -> None:
    adapter = DtourAdapter()
    with pytest.raises(ValueError, match="unique"):
        adapter.load(np.ones((2, 2), dtype=np.float64), ["a", "a"])


def test_adapter_rejects_empty_string_id() -> None:
    adapter = DtourAdapter()
    with pytest.raises(ValueError, match="empty"):
        adapter.load(np.ones((2, 2), dtype=np.float64), ["a", ""])


def test_adapter_rejects_unknown_selection() -> None:
    adapter = DtourAdapter()
    adapter.load(np.ones((2, 2), dtype=np.float64), ["a", "b"])
    with pytest.raises(ValueError, match="Unknown"):
        adapter.set_selection({"c"})


def test_adapter_rejects_selection_before_load() -> None:
    adapter = DtourAdapter()
    with pytest.raises(ValueError, match=r"load\(\)"):
        adapter.set_selection({"a"})


# ---------------------------------------------------------------------------
# Calibration fixture invariants
# ---------------------------------------------------------------------------


def test_calibration_fixture_shape(calib_matrix: NDArray[np.float64]) -> None:
    assert calib_matrix.shape == (15, 3)


def test_calibration_fixture_dtype(calib_matrix: NDArray[np.float64]) -> None:
    assert calib_matrix.dtype == np.float64


def test_calibration_fixture_nonnegative(calib_matrix: NDArray[np.float64]) -> None:
    assert np.all(calib_matrix >= 0.0)


def test_calibration_fixture_rows_sum_to_one(calib_matrix: NDArray[np.float64]) -> None:
    row_sums = calib_matrix.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=PROB_SUM_ATOL)


def test_calibration_fixture_finite(calib_matrix: NDArray[np.float64]) -> None:
    assert np.all(np.isfinite(calib_matrix))


def test_calibration_ids_count(calib_ids: list[str]) -> None:
    assert len(calib_ids) == 15


def test_calibration_ids_unique(calib_ids: list[str]) -> None:
    assert len(set(calib_ids)) == len(calib_ids)


def test_calibration_ids_nonempty(calib_ids: list[str]) -> None:
    assert all(id_ != "" for id_ in calib_ids)


def test_calibration_fixture_convenience_wrapper() -> None:
    matrix_a = calibration_matrix()
    ids_a = calibration_ids()
    matrix_b, ids_b = calibration_fixture()
    np.testing.assert_array_equal(matrix_a, matrix_b)
    assert ids_a == ids_b


def test_calibration_corners_are_one_hot(calib_matrix: NDArray[np.float64]) -> None:
    """The first three rows must be simplex corners."""
    corners = calib_matrix[:3]
    for i, row in enumerate(corners):
        assert row[i] == 1.0
        assert np.sum(row) == 1.0


def test_calibration_center_is_uniform(calib_matrix: NDArray[np.float64]) -> None:
    """Row index 6 must be the uniform center (1/3, 1/3, 1/3)."""
    center = calib_matrix[6]
    np.testing.assert_allclose(center, 1 / 3, atol=PROB_SUM_ATOL)


# ---------------------------------------------------------------------------
# Hypothesis — lightweight Sprint 0 property
# (Substantive mathematical suite arrives in Sprint 2)
# ---------------------------------------------------------------------------


@given(
    weights=st.lists(
        st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10,
    )
)
@settings(max_examples=200)
def test_normalised_weights_sum_to_one(weights: list[float]) -> None:
    """Normalising any non-negative list produces a valid probability vector."""
    arr = np.array(weights, dtype=np.float64)
    normalised = arr / arr.sum()
    assert abs(normalised.sum() - 1.0) <= PROB_SUM_ATOL
    assert np.all(normalised >= 0.0)
