"""Pytest configuration and shared fixtures for Shadowspace tests."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from shadowspace.data.calibration import calibration_fixture


@pytest.fixture(scope="session")
def calib_matrix() -> NDArray[np.float64]:
    """15x3 calibration probability matrix (session-scoped, immutable)."""
    matrix, _ = calibration_fixture()
    return matrix


@pytest.fixture(scope="session")
def calib_ids() -> list[str]:
    """15 stable calibration object IDs (session-scoped)."""
    _, ids = calibration_fixture()
    return ids
