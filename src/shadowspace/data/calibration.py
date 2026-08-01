"""Three-class calibration simplex — Sprint 0 smoke fixture.

This is the canonical 15-point fixture defined in the testing plan
(Fixture A). It is deterministic, requires no random seed, and supports
hand-calculated distance checks from Sprint 0 onward.

Point composition
-----------------
  3  simplex corners    : (1,0,0), (0,1,0), (0,0,1)
  3  edge midpoints     : (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
  1  uniform center     : (1/3, 1/3, 1/3)
  8  interior points    : deterministic, all sum to 1.0

IDs are stable and must never be renumbered between bundle versions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Raw fixture data — do not reorder; IDs are positional
# ---------------------------------------------------------------------------

_PROBABILITIES: list[tuple[float, float, float]] = [
    # corners
    (1.0, 0.0, 0.0),  # corner_0
    (0.0, 1.0, 0.0),  # corner_1
    (0.0, 0.0, 1.0),  # corner_2
    # edge midpoints
    (0.5, 0.5, 0.0),  # midpoint_01
    (0.5, 0.0, 0.5),  # midpoint_02
    (0.0, 0.5, 0.5),  # midpoint_12
    # uniform center
    (1 / 3, 1 / 3, 1 / 3),  # center
    # interior points (deterministic; all sum to 1.0)
    (0.7, 0.2, 0.1),  # interior_00
    (0.1, 0.7, 0.2),  # interior_01
    (0.2, 0.1, 0.7),  # interior_02
    (0.6, 0.3, 0.1),  # interior_03
    (0.1, 0.6, 0.3),  # interior_04
    (0.3, 0.1, 0.6),  # interior_05
    (0.4, 0.4, 0.2),  # interior_06
    (0.2, 0.4, 0.4),  # interior_07
]

_OBJECT_IDS: list[str] = [
    "corner_0",
    "corner_1",
    "corner_2",
    "midpoint_01",
    "midpoint_02",
    "midpoint_12",
    "center",
    "interior_00",
    "interior_01",
    "interior_02",
    "interior_03",
    "interior_04",
    "interior_05",
    "interior_06",
    "interior_07",
]

assert len(_PROBABILITIES) == 15, "Fixture must contain exactly 15 points."
assert len(_OBJECT_IDS) == 15, "Fixture must contain exactly 15 IDs."
assert len(set(_OBJECT_IDS)) == 15, "IDs must be unique."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calibration_matrix() -> NDArray[np.float64]:
    """Return the 15x3 probability matrix for the calibration fixture.

    All rows are non-negative and sum to 1.0 within floating-point tolerance.
    """
    return np.array(_PROBABILITIES, dtype=np.float64)


def calibration_ids() -> list[str]:
    """Return the 15 stable object IDs for the calibration fixture."""
    return list(_OBJECT_IDS)


def calibration_fixture() -> tuple[NDArray[np.float64], list[str]]:
    """Return (matrix, ids) for the calibration fixture.

    Convenience wrapper that returns both arrays together.
    """
    return calibration_matrix(), calibration_ids()
