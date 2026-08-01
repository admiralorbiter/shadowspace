"""Pinned numerical and semantic conventions for Shadowspace.

Every implementation must import constants from this module rather than
defining its own literals. Manifests, saved findings, tests, and
displayed values must copy or reference these values.

Changing any constant here is a breaking change; create a new named
constant instead of silently altering an existing one.
"""

# ---------------------------------------------------------------------------
# Fisher-Rao distance  (ADR-013)
# ---------------------------------------------------------------------------

FISHER_RAO_CONVENTION: str = "canonical_fisher_information"
"""Identifies the scaling convention used for all Fisher-Rao distances."""

FISHER_RAO_SCALE: float = 2.0
"""Multiplicative scale factor in d_FR(p, q) = FISHER_RAO_SCALE * arccos(BC(p, q)).

This corresponds to the square-root embedding p -> 2*sqrt(p) on a sphere
of radius 2 and the ordinary, unscaled Fisher information metric.
Distinct simplex corners are at distance pi under this convention.

The no-factor-two quantity is exposed as bhattacharyya_angle(p, q)
in shadowspace.metrics.probability and is NOT an alias of fisher_rao.
"""

# ---------------------------------------------------------------------------
# CLR zero policy  (ADR-014)
# ---------------------------------------------------------------------------

CLR_ZERO_POLICY: str = "multiplicative_replacement"
"""Default zero-replacement method applied before CLR transformation.

For m exact zeros in a K-component probability vector p:
    p_i* = CLR_ZERO_DELTA                    if p_i == 0
    p_i* = (1 - m * CLR_ZERO_DELTA) * p_i   if p_i > 0

This preserves ratios among originally positive components.
See ADR-014 for full implementation requirements.
"""

CLR_ZERO_DELTA: float = 1e-6
"""Replacement value for exact zeros before CLR transformation.

Must satisfy: m * CLR_ZERO_DELTA < 1, where m is the number of zero
components. Inputs where this condition fails must raise ValueError.
"""

CLR_ZERO_MATCH: str = "exact_zero_only"
"""Determines which values are treated as zero for replacement.

'exact_zero_only': only values that are exactly 0.0 are replaced.
Small positive values (e.g. 1e-9) are left untouched.
"""

# ---------------------------------------------------------------------------
# Numerical tolerances
# ---------------------------------------------------------------------------

PROB_SUM_ATOL: float = 1e-10
"""Absolute tolerance for verifying that probability rows sum to 1."""

ORTHONORMAL_ATOL: float = 1e-10
"""Absolute tolerance for verifying F^T F ≈ I_2."""

DISTANCE_ATOL: float = 1e-10
"""Absolute tolerance for distance identity-of-indiscernibles checks."""

REPLAY_ATOL_FLOAT64: float = 1e-9
"""Absolute tolerance for replaying a saved view from float64 coordinates."""

REPLAY_ATOL_FLOAT32: float = 1e-5
"""Absolute tolerance for replaying a saved view from float32 coordinates."""

# ---------------------------------------------------------------------------
# Dependency metadata
# ---------------------------------------------------------------------------

DTOUR_PINNED_VERSION: str = "0.4.4"
"""The exact dtour version pinned in requirements.txt."""

DTOUR_LICENSE: str = "MIT"
"""dtour upstream license. See THIRD_PARTY_NOTICES.txt."""
