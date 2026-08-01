"""DtourAdapter — boundary between Shadowspace's math core and dtour.

Rules (ADR-004):
- The mathematical core never imports dtour directly.
- All dtour-specific types stay inside this module.
- The adapter validates inputs at the boundary; internal kernels
  may assume pre-validated arrays.

Sprint 0: Protocol definition + minimal implementation.
Sprint 3: Orthonormal basis validation, projection state tracking, and dtour.Widget integration hooks.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from shadowspace.projection.basis import validate_orthonormal_basis

# ---------------------------------------------------------------------------
# Protocol — any renderer Shadowspace can use must satisfy this
# ---------------------------------------------------------------------------


@runtime_checkable
class RendererAdapter(Protocol):
    """Minimal interface that every Shadowspace renderer must satisfy.

    Implementations live in this module (DtourAdapter) or in future
    adapters for other renderers. The math core depends only on this
    Protocol, never on a concrete class.
    """

    def load(
        self,
        representation_matrix: NDArray[np.float64],
        object_ids: list[str],
    ) -> None:
        """Load a representation matrix and its associated stable object IDs.

        Args:
            representation_matrix: Shape (N, p), p >= 2, all finite float64.
            object_ids: Exactly N unique non-empty strings, one per row.

        Raises:
            ValueError: Shape, dimension, or ID contract violated.
            TypeError: Non-string IDs or non-numeric matrix.
        """
        ...

    def set_selection(self, object_ids: set[str]) -> None:
        """Update the selected set by stable object ID.

        Args:
            object_ids: A subset of the IDs passed to load().

        Raises:
            ValueError: Any ID is not known to the adapter.
        """
        ...

    def set_basis(self, basis: NDArray[np.float64]) -> None:
        """Set the active 2D linear projection basis matrix.

        Args:
            basis: Shape (K, 2) orthonormal projection basis matrix.

        Raises:
            ValueError: Basis is invalid or dimension does not match features.
        """
        ...

    def current_view_basis(self) -> NDArray[np.float64] | None:
        """Return the current px2 projection basis, or None for embeddings.

        Returns None when the current view is an embedding (e.g. UMAP)
        rather than a linear projection with a valid basis.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete adapter — wraps dtour.Widget
# ---------------------------------------------------------------------------


class DtourAdapter:
    """Minimal RendererAdapter backed by dtour.Widget."""

    def __init__(self) -> None:
        self._object_ids: list[str] = []
        self._n_features: int = 0
        self._basis: NDArray[np.float64] | None = None
        self._widget = None

    # ------------------------------------------------------------------
    # RendererAdapter implementation
    # ------------------------------------------------------------------

    def load(
        self,
        representation_matrix: NDArray[np.float64],
        object_ids: list[str],
    ) -> None:
        """Validate inputs and store the representation for rendering.

        Raises:
            TypeError: Matrix is not numeric or IDs are not strings.
            ValueError: Matrix is not 2-D, has fewer than 2 columns,
                        contains non-finite values, IDs are not unique,
                        or ID count does not match row count.
        """
        # --- type checks ---
        if not np.issubdtype(representation_matrix.dtype, np.floating):
            raise TypeError(
                f"representation_matrix must be floating-point, "
                f"got dtype {representation_matrix.dtype!r}"
            )
        if not all(isinstance(id_, str) for id_ in object_ids):
            raise TypeError("All object_ids must be strings.")

        # --- shape checks ---
        if representation_matrix.ndim != 2:
            raise ValueError(
                f"representation_matrix must be 2-D, got shape {representation_matrix.shape}"
            )
        n_rows, n_cols = representation_matrix.shape
        if n_cols < 2:
            raise ValueError(f"representation_matrix must have at least 2 columns, got {n_cols}.")
        if len(object_ids) != n_rows:
            raise ValueError(
                f"object_ids has {len(object_ids)} entries but "
                f"representation_matrix has {n_rows} rows."
            )

        # --- content checks ---
        if not np.all(np.isfinite(representation_matrix)):
            raise ValueError("representation_matrix contains non-finite values.")
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("object_ids must be unique.")
        if any(id_ == "" for id_ in object_ids):
            raise ValueError("object_ids must not contain empty strings.")

        self._object_ids = list(object_ids)
        self._n_features = n_cols
        self._basis = None

    def set_selection(self, object_ids: set[str]) -> None:
        """Update the selected set by stable object ID.

        Raises:
            ValueError: Any ID is unknown or load() has not been called.
        """
        if not self._object_ids:
            raise ValueError("load() must be called before set_selection().")
        unknown = object_ids - set(self._object_ids)
        if unknown:
            raise ValueError(f"Unknown object IDs: {sorted(unknown)}")

    def set_basis(self, basis: NDArray[np.float64]) -> None:
        """Set the active 2D linear projection basis matrix.

        Raises:
            ValueError: If load() has not been called, or basis is invalid,
                        or basis rows != loaded feature count.
        """
        if not self._object_ids:
            raise ValueError("load() must be called before set_basis().")

        validated_basis = validate_orthonormal_basis(basis)
        if validated_basis.shape[0] != self._n_features:
            raise ValueError(
                f"Basis feature dimension {validated_basis.shape[0]} does not match "
                f"loaded feature count {self._n_features}."
            )

        self._basis = validated_basis

    def current_view_basis(self) -> NDArray[np.float64] | None:
        """Return the current px2 basis, or None for embeddings."""
        return self._basis
