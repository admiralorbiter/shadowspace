"""InvestigationRecord and SavedView Pydantic domain models for reproducible research.

Sprint 5: Integrated research MVP.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class SavedView(BaseModel):
    """Snapshot of a specific visual and diagnostic state in the workbench."""

    id: str
    name: str
    note: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    representation_id: str
    metric_id: str
    k: int
    target_id: str
    path_kind: Literal["linear_projection", "representation_morph", "sequential_embedding"] = "linear_projection"
    semantically_valid: bool = True
    variance_explained: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InvestigationRecord(BaseModel):
    """Complete, reproducible export record of a research investigation."""

    schema_version: str = "0.1.0"
    bundle_id: str = "calibration_bundle_v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    software_version: str = "shadowspace-0.1.0"
    saved_views: list[SavedView] = Field(default_factory=list)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    summary_note: str = ""
