"""Domain schemas for Shadowspace data contracts and artifact bundles.

All schemas follow Pydantic v2 conventions and support JSON serialization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ZeroPolicy(BaseModel):
    """Configuration for handling zeros in transformations (e.g., CLR)."""

    policy: str = "multiplicative_replacement"
    delta: float = 1e-6
    match: str = "exact_zero_only"


class TransformProvenance(BaseModel):
    """Provenance details for a coordinate representation transformation."""

    method: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    implementation_version: str = "0.1.0"


class SourceObject(BaseModel):
    """One persistent identity across every representation and view."""

    id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    payload_ref: str | None = None


class RepresentationSpec(BaseModel):
    """Describes one coordinate representation of source objects."""

    id: str
    path: str
    dimension: int
    object_id_column: str = "object_id"
    feature_columns: list[str]
    constraints: list[str] = Field(default_factory=list)
    transform: TransformProvenance | None = None
    compatible_metrics: list[str] = Field(default_factory=list)
    default_metric: str
    zero_policy: ZeroPolicy | None = None
    sha256: str | None = None


class MetricSpec(BaseModel):
    """Specification for a distance or dissimilarity metric."""

    id: str
    display_name: str
    representation_ids: list[str]
    is_metric: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)
    units_or_scale: str | None = None
    implementation_version: str = "0.1.0"


class DisplayTransform(BaseModel):
    """2D screen transformation parameters (rotation, flip, scale)."""

    rotation_deg: float = 0.0
    flip_x: bool = False
    flip_y: bool = False
    scale: float = 1.0


class ViewSpec(BaseModel):
    """A static displayed state."""

    id: str
    representation_id: str
    kind: Literal["linear_projection", "embedding"]
    basis_ref: str | None = None
    coordinates_ref: str | None = None
    display_transform: DisplayTransform = Field(default_factory=DisplayTransform)
    created_by: str = "system"
    seed: int | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_refs(self) -> ViewSpec:
        if self.kind == "linear_projection" and self.basis_ref is None:
            raise ValueError("basis_ref is required for linear_projection views.")
        if self.kind == "embedding" and self.coordinates_ref is None:
            raise ValueError("coordinates_ref is required for embedding views.")
        return self


class PathSpec(BaseModel):
    """A sequence or continuous family of views."""

    id: str
    kind: Literal[
        "linear_projection",
        "sequential_embedding",
        "domain_geodesic",
        "representation_transition",
        "representation_morph",
    ]
    keyframes: list[str]
    interpolation_method: str = "linear"
    intermediate_frames_semantically_valid: bool | None = None
    semantics_note: str = ""
    pacing_metric: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_default_semantic_validity(self) -> PathSpec:
        if self.intermediate_frames_semantically_valid is None:
            if self.kind == "linear_projection":
                self.intermediate_frames_semantically_valid = True
            elif self.kind == "domain_geodesic":
                self.intermediate_frames_semantically_valid = True
            else:
                self.intermediate_frames_semantically_valid = False
        return self


class DiagnosticSpec(BaseModel):
    """Diagnostic state specification relative to a declared source representation and metric."""

    id: str
    source_representation_id: str
    source_metric_id: str
    view_id: str
    method: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    artifact_ref: str


class ObjectTableSpec(BaseModel):
    """Metadata for the primary objects table."""

    path: str = "objects.parquet"
    object_count: int
    sha256: str


class BundleManifest(BaseModel):
    """Complete manifest for a Shadowspace artifact bundle."""

    schema_version: str = "0.1.0"
    bundle_id: str
    created_at: datetime
    description: str = ""
    object_table: ObjectTableSpec
    representations: list[RepresentationSpec]
    metrics: list[MetricSpec] = Field(default_factory=list)
    views: list[ViewSpec] = Field(default_factory=list)
    paths: list[PathSpec] = Field(default_factory=list)
    diagnostics: list[DiagnosticSpec] = Field(default_factory=list)
    extra_artifacts: dict[str, str] = Field(default_factory=dict)
