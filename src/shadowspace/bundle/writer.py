"""Writer for Shadowspace artifact bundles.

Writes objects.parquet, representation tables, manifest.json, and SHA-256 hashes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from shadowspace.bundle._utils import compute_sha256
from shadowspace.models.schemas import (
    BundleManifest,
    DiagnosticSpec,
    MetricSpec,
    ObjectTableSpec,
    PathSpec,
    RepresentationSpec,
    ViewSpec,
)


class BundleWriter:
    """Creates a self-describing Shadowspace artifact bundle on disk."""

    def __init__(self, output_dir: Path | str, bundle_id: str, description: str = "") -> None:
        self.output_dir = Path(output_dir)
        self.bundle_id = bundle_id
        self.description = description
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "representations").mkdir(exist_ok=True)

        self._objects_df: pl.DataFrame | None = None
        self._representations: dict[str, tuple[pl.DataFrame, RepresentationSpec]] = {}
        self._metrics: list[MetricSpec] = []
        self._views: list[ViewSpec] = []
        self._paths: list[PathSpec] = []
        self._diagnostics: list[DiagnosticSpec] = []
        self._extra_artifacts: dict[str, str] = {}

    def set_objects(self, df: pl.DataFrame) -> None:
        """Set the primary objects DataFrame.

        Must contain an 'object_id' column with unique string IDs.
        """
        if "object_id" not in df.columns:
            raise ValueError("objects DataFrame must contain an 'object_id' column.")
        if df["object_id"].is_duplicated().any():
            raise ValueError("object_id column in objects DataFrame must contain unique values.")
        self._objects_df = df

    def add_representation(
        self,
        rep_id: str,
        df: pl.DataFrame,
        spec: RepresentationSpec,
    ) -> None:
        """Add a representation table and spec to the bundle."""
        if "object_id" not in df.columns:
            raise ValueError(f"Representation {rep_id!r} DataFrame must contain 'object_id'.")
        self._representations[rep_id] = (df, spec)

    def add_metric(self, metric: MetricSpec) -> None:
        """Add a metric specification."""
        self._metrics.append(metric)

    def add_view(self, view: ViewSpec) -> None:
        """Add a view specification."""
        self._views.append(view)

    def add_path(self, path: PathSpec) -> None:
        """Add a path specification."""
        self._paths.append(path)

    def add_diagnostic(self, diagnostic: DiagnosticSpec) -> None:
        """Add a diagnostic specification."""
        self._diagnostics.append(diagnostic)

    def add_extra_artifact(self, key: str, path: str) -> None:
        """Register an extra artifact path in the manifest.

        Args:
            key: Manifest key (e.g., 'readme', 'provenance_log').
            path: Relative path within the bundle directory.
        """
        self._extra_artifacts[key] = path

    def write(self) -> Path:
        """Write all bundle artifacts to output_dir and create manifest.json.

        Returns:
            Path to the generated manifest.json file.
        """
        if self._objects_df is None:
            raise ValueError("Objects DataFrame must be set via set_objects() before writing.")

        # 1. Write objects.parquet
        objects_path = self.output_dir / "objects.parquet"
        self._objects_df.write_parquet(objects_path)
        objects_sha256 = compute_sha256(objects_path)

        object_table_spec = ObjectTableSpec(
            path="objects.parquet",
            object_count=len(self._objects_df),
            sha256=objects_sha256,
        )

        # 2. Write representations/*.parquet and update specs with SHA-256
        rep_specs: list[RepresentationSpec] = []
        for rep_id, (df, spec) in self._representations.items():
            rel_path = f"representations/{rep_id}.parquet"
            full_path = self.output_dir / rel_path
            df.write_parquet(full_path)
            sha256_val = compute_sha256(full_path)

            spec_dict = spec.model_dump()
            spec_dict["path"] = rel_path
            spec_dict["sha256"] = sha256_val
            rep_specs.append(RepresentationSpec(**spec_dict))

        # 3. Create and write manifest.json
        manifest = BundleManifest(
            schema_version="0.1.0",
            bundle_id=self.bundle_id,
            created_at=datetime.now(UTC),
            description=self.description,
            object_table=object_table_spec,
            representations=rep_specs,
            metrics=self._metrics,
            views=self._views,
            paths=self._paths,
            diagnostics=self._diagnostics,
            extra_artifacts=self._extra_artifacts,
        )

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        return manifest_path
