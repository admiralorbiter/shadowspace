"""Reader and validator for Shadowspace artifact bundles.

Validates bundle schemas, SHA-256 file hashes, object ID alignment, and numerical constraints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from shadowspace.bundle._utils import compute_sha256
from shadowspace.conventions import PROB_SUM_ATOL
from shadowspace.models.schemas import BundleManifest, RepresentationSpec


@dataclass
class ValidationResult:
    """Result of validating a Shadowspace artifact bundle."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


class BundleValidator:
    """Validates an artifact bundle directory against the data contract."""

    def __init__(self, bundle_dir: Path | str) -> None:
        self.bundle_dir = Path(bundle_dir)

    def validate(self) -> ValidationResult:
        """Perform full validation of the bundle.

        Returns:
            ValidationResult containing status, errors, and warnings.
        """
        result = ValidationResult(is_valid=True)

        manifest_path = self.bundle_dir / "manifest.json"
        if not manifest_path.exists():
            result.add_error(f"Manifest missing at {manifest_path}")
            return result

        try:
            with open(manifest_path, encoding="utf-8") as f:
                data = json.load(f)
            manifest = BundleManifest(**data)
        except Exception as e:
            result.add_error(f"Manifest parsing failed: {e}")
            return result

        # 1. Validate objects.parquet
        obj_table_path = self.bundle_dir / manifest.object_table.path
        if not obj_table_path.exists():
            result.add_error(f"Object table missing at {obj_table_path}")
            return result

        if manifest.object_table.sha256:
            actual_hash = compute_sha256(obj_table_path)
            if actual_hash != manifest.object_table.sha256:
                result.add_error(
                    f"SHA-256 mismatch for {manifest.object_table.path}: "
                    f"expected {manifest.object_table.sha256}, got {actual_hash}"
                )
                return result  # file is corrupt; don't validate contents

        try:
            objects_df = pl.read_parquet(obj_table_path)
        except Exception as e:
            result.add_error(f"Failed to read objects.parquet: {e}")
            return result

        if len(objects_df) != manifest.object_table.object_count:
            result.add_error(
                f"Row count mismatch in objects.parquet: "
                f"manifest says {manifest.object_table.object_count}, file has {len(objects_df)}"
            )

        if "object_id" not in objects_df.columns:
            result.add_error("objects.parquet missing required column 'object_id'")
            return result

        object_ids = objects_df["object_id"].to_list()
        if len(set(object_ids)) != len(object_ids):
            result.add_error("objects.parquet contains duplicate object_id values")

        # 2. Validate representations
        for rep in manifest.representations:
            self._validate_representation(rep, object_ids, result)

        return result

    def _validate_representation(
        self,
        rep: RepresentationSpec,
        expected_ids: list[str],
        result: ValidationResult,
    ) -> None:
        rep_path = self.bundle_dir / rep.path
        if not rep_path.exists():
            result.add_error(f"Representation {rep.id!r} file missing at {rep_path}")
            return

        if rep.sha256:
            actual_hash = compute_sha256(rep_path)
            if actual_hash != rep.sha256:
                result.add_error(
                    f"SHA-256 mismatch for representation {rep.id!r} at {rep.path}: "
                    f"expected {rep.sha256}, got {actual_hash}"
                )
                return  # file is corrupt; don't validate contents

        try:
            df = pl.read_parquet(rep_path)
        except Exception as e:
            result.add_error(f"Failed to read representation {rep.id!r} parquet: {e}")
            return

        if "object_id" not in df.columns:
            result.add_error(f"Representation {rep.id!r} table missing 'object_id' column")
            return

        rep_ids = df["object_id"].to_list()
        if rep_ids != expected_ids:
            if set(rep_ids) == set(expected_ids):
                result.add_error(
                    f"Representation {rep.id!r} row order differs from objects.parquet"
                )
            else:
                result.add_error(
                    f"Representation {rep.id!r} object IDs do not match objects.parquet"
                )

        missing_cols = [c for c in rep.feature_columns if c not in df.columns]
        if missing_cols:
            result.add_error(
                f"Representation {rep.id!r} missing declared feature columns: {missing_cols}"
            )
            return

        if len(rep.feature_columns) != rep.dimension:
            result.add_error(
                f"Representation {rep.id!r} dimension mismatch: "
                f"spec says {rep.dimension}, feature_columns count is {len(rep.feature_columns)}"
            )

        # Numerical constraints check
        mat = df.select(rep.feature_columns).to_numpy()

        if "finite" in rep.constraints:
            if not np.all(np.isfinite(mat)):
                result.add_error(f"Representation {rep.id!r} contains non-finite values (NaN/Inf)")

        if "nonnegative" in rep.constraints:
            if np.any(mat < 0.0):
                result.add_error(
                    f"Representation {rep.id!r} contains negative values violating 'nonnegative' constraint"
                )

        if "row_sum_1" in rep.constraints:
            row_sums = mat.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=PROB_SUM_ATOL):
                result.add_error(
                    f"Representation {rep.id!r} rows do not sum to 1.0 within tolerance"
                )


class BundleReader:
    """Reads and provides access to a validated Shadowspace artifact bundle."""

    def __init__(self, bundle_dir: Path | str) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.manifest_path = self.bundle_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        with open(self.manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        self.manifest = BundleManifest(**data)

    def validate(self) -> ValidationResult:
        """Validate the bundle."""
        validator = BundleValidator(self.bundle_dir)
        return validator.validate()

    def get_objects(self) -> pl.DataFrame:
        """Load and return the primary objects DataFrame."""
        path = self.bundle_dir / self.manifest.object_table.path
        return pl.read_parquet(path)

    def get_representation_spec(self, rep_id: str) -> RepresentationSpec:
        """Return the RepresentationSpec for a specific representation."""
        spec = next((r for r in self.manifest.representations if r.id == rep_id), None)
        if spec is None:
            raise KeyError(f"Representation {rep_id!r} not found in manifest.")
        return spec

    def get_representation(self, rep_id: str) -> pl.DataFrame:
        """Load and return the DataFrame for a specific representation."""
        spec = self.get_representation_spec(rep_id)
        path = self.bundle_dir / spec.path
        return pl.read_parquet(path)

    def get_representation_matrix(self, rep_id: str) -> tuple[NDArray[np.float64], list[str]]:
        """Return (numpy_matrix, object_ids) for a specific representation."""
        spec = next((r for r in self.manifest.representations if r.id == rep_id), None)
        if spec is None:
            raise KeyError(f"Representation {rep_id!r} not found in manifest.")
        df = self.get_representation(rep_id)
        matrix = df.select(spec.feature_columns).to_numpy()
        ids = df["object_id"].to_list()
        return matrix, ids
