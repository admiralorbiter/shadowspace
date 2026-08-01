"""Tests for Shadowspace Pydantic domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shadowspace.models.schemas import (
    PathSpec,
    SourceObject,
    ViewSpec,
)


def test_source_object_creation() -> None:
    obj = SourceObject(id="obj_001", metadata={"category": "A"})
    assert obj.id == "obj_001"
    assert obj.metadata["category"] == "A"
    assert obj.payload_ref is None


def test_view_spec_validation() -> None:
    # Valid linear projection
    view_lp = ViewSpec(
        id="view_lp",
        representation_id="prob",
        kind="linear_projection",
        basis_ref="views/basis.npz",
    )
    assert view_lp.basis_ref == "views/basis.npz"

    # Missing basis_ref for linear_projection raises ValidationError
    with pytest.raises(ValidationError, match="basis_ref is required"):
        ViewSpec(id="view_lp_bad", representation_id="prob", kind="linear_projection")

    # Valid embedding
    view_emb = ViewSpec(
        id="view_emb",
        representation_id="prob",
        kind="embedding",
        coordinates_ref="views/coords.parquet",
    )
    assert view_emb.coordinates_ref == "views/coords.parquet"

    # Missing coordinates_ref for embedding raises ValidationError
    with pytest.raises(ValidationError, match="coordinates_ref is required"):
        ViewSpec(id="view_emb_bad", representation_id="prob", kind="embedding")


def test_path_spec_semantic_validity_defaults() -> None:
    # linear_projection defaults to True
    p1 = PathSpec(id="p1", kind="linear_projection", keyframes=["v1", "v2"])
    assert p1.intermediate_frames_semantically_valid is True

    # sequential_embedding defaults to False
    p2 = PathSpec(id="p2", kind="sequential_embedding", keyframes=["v1", "v2"])
    assert p2.intermediate_frames_semantically_valid is False

    # domain_geodesic defaults to True
    p3 = PathSpec(id="p3", kind="domain_geodesic", keyframes=["v1", "v2"])
    assert p3.intermediate_frames_semantically_valid is True

    # representation_morph defaults to False
    p4 = PathSpec(id="p4", kind="representation_morph", keyframes=["v1", "v2"])
    assert p4.intermediate_frames_semantically_valid is False

    # Explicit override respected
    p5 = PathSpec(
        id="p5",
        kind="sequential_embedding",
        keyframes=["v1", "v2"],
        intermediate_frames_semantically_valid=True,
    )
    assert p5.intermediate_frames_semantically_valid is True
