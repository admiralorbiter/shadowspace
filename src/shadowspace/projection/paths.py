"""shadowspace.projection.paths — Tour path semantic specifications and metadata helpers."""

from __future__ import annotations

from shadowspace.models.schemas import PathSpec, ViewSpec

__all__ = [
    "create_linear_projection_path",
    "create_representation_morph_path",
    "create_sequential_embedding_path",
]


def create_linear_projection_path(
    path_id: str,
    views: list[ViewSpec],
    display_name: str = "Linear Projection Path",
) -> PathSpec:
    """Create a PathSpec for a sequence of linear projections in the same representation space.

    Intermediate frames in a linear projection path represent true linear projections
    Y = X F(t) and are semantically valid.

    Args:
        path_id: Path ID.
        views: List of ViewSpecs along the tour path (all must share representation_id).
        display_name: Human-readable display name.

    Returns:
        PathSpec with kind='linear_projection' and intermediate_frames_semantically_valid=True.

    Raises:
        ValueError: If views list is empty or views have inconsistent representation_ids.
    """
    if not views:
        raise ValueError("Linear projection path requires at least one ViewSpec.")

    rep_ids = {v.representation_id for v in views}
    if len(rep_ids) > 1:
        raise ValueError(
            f"All views in a linear projection path must share the same representation_id, "
            f"got multiple: {sorted(rep_ids)}"
        )

    return PathSpec(
        id=path_id,
        kind="linear_projection",
        keyframes=[v.id for v in views],
        intermediate_frames_semantically_valid=True,
        metadata={
            "display_name": display_name,
            "representation_id": views[0].representation_id,
        },
    )


def create_representation_morph_path(
    path_id: str,
    source_view: ViewSpec,
    target_view: ViewSpec,
    display_name: str = "Representation Morph",
) -> PathSpec:
    """Create a PathSpec for an animated transition between two different representations.

    Intermediate morph frames are visual interpolations and are NOT valid linear projections
    in either representation space.

    Args:
        path_id: Path ID.
        source_view: Starting representation ViewSpec.
        target_view: Destination representation ViewSpec.
        display_name: Human-readable display name.

    Returns:
        PathSpec with kind='representation_morph' and intermediate_frames_semantically_valid=False.
    """
    return PathSpec(
        id=path_id,
        kind="representation_morph",
        keyframes=[source_view.id, target_view.id],
        intermediate_frames_semantically_valid=False,
        metadata={
            "display_name": display_name,
            "source_representation_id": source_view.representation_id,
            "target_representation_id": target_view.representation_id,
            "warning": (
                "Intermediate frames during a representation morph are coordinate interpolations "
                "and do not represent valid linear projections in either representation space."
            ),
        },
    )


def create_sequential_embedding_path(
    path_id: str,
    views: list[ViewSpec],
    display_name: str = "Sequential Embedding Path",
) -> PathSpec:
    """Create a PathSpec for a sequence of non-linear embedding layouts.

    Args:
        path_id: Path ID.
        views: List of ViewSpecs.
        display_name: Human-readable display name.

    Returns:
        PathSpec with kind='sequential_embedding' and intermediate_frames_semantically_valid=False.
    """
    return PathSpec(
        id=path_id,
        kind="sequential_embedding",
        keyframes=[v.id for v in views],
        intermediate_frames_semantically_valid=False,
        metadata={
            "display_name": display_name,
            "warning": "Intermediate frames between sequential embeddings are not guaranteed to be semantically valid.",
        },
    )
