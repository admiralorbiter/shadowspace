"""shadowspace.projection.paths — Tour path semantic specifications and metadata helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from shadowspace.models.schemas import PathSpec, ViewSpec
from shadowspace.projection.basis import project, validate_orthonormal_basis

__all__ = [
    "create_linear_projection_path",
    "create_representation_morph_path",
    "create_sequential_embedding_path",
    "generate_grand_tour_path",
    "interpolate_orthonormal_bases",
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


def grassmann_geodesic(
    basis_a: NDArray[np.float64],
    basis_b: NDArray[np.float64],
    tau: float,
) -> NDArray[np.float64]:
    """Geodesic interpolation between two 2D projection subspaces on Gr(K, 2).

    Implements GLERP (Grassmann Linear Interpolation) via SVD principal angle decomposition.
    Guarantees exact orthonormality and constant geodesic angular velocity across tau in [0, 1].

    Args:
        basis_a: K x 2 orthonormal basis (start subspace).
        basis_b: K x 2 orthonormal basis (end subspace).
        tau: Interpolation parameter in [0, 1].

    Returns:
        K x 2 orthonormal basis at geodesic position tau.
    """
    if tau <= 0.0:
        return validate_orthonormal_basis(basis_a)
    if tau >= 1.0:
        return validate_orthonormal_basis(basis_b)

    mat_m = basis_a.T @ basis_b  # 2 x 2
    u_mat, s, v_t = np.linalg.svd(mat_m)
    s = np.clip(s, -1.0, 1.0)
    theta = np.arccos(s)  # principal angles (2,)

    y0_star = basis_a @ u_mat  # K x 2
    y1_star = basis_b @ v_t.T  # K x 2

    z_cols: list[NDArray[np.float64]] = []
    for i in range(2):
        sin_th = np.sin(theta[i])
        cos_th = np.cos(theta[i])
        if sin_th > 1e-8:
            q_i = (y1_star[:, i] - cos_th * y0_star[:, i]) / sin_th
        else:
            q_i = np.zeros_like(y0_star[:, i])

        col_i = np.cos(tau * theta[i]) * y0_star[:, i] + np.sin(tau * theta[i]) * q_i
        z_cols.append(col_i)

    z_mat = np.column_stack(z_cols)
    return validate_orthonormal_basis(z_mat)


def interpolate_orthonormal_bases(
    basis_a: NDArray[np.float64],
    basis_b: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    """Interpolate smoothly between two p x 2 orthonormal basis matrices via GLERP."""
    return grassmann_geodesic(basis_a, basis_b, alpha)


def generate_grand_tour_path(
    matrix: NDArray[np.float64],
    n_frames: int = 180,
    seed: int = 42,
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    """Generate a sequence of continuous Grand Tour 2D projection frames via Grassmann geodesics.

    Input matrix is z-score normalised before projection so all frames produce coordinates
    in a stable, bounded range.  The 2-D output is additionally normalised globally so the
    client can use a fixed viewport and never needs to adapt the scale frame-to-frame.

    Keyframe frame budgets are proportional to geodesic distance (principal angle L2 norm)
    so all transitions animate at constant angular velocity without visual jumps.

    Returns:
        (frames_coords, bases_matrices)
    """
    _n_samples, n_features = matrix.shape
    if n_features < 2:
        raise ValueError("Grand tour requires matrix with at least 2 feature columns.")

    # Z-score normalise: zero mean, unit std per feature so that no single
    # feature dominates the projection geometry and all frames share a similar scale.
    mat_centered = matrix - matrix.mean(axis=0)
    stds = mat_centered.std(axis=0)
    stds[stds < 1e-8] = 1.0  # avoid division-by-zero for constant features
    mat_norm = mat_centered / stds

    rng = np.random.default_rng(seed)

    # Generate target keyframe bases
    n_keyframes = max(4, n_frames // 30)
    keyframes: list[NDArray[np.float64]] = []
    for _ in range(n_keyframes):
        raw = rng.normal(size=(n_features, 2))
        q, _ = np.linalg.qr(raw)
        keyframes.append(validate_orthonormal_basis(q[:, :2]))

    # Loop back to keyframes[0] to form a closed tour
    keyframes.append(keyframes[0])

    # Calculate geodesic distances between consecutive keyframes
    distances: list[float] = []
    for k in range(len(keyframes) - 1):
        mat_m = keyframes[k].T @ keyframes[k + 1]
        _, s, _ = np.linalg.svd(mat_m)
        s = np.clip(s, -1.0, 1.0)
        theta = np.arccos(s)
        dist = float(np.linalg.norm(theta))
        distances.append(max(dist, 1e-5))

    total_dist = sum(distances)

    raw_frames: list[NDArray[np.float64]] = []
    bases_matrices: list[list[list[float]]] = []

    for k in range(len(keyframes) - 1):
        b_start = keyframes[k]
        b_end = keyframes[k + 1]
        seg_frames = max(2, round(n_frames * (distances[k] / total_dist)))

        for step in range(seg_frames):
            tau = step / float(seg_frames)
            b_interp = grassmann_geodesic(b_start, b_end, tau)
            coords_2d = project(mat_norm, b_interp)
            raw_frames.append(coords_2d)
            bases_matrices.append(b_interp.tolist())

    # Global normalise the 2-D output across ALL frames so the client can use
    # a single fixed viewport [-1, 1] x [-1, 1] with no per-frame rescaling.
    all_coords = np.concatenate(raw_frames, axis=0)  # (N*T, 2)
    global_min = all_coords.min(axis=0)
    global_max = all_coords.max(axis=0)
    global_range = global_max - global_min
    global_range[global_range < 1e-8] = 1.0

    frames_coords: list[list[list[float]]] = []
    for frame in raw_frames:
        normed = (frame - global_min) / global_range * 2.0 - 1.0  # -> [-1, 1]
        frames_coords.append(normed.tolist())

    return frames_coords, bases_matrices
