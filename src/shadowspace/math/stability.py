"""Point stability and Rashomon set analysis for high-dimensional projections.

This module provides tools to assess neighborhood persistence across multiple candidate
projection planes and construct structurally diverse Rashomon candidate sets on the
Grassmannian manifold Gr(k, p).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from shadowspace.diagnostics.trustworthiness import compute_view_trustworthiness
from shadowspace.math.metrics import pairwise_euclidean
from shadowspace.math.subspace_angles import compute_canonical_angles
from shadowspace.projection.basis import canonicalize_basis, validate_orthonormal_basis


def sample_uniform_haar_grassmannian(p: int, k: int = 2, seed: Optional[int] = None) -> np.ndarray:
    """Sample an orthonormal basis for a random k-plane in R^p under the invariant Haar measure.

    Uses Gaussian QR decomposition with Eaton-Mezzadri sign correction to guarantee
    uniform Haar measure distribution on Gr(k, p).

    Args:
        p: High-dimensional feature count.
        k: Projection dimension (default 2).
        seed: Optional random seed.

    Returns:
        Orthonormal basis matrix of shape (p, k).
    """
    if p < k:
        raise ValueError(f"Feature dimension p ({p}) must be >= projection dimension k ({k})")
    
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((p, k))
    Q, R = np.linalg.qr(Z)
    
    # Eaton-Mezzadri (2007) sign correction for Haar measure uniformity
    d = np.diag(R)
    ph = np.sign(d)
    ph[ph == 0.0] = 1.0
    Q = Q * ph
    return Q


def compute_point_stability(
    X: np.ndarray,
    catalog_coords: Dict[str, np.ndarray],
    src_knn_indices: np.ndarray,
    k: int = 5,
) -> Dict[str, Any]:
    """Compute per-point neighborhood persistence overlap ratios across candidate projection views.

    Args:
        X: High-dimensional matrix of shape (N, p).
        catalog_coords: Dictionary mapping view IDs to 2D coordinate matrices (N, 2).
        src_knn_indices: High-dimensional k-NN indices array of shape (N, k).
        k: Neighborhood size.

    Returns:
        Dictionary containing:
            - mean_stability: float
            - persistence_index: float (% of points with S_i >= 0.65)
            - volatile_index: float (% of points with S_i < 0.35)
            - stability_scores: List[float] per point
    """
    n_pts = len(X)
    if n_pts == 0 or len(catalog_coords) == 0:
        return {
            "mean_stability": 1.0,
            "persistence_index": 1.0,
            "volatile_index": 0.0,
            "stability_scores": [1.0] * n_pts,
        }

    k_eff = min(k, max(1, n_pts - 1))
    src_knn_trimmed = src_knn_indices[:, :k_eff]

    # Pre-build boolean mask for source k-NN (N, N)
    src_mask = np.zeros((n_pts, n_pts), dtype=bool)
    rows = np.repeat(np.arange(n_pts), k_eff)
    cols = src_knn_trimmed.ravel()
    src_mask[rows, cols] = True

    # Build 2D k-NN boolean masks for each candidate view
    view_masks = []
    for view_id, coords_2d in catalog_coords.items():
        coords_arr = np.ascontiguousarray(coords_2d, dtype=np.float64)
        if len(coords_arr) != n_pts:
            continue
        
        # Fast pairwise squared Euclidean distance in 2D
        diffs = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=-1)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Top k_eff neighbors in 2D
        proj_knn = np.argpartition(dists_sq, k_eff - 1, axis=1)[:, :k_eff]
        
        v_mask = np.zeros((n_pts, n_pts), dtype=bool)
        v_rows = np.repeat(np.arange(n_pts), k_eff)
        v_cols = proj_knn.ravel()
        v_mask[v_rows, v_cols] = True
        view_masks.append(v_mask)

    if not view_masks:
        return {
            "mean_stability": 1.0,
            "persistence_index": 1.0,
            "volatile_index": 0.0,
            "stability_scores": [1.0] * n_pts,
        }

    # Vectorized boolean tensor intersection
    proj_tensor = np.stack(view_masks, axis=0) # (M, N, N)
    intersections = np.sum(src_mask[np.newaxis, :, :] & proj_tensor, axis=2) # (M, N)
    overlap_ratios = intersections / float(k_eff) # (M, N)
    stability_scores = np.mean(overlap_ratios, axis=0) # (N,)

    scores_list = [round(float(s), 4) for s in stability_scores]
    mean_stab = float(np.mean(stability_scores))
    persistent_count = int(np.sum(stability_scores >= 0.65))
    volatile_count = int(np.sum(stability_scores < 0.35))

    return {
        "mean_stability": round(mean_stab, 4),
        "persistence_index": round(persistent_count / n_pts, 4),
        "volatile_index": round(volatile_count / n_pts, 4),
        "stability_scores": scores_list,
    }


def generate_rashomon_set(
    X: np.ndarray,
    Y_labels: Optional[np.ndarray] = None,
    current_basis: Optional[np.ndarray] = None,
    n_candidates: int = 6,
    quality_threshold: float = 0.50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate a diverse Rashomon candidate set of projection bases on Gr(2, p).

    Selects candidate projection bases (PCA, Fisher LDA, Covariance integrity, Haar random)
    that satisfy a minimum trustworthiness quality threshold, filtered by Grassmannian
    Max-Min distance sampling.

    Args:
        X: Feature matrix of shape (N, p).
        Y_labels: Optional class labels array of shape (N,).
        current_basis: Active projection basis matrix of shape (p, 2).
        n_candidates: Maximum candidate bases to return.
        quality_threshold: Minimum trustworthiness score.
        seed: Random seed for Haar sampling.

    Returns:
        List of dictionaries with candidate view properties (id, display_name, trustworthiness, grassmannian_dist_deg, basis).
    """
    N, p = X.shape
    if p < 2:
        return []

    from shadowspace.projection.subspace import find_discriminative_basis, find_integrity_optimal_basis

    candidate_pool: List[Tuple[str, str, np.ndarray]] = []

    # 1. Deterministic PCA Bases
    try:
        X_centered = X - np.mean(X, axis=0)
        _, _, vt = np.linalg.svd(X_centered, full_matrices=False)
        raw_b12 = vt[:2, :].T
        b_pca12 = validate_orthonormal_basis(canonicalize_basis(raw_b12))
        candidate_pool.append(("pca_corners", "PCA Corner View (PC1-PC2)", b_pca12))
        if p >= 4:
            raw_b34 = vt[2:4, :].T
            b_pca34 = validate_orthonormal_basis(canonicalize_basis(raw_b34))
            candidate_pool.append(("pca_minor", "PCA Secondary View (PC3-PC4)", b_pca34))
    except Exception:
        pass

    # 2. Fisher LDA Discriminative Basis
    if Y_labels is not None and len(np.unique(Y_labels)) > 1:
        try:
            b_lda = find_discriminative_basis(X, Y_labels)
            candidate_pool.append(("fisher_lda", "Fisher LDA Discriminative", b_lda))
        except Exception:
            pass

    # 3. Covariance Local Integrity Basis
    try:
        b_cov = find_integrity_optimal_basis(X, list(range(len(X))))
        candidate_pool.append(("cov_integrity", "Covariance Integrity View", b_cov))
    except Exception:
        pass

    # 4. Uniform Haar Random Bases
    rng = np.random.default_rng(seed)
    for idx in range(12):
        try:
            b_haar = sample_uniform_haar_grassmannian(p, k=2, seed=int(rng.integers(0, 100000)))
            candidate_pool.append((f"haar_sample_{idx+1}", f"Haar Random Sample #{idx+1}", b_haar))
        except Exception:
            pass

    # Pre-compute high-D pairwise distances
    src_dists = pairwise_euclidean(X)

    # Evaluate Trustworthiness for each candidate
    evaluated_candidates = []
    for cand_id, display_name, basis in candidate_pool:
        validate_orthonormal_basis(basis)
        proj_2d = X @ basis
        proj_dists = pairwise_euclidean(proj_2d)
        t_score = float(compute_view_trustworthiness(src_dists, proj_dists, k=5))
        if t_score >= quality_threshold:
            evaluated_candidates.append({
                "id": cand_id,
                "display_name": display_name,
                "trustworthiness": round(t_score, 4),
                "basis": basis,
            })

    if not evaluated_candidates:
        return []

    # Sort candidates by trustworthiness descending
    evaluated_candidates.sort(key=lambda c: c["trustworthiness"], reverse=True)

    # Grassmannian Max-Min Diversity Selection
    selected_candidates: List[Dict[str, Any]] = []
    
    # Reference basis for distance (use current_basis or first candidate)
    ref_basis = current_basis if (current_basis is not None and current_basis.shape == (p, 2)) else evaluated_candidates[0]["basis"]

    for cand in evaluated_candidates:
        if len(selected_candidates) >= n_candidates:
            break

        cand_b = cand["basis"]
        
        # Calculate distance to reference
        angles_ref = compute_canonical_angles(ref_basis, cand_b)
        dist_ref = float(np.hypot(angles_ref[0], angles_ref[1]))

        # Calculate distance to already selected candidates
        min_dist_to_selected = 180.0
        for sel in selected_candidates:
            sel_b = np.array(sel["basis"], dtype=np.float64)
            angles_sel = compute_canonical_angles(sel_b, cand_b)
            d_sel = float(np.hypot(angles_sel[0], angles_sel[1]))
            if d_sel < min_dist_to_selected:
                min_dist_to_selected = d_sel

        # Accept if distinct enough (> 12 degrees separation from existing selections)
        if len(selected_candidates) == 0 or min_dist_to_selected >= 12.0:
            selected_candidates.append({
                "id": cand["id"],
                "display_name": cand["display_name"],
                "trustworthiness": cand["trustworthiness"],
                "grassmannian_dist_deg": round(dist_ref, 2),
                "basis": cand_b.tolist(),
            })

    return selected_candidates
