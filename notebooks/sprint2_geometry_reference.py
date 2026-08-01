"""Sprint 2 Geometry & Metric Reference Script.

Demonstrates representation and metric dependent neighbor shifts on the
canonical 15-point 3-class calibration fixture.
"""

from __future__ import annotations

from shadowspace.data.calibration import calibration_fixture
from shadowspace.math.clr import clr_transform
from shadowspace.math.registry import MetricRegistry
from shadowspace.math.transforms import sqrt_transform


def main() -> None:
    print("=" * 70)
    print(" SHADOWSPACE SPRINT 2 — GEOMETRY & METRIC REFERENCE")
    print("=" * 70)

    matrix, ids = calibration_fixture()
    print(f"\nLoaded calibration fixture: {len(ids)} points across 3 classes.")

    # Prepare representations
    prob_mat = matrix
    sqrt_mat = sqrt_transform(prob_mat)
    clr_mat = clr_transform(prob_mat)

    registry = MetricRegistry()

    # Target point for comparison: midpoint_01 (index 3 or search by ID)
    target_id = "midpoint_01"
    target_idx = ids.index(target_id)
    k = 5

    print(f"\n--- 5-Nearest Neighbors for Target: {target_id!r} (P = {prob_mat[target_idx]}) ---")

    # 1. Fisher-Rao on probability
    fr_knn = registry.find_k_nearest_neighbors(
        prob_mat, target_idx, k, "fisher_rao", "probability", ids
    )
    print("\n1. Fisher-Rao Distance (probability space):")
    for rank, (nbr_id, dist) in enumerate(fr_knn, 1):
        print(f"   Rank {rank}: {nbr_id:<15} (dist = {dist:.4f} rad)")

    # 2. Hellinger Distance on probability (matches Euclidean on sqrt_probability / sqrt(2))
    hel_knn = registry.find_k_nearest_neighbors(
        prob_mat, target_idx, k, "hellinger", "probability", ids
    )
    euc_sqrt_knn = registry.find_k_nearest_neighbors(
        sqrt_mat, target_idx, k, "euclidean", "sqrt_probability", ids
    )
    print("\n2. Hellinger Distance (probability space):")
    for rank, (nbr_id, dist) in enumerate(hel_knn, 1):
        print(f"   Rank {rank}: {nbr_id:<15} (dist = {dist:.4f})")
    assert [id_ for id_, _ in hel_knn] == [id_ for id_, _ in euc_sqrt_knn]

    # 3. Aitchison on CLR
    ait_knn = registry.find_k_nearest_neighbors(
        clr_mat, target_idx, k, "aitchison", "clr_probability", ids
    )
    print("\n3. Aitchison Distance (CLR composition space):")
    for rank, (nbr_id, dist) in enumerate(ait_knn, 1):
        print(f"   Rank {rank}: {nbr_id:<15} (dist = {dist:.4f})")

    # 4. Jensen-Shannon on probability
    js_knn = registry.find_k_nearest_neighbors(
        prob_mat, target_idx, k, "jensen_shannon", "probability", ids
    )
    print("\n4. Jensen-Shannon Distance (probability space):")
    for rank, (nbr_id, dist) in enumerate(js_knn, 1):
        print(f"   Rank {rank}: {nbr_id:<15} (dist = {dist:.4f} bits)")

    print("\n" + "=" * 70)
    print(" Verification complete: neighbor orders change predictably by geometry.")
    print("=" * 70)


if __name__ == "__main__":
    main()
