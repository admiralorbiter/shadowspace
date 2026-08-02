"""Recompute storage-order instability with explicit self-exclusion.

The deterministic policy selects lower storage indices within an exact boundary
tie. Random row permutations therefore change only the arbitrary order inside
the tied boundary block; persistent item identities are restored before overlap
is measured.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import build_distance_matrix

K_VALUES = (5, 10, 20, 50)
N_PERMUTATIONS = 1000
BASE_SEED = 20260802
ATOL = 1e-7


def _boundary_blocks(
    distances: np.ndarray,
    k: int,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    n_items = distances.shape[0]
    boundary_candidates: list[np.ndarray] = []
    base_boundary_selections: list[np.ndarray] = []
    closer_counts = np.zeros(n_items, dtype=np.int16)

    for item_index in range(n_items):
        row = distances[item_index].copy()
        row[item_index] = np.inf
        kth_distance = np.partition(row, k - 1)[k - 1]
        tied = np.flatnonzero(np.isclose(row, kth_distance, atol=ATOL))
        closer = np.flatnonzero(
            (row < kth_distance) & ~np.isclose(row, kth_distance, atol=ATOL)
        )
        remaining = k - len(closer)
        if remaining < 1 or remaining > len(tied):
            raise ValueError(
                f"Invalid boundary block for item {item_index}, k={k}: "
                f"closer={len(closer)}, tied={len(tied)}, remaining={remaining}"
            )
        boundary_candidates.append(tied)
        base_boundary_selections.append(np.sort(tied)[:remaining])
        closer_counts[item_index] = len(closer)

    return boundary_candidates, base_boundary_selections, closer_counts


def _audit_k(
    distances: np.ndarray,
    k: int,
    permutations: list[np.ndarray],
) -> dict[str, object]:
    n_items = distances.shape[0]
    boundaries, base_selections, closer_counts = _boundary_blocks(distances, k)
    overlap_sums = np.zeros(n_items, dtype=np.float64)
    changed = np.zeros(n_items, dtype=bool)
    global_scores = []

    for permutation in permutations:
        storage_position = np.empty(n_items, dtype=np.int32)
        storage_position[permutation] = np.arange(n_items, dtype=np.int32)
        local = np.empty(n_items, dtype=np.float64)

        for item_index, candidates in enumerate(boundaries):
            remaining = k - int(closer_counts[item_index])
            if remaining == len(candidates):
                boundary_overlap = remaining
            else:
                selected_offsets = np.argpartition(
                    storage_position[candidates],
                    remaining - 1,
                )[:remaining]
                selected = candidates[selected_offsets]
                boundary_overlap = np.isin(
                    selected,
                    base_selections[item_index],
                    assume_unique=True,
                ).sum()
            local[item_index] = (
                int(closer_counts[item_index]) + int(boundary_overlap)
            ) / k

        overlap_sums += local
        changed |= local < 1.0
        global_scores.append(float(local.mean()))

    per_item_mean = overlap_sums / len(permutations)
    scores = np.asarray(global_scores)
    return {
        "k": k,
        "global_mean": float(scores.mean()),
        "global_sd": float(scores.std(ddof=1)),
        "global_interval_95": np.percentile(scores, [2.5, 97.5]).tolist(),
        "item_mean_median": float(np.median(per_item_mean)),
        "item_mean_interval_5_95": np.percentile(per_item_mean, [5, 95]).tolist(),
        "item_mean_min": float(per_item_mean.min()),
        "items_changed": int(changed.sum()),
        "items_changed_pct": float(changed.mean() * 100.0),
    }


def main() -> int:
    started = time.perf_counter()
    items_path = Path("data/chaosnli/processed/canonical_items_posterior.parquet")
    if not items_path.exists():
        items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    frame = pl.read_parquet(items_path)
    probabilities = frame.select(
        ["human_p_entailment", "human_p_neutral", "human_p_contradiction"]
    ).to_numpy()
    distances = build_distance_matrix(probabilities, metric="hellinger")

    rng = np.random.default_rng(BASE_SEED)
    permutations = [rng.permutation(len(frame)) for _ in range(N_PERMUTATIONS)]
    rows = []
    for k in K_VALUES:
        print(f"Auditing storage-order instability at k={k}...", flush=True)
        rows.append(_audit_k(distances, k, permutations))

    output = {
        "description": (
            "Stable storage-index tie resolution under random row permutations, "
            "with self-distance explicitly excluded before neighbor selection."
        ),
        "n_items": len(frame),
        "n_permutations": N_PERMUTATIONS,
        "base_seed": BASE_SEED,
        "rows": rows,
        "total_runtime_ms": (time.perf_counter() - started) * 1000.0,
    }
    output_path = Path("research/chaosnli/artifacts/row_order_audit.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Row-order audit written to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
