"""Recompute the 100-repetition boundary-tie phase diagram.

The output is an ignored recomputation artifact. Promote it only after the
full grid and empirical reference pass the canonical release validations.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from math import comb
from pathlib import Path

import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.neighbors_soft import compute_boundary_tie_percentage

CATEGORIES = (2, 3, 5, 7, 10)
VOTE_DEPTHS = (3, 5, 10, 20, 30, 50, 100)
ALPHAS = (0.1, 0.5, 1.0)
N_ITEMS = 3113
K = 10
N_REPETITIONS = 100
BASE_SEED = 20260802


def _simulate_cell(args: tuple[float, int, int]) -> dict[str, float | int]:
    alpha, categories, n_votes = args
    tie_percentages = []

    for repetition in range(N_REPETITIONS):
        seed = (
            BASE_SEED
            + repetition * 100_000
            + int(alpha * 1_000) * 1_000
            + categories * 100
            + n_votes
        )
        rng = np.random.default_rng(seed)
        theta = rng.dirichlet(np.full(categories, alpha), size=N_ITEMS)
        counts = np.array(
            [rng.multinomial(n_votes, theta_i) for theta_i in theta],
            dtype=np.int16,
        )
        probabilities = counts / float(n_votes)
        distances = build_distance_matrix(probabilities, metric="hellinger")
        tie_percentages.append(compute_boundary_tie_percentage(distances, k=K))

    values = np.asarray(tie_percentages)
    return {
        "alpha": alpha,
        "c": categories,
        "n_votes": n_votes,
        "mean_tie_pct": round(float(values.mean()), 1),
        "sd_tie_pct": round(float(values.std(ddof=1)), 2),
    }


def _empirical_tie_percentage() -> float:
    items_path = Path("data/chaosnli/processed/canonical_items_posterior.parquet")
    if not items_path.exists():
        items_path = Path("data/chaosnli/processed/canonical_items.parquet")
    frame = pl.read_parquet(items_path)
    probabilities = frame.select(
        ["human_p_entailment", "human_p_neutral", "human_p_contradiction"]
    ).to_numpy()
    distances = build_distance_matrix(probabilities, metric="hellinger")
    return compute_boundary_tie_percentage(distances, k=K)


def main() -> int:
    started = time.perf_counter()
    tasks = [
        (alpha, categories, n_votes)
        for alpha in ALPHAS
        for categories in CATEGORIES
        for n_votes in VOTE_DEPTHS
    ]
    workers = min(os.cpu_count() or 4, 16)
    print(
        f"Running {len(tasks)} phase cells x {N_REPETITIONS} repetitions "
        f"across {workers} workers...",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cells = list(pool.map(_simulate_cell, tasks))

    empirical_tie_pct = _empirical_tie_percentage()
    output = {
        "description": (
            "Boundary tie prevalence at k=10 for N=3,113 categorical-vote items; "
            "100 deterministic repetitions per Dirichlet regime cell."
        ),
        "n_repetitions_per_cell": N_REPETITIONS,
        "n_items": N_ITEMS,
        "k": K,
        "theoretical_lattice_capacity": [
            {
                "n_votes": n_votes,
                "c": categories,
                "capacity": comb(n_votes + categories - 1, categories - 1),
            }
            for n_votes in VOTE_DEPTHS
            for categories in CATEGORIES
        ],
        "empirical_chaosnli_tie_pct": empirical_tie_pct,
        "phase_diagram_100reps": cells,
        "total_runtime_ms": (time.perf_counter() - started) * 1000.0,
    }

    output_path = Path("research/chaosnli/artifacts/phase_diagram_100reps.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Phase-diagram recomputation written to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
