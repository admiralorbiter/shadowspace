"""
multi_seed_reference_surface.py
================================
Computes R_reference(n, k) = Q(G_n^rep, G_100^obs) with multiple seeds per cell
to obtain uncertainty intervals for the monotonicity claim.

For each (n_votes, k) cell:
  - Draw N_SEEDS independent plug-in multinomial replicates:
      y_i ~ Multinomial(n_votes, p_hat_i)  for each item i
      p_rep_i = y_i / n_votes
  - Compute G_n^rep from p_rep using Hellinger distance
  - Compute Q(G_n^rep, G_100^obs)
  - Aggregate: mean, SD, 95% interval over N_SEEDS

Output: research/chaosnli/artifacts/multi_seed_reference_surface.json
Also prints the updated table with uncertainty intervals.
"""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights, compute_soft_qnx

N_SEEDS = 50
DATA_PATH = "data/chaosnli/processed/canonical_items_posterior.parquet"

n_depths = [3, 5, 10, 20, 30, 50, 75, 100]
k_list = [5, 10, 20, 50, 100]

print("=" * 72)
print(f"  MULTI-SEED REFERENCE SURFACE (N_SEEDS={N_SEEDS} per cell)")
print("=" * 72)

# Load data
df = pl.read_parquet(DATA_PATH)
p_human = df.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
n_items = len(df)
print(f"Loaded {n_items} items")

# Build observed empirical reference graph (G_100^obs) for each k
print("Building G_100^obs for each k...")
d_emp = build_distance_matrix(p_human, metric="hellinger")
w_emp_k = {}
for k_v in k_list:
    w_emp_k[k_v] = compute_soft_neighborhood_weights(d_emp, k=k_v)

# For reproducibility, use base seed = n_votes * 1000 + seed_offset
print(f"\nRunning {N_SEEDS} seeds per cell across {len(n_depths)} vote depths x {len(k_list)} k values...")
t0 = time.time()

results = {}  # (n_v, k_v) -> {"mean": ..., "sd": ..., "ci_lo": ..., "ci_hi": ..., "seeds": [...]}

for n_v in n_depths:
    results[n_v] = {}
    seed_vals = {k_v: [] for k_v in k_list}

    for seed_offset in range(N_SEEDS):
        rng = np.random.default_rng(n_v * 10000 + seed_offset)

        # Draw one replicate at this n_votes
        counts_sub = np.zeros((n_items, 3), dtype=np.int32)
        for i in range(n_items):
            counts_sub[i] = rng.multinomial(n_v, p_human[i])
        p_sub = counts_sub / float(n_v)

        # Compute distance matrix (reused across k values)
        d_sub = build_distance_matrix(p_sub, metric="hellinger")

        for k_v in k_list:
            w_sub = compute_soft_neighborhood_weights(d_sub, k=k_v)
            q_ref, _ = compute_soft_qnx(w_sub, w_emp_k[k_v], k=k_v)
            seed_vals[k_v].append(float(q_ref))

    for k_v in k_list:
        arr = np.array(seed_vals[k_v])
        results[n_v][k_v] = {
            "mean": round(float(np.mean(arr)), 4),
            "sd": round(float(np.std(arr, ddof=1)), 4),
            "ci_lo": round(float(np.percentile(arr, 2.5)), 4),
            "ci_hi": round(float(np.percentile(arr, 97.5)), 4),
            "single_seed_value": round(float(seed_vals[k_v][0]), 4),  # seed_offset=0 ≈ original single seed
        }

    elapsed = time.time() - t0
    print(f"  n_votes={n_v:3d} done  ({elapsed:.1f}s elapsed)")

print(f"\nAll cells computed in {time.time()-t0:.1f}s")

# ==========================================================================
# Print updated table
# ==========================================================================
print("\n--- MULTI-SEED REFERENCE SURFACE: MEAN (SD) [95% CI] ---")
print(f"{'n votes':<10} | " + " | ".join([f"{'k='+str(k):<20}" for k in k_list]))
print("-" * (10 + 25 * len(k_list)))

for n_v in n_depths:
    row = f"{n_v:<10} | "
    cells = []
    for k_v in k_list:
        r = results[n_v][k_v]
        cells.append(f"{r['mean']:.4f}({r['sd']:.4f})")
    row += " | ".join([f"{c:<20}" for c in cells])
    print(row)

print("\n--- MEAN-ONLY TABLE (for document) ---")
print(f"{'n votes':<10} | " + " | ".join([f"{'k='+str(k):>8}" for k in k_list]))
print("-" * 60)
for n_v in n_depths:
    row = f"{n_v:<10} | "
    cells = []
    for k_v in k_list:
        cells.append(f"{results[n_v][k_v]['mean']:>8.4f}")
    row += " | ".join(cells)
    print(row)

# Check monotonicity: does mean increase with n for every k?
print("\n--- MONOTONICITY CHECK ---")
for k_v in k_list:
    means = [results[n_v][k_v]["mean"] for n_v in n_depths]
    is_monotone = all(means[i] <= means[i+1] for i in range(len(means)-1))
    print(f"  k={k_v}: means={[f'{m:.4f}' for m in means]}  MONOTONE={is_monotone}")

# Check if CI lower bounds are all monotone too (stronger claim)
print("\n--- MONOTONICITY OF CI LOWER BOUNDS (k=10) ---")
for k_v in k_list:
    ci_los = [results[n_v][k_v]["ci_lo"] for n_v in n_depths]
    is_monotone_lo = all(ci_los[i] <= ci_los[i+1] for i in range(len(ci_los)-1))
    print(f"  k={k_v}: CI-lo monotone={is_monotone_lo}  bounds={[f'{c:.4f}' for c in ci_los]}")

# Save
output = {
    "n_seeds": N_SEEDS,
    "n_depths": n_depths,
    "k_list": k_list,
    "results": {
        str(n_v): {
            str(k_v): results[n_v][k_v]
            for k_v in k_list
        }
        for n_v in n_depths
    }
}

out_path = Path("research/chaosnli/artifacts/multi_seed_reference_surface.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
    f.write("\n")
print(f"\nSaved to {out_path}")
print("=" * 72)
