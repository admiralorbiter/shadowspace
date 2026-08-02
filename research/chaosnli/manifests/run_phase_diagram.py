"""Phase Diagram Simulation Script for Boundary Tie Regimes.

Simulates categorical vote distributions across C categories, n votes, N items,
and 3 Dirichlet concentration regimes to compute boundary tie probability at k=10.
"""

from math import comb
import numpy as np
import polars as pl
from shadowspace.chaosnli.distances import build_distance_matrix

print("=========================================================================", flush=True)
print("           RUNNING TIE-REGIME PHASE DIAGRAM SIMULATION                   ", flush=True)
print("=========================================================================\n", flush=True)

categories_list = [2, 3, 5, 7, 10]
votes_list = [3, 5, 10, 20, 30, 50, 75, 100]
k_eval = 10

# 1. Theoretical Lattice Capacity S(n, C)
print("--- 1. THEORETICAL LATTICE CAPACITY S(n, C) = (n+C-1 choose C-1) ---", flush=True)
print(f"{'n votes':<10} | " + " | ".join([f"C={c:<5}" for c in categories_list]), flush=True)
print("-" * 65, flush=True)
for n in votes_list:
    row_str = f"{n:<10} | "
    capacities = [f"{comb(n + c - 1, c - 1):<7,d}" for c in categories_list]
    print(row_str + " | ".join(capacities), flush=True)


# 2. Empirical Boundary Tie Probability Simulation across Regimes
print("\n--- 2. BOUNDARY TIE PROBABILITY AT k=10 (N=3,113 Items) ---", flush=True)
regimes = {
    "Concentrated (Dirichlet alpha=0.5)": 0.5,
    "Uniform (Dirichlet alpha=1.0)": 1.0,
    "Boundary-Heavy (Dirichlet alpha=0.1)": 0.1
}

rng = np.random.default_rng(20260802)
results_table = []

for r_name, alpha_val in regimes.items():
    print(f"\nRegime: {r_name}", flush=True)
    print(f"{'n votes':<10} | " + " | ".join([f"C={c:<5}" for c in categories_list]), flush=True)
    print("-" * 65, flush=True)

    for n_votes in votes_list:
        row_pcts = []
        for C in categories_list:
            alpha_vec = np.full(C, alpha_val)
            theta = rng.dirichlet(alpha_vec, size=3113)

            # Fast vector sampling
            counts = np.zeros((3113, C), dtype=int)
            for i in range(3113):
                counts[i] = rng.multinomial(n_votes, theta[i])

            p_emp = counts / float(n_votes)
            d_mat = build_distance_matrix(p_emp, metric="hellinger")

            d_sorted = np.sort(d_mat, axis=1)
            tie_mask = (d_sorted[:, k_eval] == d_sorted[:, k_eval + 1])
            tie_pct = float(np.mean(tie_mask) * 100.0)
            row_pcts.append(f"{tie_pct:<6.1f}%")

            results_table.append({
                "regime": r_name,
                "n_votes": n_votes,
                "categories": C,
                "n_items": 3113,
                "tie_pct": tie_pct,
                "lattice_capacity": comb(n_votes + C - 1, C - 1)
            })

        print(f"{n_votes:<10} | " + " | ".join(row_pcts), flush=True)

df_res = pl.DataFrame(results_table)
df_res.write_parquet("data/external/phase_diagram_simulation.parquet")
print("\nPhase diagram simulation saved to data/external/phase_diagram_simulation.parquet", flush=True)

print("\n=========================================================================", flush=True)
print("          PHASE DIAGRAM SIMULATION COMPLETED CLEANLY                     ", flush=True)
print("=========================================================================", flush=True)
