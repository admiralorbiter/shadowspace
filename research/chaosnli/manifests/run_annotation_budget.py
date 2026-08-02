"""Annotation-Budget Retrospective Simulation Script.

Simulates subsampling vote depths n in {3, 5, 10, 20, 30, 50, 75, 100} from existing
100-vote ChaosNLI counts across 100 simulation passes to compute the R(n, k)
reliability surface across k in {5, 10, 20, 50, 100}.
"""

from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights, compute_soft_qnx

# Load canonical data
df = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")
counts_full = df.select(["human_count_entailment", "human_count_neutral", "human_count_contradiction"]).to_numpy()
n_items = len(df)

n_vote_depths = [3, 5, 10, 20, 30, 50, 75, 100]
k_list = [5, 10, 20, 50, 100]
n_sims = 50  # 50 simulation runs per n_vote depth

print("=========================================================================", flush=True)
print("       RUNNING ANNOTATION-BUDGET RETROSPECTIVE SIMULATION R(n, k)        ", flush=True)
print("=========================================================================\n", flush=True)

def _eval_single_depth_pass(args: tuple[int, int]) -> dict:
    n_votes, seed = args
    rng = np.random.default_rng(seed)

    # Subsample n_votes from 100-vote multinomial for replicate 1 and replicate 2
    # Probability vector p_true is counts_full / 100
    p_true = counts_full / 100.0

    counts_sub1 = np.zeros((n_items, 3), dtype=int)
    counts_sub2 = np.zeros((n_items, 3), dtype=int)
    for i in range(n_items):
        counts_sub1[i] = rng.multinomial(n_votes, p_true[i])
        counts_sub2[i] = rng.multinomial(n_votes, p_true[i])

    p1 = counts_sub1 / float(n_votes)
    p2 = counts_sub2 / float(n_votes)

    d1 = build_distance_matrix(p1, metric="hellinger")
    d2 = build_distance_matrix(p2, metric="hellinger")

    res_pass = {"n_votes": n_votes, "seed": seed}
    for k_val in k_list:
        w1 = compute_soft_neighborhood_weights(d1, k=k_val)
        w2 = compute_soft_neighborhood_weights(d2, k=k_val)
        val, _ = compute_soft_qnx(w1, w2, k=k_val)
        res_pass[f"k_{k_val}"] = float(val)

    return res_pass

def main():
    tasks = []
    for n_votes in n_vote_depths:
        for s in range(n_sims):
            tasks.append((n_votes, s * 1000 + n_votes))

    n_workers = min(os.cpu_count() or 4, 16)
    print(f"Running {len(tasks)} simulation tasks across {n_workers} CPU cores...", flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        all_results = list(pool.map(_eval_single_depth_pass, tasks))

    df_raw = pl.DataFrame(all_results)
    df_raw.write_parquet("data/external/annotation_budget_raw_simulations.parquet")

    # Aggregate R(n, k) surface
    print("\n--- ANNOTATION-BUDGET RELIABILITY SURFACE R(n, k) (MEAN [95% CI]) ---", flush=True)
    print(f"{'n votes':<10} | " + " | ".join([f"k={k:<12}" for k in k_list]), flush=True)
    print("-" * 85, flush=True)

    summary_table = []
    for n_votes in n_vote_depths:
        df_n = df_raw.filter(pl.col("n_votes") == n_votes)
        row_str = f"{n_votes:<10} | "
        k_means = []
        for k_val in k_list:
            vals = df_n[f"k_{k_val}"].to_numpy()
            mean_v = float(np.mean(vals))
            low_v = float(np.percentile(vals, 2.5))
            high_v = float(np.percentile(vals, 97.5))
            k_means.append(f"{mean_v:.4f}")
            summary_table.append({
                "n_votes": n_votes,
                "k": k_val,
                "r_mean": mean_v,
                "r_low": low_v,
                "r_high": high_v,
                "lcmc_mean": mean_v - (k_val / (n_items - 1))
            })
        print(row_str + " | ".join(k_means), flush=True)

    pl.DataFrame(summary_table).write_parquet("data/external/annotation_budget_surface_summary.parquet")
    print("\nAnnotation budget surface saved to data/external/annotation_budget_surface_summary.parquet", flush=True)

    print("\n=========================================================================", flush=True)
    print("      ANNOTATION-BUDGET SIMULATION R(n, k) COMPLETED CLEANLY             ", flush=True)
    print("=========================================================================", flush=True)

if __name__ == "__main__":
    main()
