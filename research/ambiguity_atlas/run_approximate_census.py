"""Run approximate doppelgänger search and Pareto frontier extraction."""

import os
import polars as pl
from shadowspace.ambiguity_atlas.pair_index import find_approximate_doppelgaenger_pairs

CANONICAL_PATH = "data/chaosnli/processed/canonical_items.parquet"
OUTPUT_PATH = "results/ambiguity_atlas/approximate_pairs.parquet"


def run_approximate_census():
    """Execute approximate doppelgänger census across tolerance levels."""
    print("=== Running Approximate Doppelgänger Census & Pareto Frontier ===")
    
    df_canon = pl.read_parquet(CANONICAL_PATH)
    
    # Loose tolerance bound to capture all candidate pairs
    df_approx = find_approximate_doppelgaenger_pairs(
        df_canon,
        max_conf_diff=0.02,
        max_entropy_diff=0.05
    )
    
    if df_approx.height == 0:
        print("No approximate pairs found under loose tolerance.")
        return
        
    print(f"Total candidate approximate pairs (Loose tolerance): {df_approx.height}")
    
    # Stratify by tolerance tiers
    tight = df_approx.filter((pl.col("confidence_diff") <= 0.005) & (pl.col("entropy_diff") <= 0.01))
    standard = df_approx.filter((pl.col("confidence_diff") <= 0.010) & (pl.col("entropy_diff") <= 0.02))
    
    print(f"Tight tolerance pairs (conf <= 0.005, ent <= 0.01): {tight.height}")
    print(f"Standard tolerance pairs (conf <= 0.010, ent <= 0.02): {standard.height}")
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_approx.write_parquet(OUTPUT_PATH)
    print(f"Approximate pairs saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_approximate_census()
