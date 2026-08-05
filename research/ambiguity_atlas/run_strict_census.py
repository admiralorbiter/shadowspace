"""Run exact human ambiguity doppelgänger census on ChaosNLI items."""

import os
import json
import polars as pl
from shadowspace.ambiguity_atlas.pair_index import find_strict_doppelgaenger_pairs

CANONICAL_PATH = "data/chaosnli/processed/canonical_items.parquet"
PAIRS_OUTPUT_PATH = "results/ambiguity_atlas/strict_pairs.parquet"
SUMMARY_OUTPUT_PATH = "results/ambiguity_atlas/strict_summary.json"


def run_strict_census():
    """Execute exact doppelgänger pairing census."""
    print("=== Running Exact Human Ambiguity Doppelgänger Census ===")
    
    df_canon = pl.read_parquet(CANONICAL_PATH)
    pairs_df, summary = find_strict_doppelgaenger_pairs(df_canon)
    
    os.makedirs(os.path.dirname(PAIRS_OUTPUT_PATH), exist_ok=True)
    pairs_df.write_parquet(PAIRS_OUTPUT_PATH)
    
    # Enrich summary with metric stats
    if pairs_df.height > 0:
        dh_median = float(pairs_df["d_hellinger"].median())
        dh_mean = float(pairs_df["d_hellinger"].mean())
        dh_min = float(pairs_df["d_hellinger"].min())
        dh_max = float(pairs_df["d_hellinger"].max())
        
        maj_counts = pairs_df["majority_label"].value_counts().to_dicts()
        ds_a = pairs_df["source_dataset_a"].to_list()
        ds_b = pairs_df["source_dataset_b"].to_list()
        
        within_snli = sum(1 for a, b in zip(ds_a, ds_b) if "snli" in a and "snli" in b)
        within_mnli = sum(1 for a, b in zip(ds_a, ds_b) if "mnli" in a and "mnli" in b)
        cross_source = sum(1 for a, b in zip(ds_a, ds_b) if ("snli" in a and "mnli" in b) or ("mnli" in a and "snli" in b))

        
        summary.update({
            "hellinger_median": dh_median,
            "hellinger_mean": dh_mean,
            "hellinger_min": dh_min,
            "hellinger_max": dh_max,
            "majority_label_distribution": maj_counts,
            "source_splits": {
                "within_snli": within_snli,
                "within_mnli": within_mnli,
                "cross_source": cross_source,
            }
        })

    with open(SUMMARY_OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Strict pairs written to {PAIRS_OUTPUT_PATH} ({pairs_df.height} pairs)")
    print(f"Exact collision groups: {summary['exact_groups_count']}")
    print(f"Participating items: {summary['participating_items_count']} / {df_canon.height}")
    print(f"Summary report saved to {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    run_strict_census()
