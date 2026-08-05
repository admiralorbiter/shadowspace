"""Run model doppelgänger contrast retention audit across models and calibration tiers."""

import os
import json
import polars as pl
from shadowspace.ambiguity_atlas.retention import evaluate_model_retention

STRICT_PAIRS_PATH = "results/ambiguity_atlas/strict_pairs.parquet"
OOF_PATH = "results/exploratory/oof_predictions.parquet"
OUTPUT_PARQUET_PATH = "results/ambiguity_atlas/model_retention.parquet"
SUMMARY_JSON_PATH = "results/ambiguity_atlas/model_retention_summary.json"


def run_model_retention():
    """Execute model prediction retention audit."""
    print("=== Running Frozen Model Retention Audit ===")
    
    df_pairs = pl.read_parquet(STRICT_PAIRS_PATH)
    df_oof = pl.read_parquet(OOF_PATH)
    
    df_ret, df_sum = evaluate_model_retention(df_pairs, df_oof)
    
    os.makedirs(os.path.dirname(OUTPUT_PARQUET_PATH), exist_ok=True)
    df_ret.write_parquet(OUTPUT_PARQUET_PATH)
    
    summary_dict = df_sum.to_dicts()
    with open(SUMMARY_JSON_PATH, "w") as f:
        json.dump(summary_dict, f, indent=2)
        
    print(f"Model retention detailed records saved to {OUTPUT_PARQUET_PATH} ({df_ret.height} records)")
    print(f"Model retention summary saved to {SUMMARY_JSON_PATH}")
    
    print("\nRetention Summary across Models and Calibration Tiers:")
    for row in summary_dict:
        print(f"  Model: {row['model_name']} | Tier: {row['tier']} | Mean Ret Ratio: {row['mean_retention_ratio']:.3f} | Collapse Rate: {row['collapse_rate']:.1%} | Inversion Rate: {row['inversion_rate']:.1%}")


if __name__ == "__main__":
    run_model_retention()
