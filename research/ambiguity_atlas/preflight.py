"""Preflight script for Ambiguity Doppelgänger Atlas study."""

import os
import json
import hashlib
import polars as pl
from shadowspace.ambiguity_atlas.schemas import validate_canonical_df, validate_oof_df

CANONICAL_PATH = "data/chaosnli/processed/canonical_items.parquet"
OOF_PATH = "results/exploratory/oof_predictions.parquet"
REPORT_PATH = "results/ambiguity_atlas/preflight_report.json"


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_preflight():
    """Run environment and data preflight checks."""
    print("=== Running Ambiguity Doppelgänger Atlas Preflight Checks ===")
    
    if not os.path.exists(CANONICAL_PATH):
        raise FileNotFoundError(f"Canonical items file not found at {CANONICAL_PATH}")
    if not os.path.exists(OOF_PATH):
        raise FileNotFoundError(f"OOF predictions file not found at {OOF_PATH}")

    # Read datasets
    df_canon = pl.read_parquet(CANONICAL_PATH)
    df_oof = pl.read_parquet(OOF_PATH)

    # Validate schemas
    canon_res = validate_canonical_df(df_canon)
    canon_ids = df_canon["object_id"].to_list()
    oof_res = validate_oof_df(df_oof, canon_ids)

    # Hashes
    canon_hash = compute_sha256(CANONICAL_PATH)
    oof_hash = compute_sha256(OOF_PATH)

    report = {
        "status": "PREFLIGHT_PASSED",
        "canonical_items": {
            "path": CANONICAL_PATH,
            "sha256": canon_hash,
            "row_count": canon_res["row_count"],
            "max_prob_sum_diff": canon_res["max_prob_sum_diff"],
        },
        "oof_predictions": {
            "path": OOF_PATH,
            "sha256": oof_hash,
            "row_count": oof_res["row_count"],
            "models": oof_res["models"],
            "unique_objects": oof_res["unique_objects"],
        },
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Preflight passed successfully. Report saved to {REPORT_PATH}")
    print(f"Canonical Items: {canon_res['row_count']} rows")
    print(f"OOF Predictions: {oof_res['row_count']} rows across models: {oof_res['models']}")


if __name__ == "__main__":
    run_preflight()
