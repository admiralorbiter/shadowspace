"""Export Full 3,113-Item Manifest for E005, E007, E008 Full Runs.

Reads data/chaosnli/processed/canonical_items_posterior.parquet and writes
research/chaosnli/artifacts/E004/manifests/full_3113.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path
import polars as pl

PARQUET_PATH = Path("data/chaosnli/processed/canonical_items_posterior.parquet")
OUT_JSONL = Path("research/chaosnli/artifacts/E004/manifests/full_3113.jsonl")

def main():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Missing parquet dataset: {PARQUET_PATH}")

    df = pl.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df)} canonical items from {PARQUET_PATH}")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    records = df.to_dicts()

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for idx, rec in enumerate(records):
            rec["row_index"] = idx
            f.write(json.dumps(rec) + "\n")

    print(f"Exported {len(records)} items to {OUT_JSONL}")

if __name__ == "__main__":
    main()
