"""Export JSON input artifacts for Rust binary from canonical Parquet data."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from shadowspace.chaosnli.models import compute_model_probabilities, load_model_predictions


def export_rust_inputs() -> None:
    items_parquet = Path("data/chaosnli/processed/canonical_items_posterior.parquet")
    if not items_parquet.exists():
        items_parquet = Path("data/chaosnli/processed/canonical_items.parquet")

    df = pl.read_parquet(items_parquet)
    records = df.to_dicts()

    json_items_path = Path("data/chaosnli/processed/canonical_items_posterior.json")
    json_items_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_items_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Exported {len(records)} items to {json_items_path}")

    # Load model predictions
    models = load_model_predictions()
    model_probs = {}
    model_logits = {}
    for m_key, m_data in models.items():
        probs = compute_model_probabilities(m_data["logits"], temperature=1.0).tolist()
        model_probs[m_key] = probs
        model_logits[m_key] = m_data["logits"].tolist()

    model_probs_path = Path("research/chaosnli/rust_manifest/model_probs.json")
    model_probs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_probs_path, "w", encoding="utf-8") as f:
        json.dump(model_probs, f, indent=2)

    model_logits_path = Path("research/chaosnli/rust_manifest/model_logits.json")
    with open(model_logits_path, "w", encoding="utf-8") as f:
        json.dump(model_logits, f, indent=2)

    print(f"Exported {len(model_probs)} model probabilities and logits to {model_probs_path} and {model_logits_path}")


if __name__ == "__main__":
    export_rust_inputs()
