"""Export JSON input artifacts for Rust binary preserving authoritative item order."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl


def export_rust_inputs() -> None:
    items_parquet = Path("data/chaosnli/processed/canonical_items.parquet")
    df = pl.read_parquet(items_parquet)  # Do NOT sort! Preserve authoritative order!
    object_ids = df["object_id"].to_list()
    records = df.to_dicts()

    json_items_path = Path("data/chaosnli/processed/canonical_items.json")
    json_items_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_items_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Exported {len(records)} items to {json_items_path}")

    # Compute object-order SHA256
    obj_ids_json = json.dumps(object_ids).encode("utf-8")
    obj_ids_sha = hashlib.sha256(obj_ids_json).hexdigest()

    model_probs_path = Path("research/chaosnli/rust_manifest/model_probs.json")
    if not model_probs_path.exists():
        print(f"WARNING: {model_probs_path} does not exist yet; please generate model_probs.json.")
        return

    model_probs_bytes = model_probs_path.read_bytes()
    model_probs_sha = hashlib.sha256(model_probs_bytes).hexdigest()

    model_probs = json.loads(model_probs_bytes.decode("utf-8"))

    manifest_info = {
        "ordered_object_ids_sha256": obj_ids_sha,
        "model_probs_sha256": model_probs_sha,
        "label_order": ["entailment", "neutral", "contradiction"],
        "n_items": len(object_ids),
        "n_models": len(model_probs),
        "models": list(model_probs.keys())
    }
    manifest_path = model_probs_path.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_info, f, indent=2)

    print(f"Exported model probabilities sidecar manifest to {manifest_path}")
    print(f"  Authoritative Object-Order SHA256: {obj_ids_sha}")
    print(f"  Model Probs File SHA256:           {model_probs_sha}")


if __name__ == "__main__":
    export_rust_inputs()
