"""Export Full 3,113-Item Manifest for E005, E007, E008 Full Runs.

Reads data/chaosnli/processed/canonical_items_posterior.json (canonical E001 order,
verified SHA256: 121c49cbd40b171d100ba88c1a23d809818c28bad9249bea99a52ec8f5af19d6)
and writes research/chaosnli/artifacts/E004/manifests/full_3113.jsonl.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

JSON_PATH = Path("data/chaosnli/processed/canonical_items_posterior.json")
OUT_JSONL = Path("research/chaosnli/artifacts/E004/manifests/full_3113.jsonl")

EXPECTED_OBJECT_IDS_SHA256 = "121c49cbd40b171d100ba88c1a23d809818c28bad9249bea99a52ec8f5af19d6"

def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Missing canonical JSON dataset: {JSON_PATH}")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    object_ids = [rec["object_id"] for rec in records]
    actual_ids_sha256 = hashlib.sha256(json.dumps(object_ids, separators=(",", ":")).encode("utf-8")).hexdigest()

    assert actual_ids_sha256 == EXPECTED_OBJECT_IDS_SHA256, (
        f"Object ID order mismatch! Expected {EXPECTED_OBJECT_IDS_SHA256}, got {actual_ids_sha256}"
    )

    print(f"Loaded {len(records)} canonical items from {JSON_PATH}")
    print(f"VERIFIED Object ID Sequence SHA-256: {actual_ids_sha256[:16]}...")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for idx, rec in enumerate(records):
            rec_copy = dict(rec)
            rec_copy["row_index"] = idx
            f.write(json.dumps(rec_copy) + "\n")

    print(f"Exported {len(records)} verified items to {OUT_JSONL}")

if __name__ == "__main__":
    main()
