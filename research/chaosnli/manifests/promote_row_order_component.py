"""Atomically promote a corrected row-order audit into the canonical release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_K_VALUES = (5, 10, 20, 50)
EXPECTED_N_ITEMS = 3113
EXPECTED_PERMUTATIONS = 1000


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    if audit.get("n_items") != EXPECTED_N_ITEMS:
        raise ValueError("Row-order audit must use the locked 3,113-item scope")
    if audit.get("n_permutations") != EXPECTED_PERMUTATIONS:
        raise ValueError("Row-order audit must contain exactly 1,000 permutations")
    rows = audit.get("rows")
    if not isinstance(rows, list) or {row["k"] for row in rows} != set(EXPECTED_K_VALUES):
        raise ValueError("Row-order audit must contain exactly k={5,10,20,50}")
    if len(rows) != len(EXPECTED_K_VALUES):
        raise ValueError("Row-order audit contains duplicate scale rows")

    for row in rows:
        bounded_fields = (
            "global_mean",
            "item_mean_median",
            "item_mean_min",
            "items_changed_pct",
        )
        for field in bounded_fields:
            value = float(row[field])
            upper = 100.0 if field == "items_changed_pct" else 1.0
            if not math.isfinite(value) or not 0.0 <= value <= upper:
                raise ValueError(f"Invalid {field} for k={row['k']}: {value}")
        if not math.isfinite(float(row["global_sd"])) or row["global_sd"] < 0.0:
            raise ValueError(f"Invalid global_sd for k={row['k']}")
    return sorted(rows, key=lambda row: row["k"])


def promote_row_order_component(
    canonical_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    release = _read_json(canonical_path)
    if release.get("schema_version") != "1.0.0":
        raise ValueError("Row-order promotion only supports canonical schema 1.0.0")
    audit = _read_json(audit_path)
    rows = _validated_rows(audit)
    primary = next(row for row in rows if row["k"] == 10)

    release["tie_audit"]["row_order_experiment"] = {
        "tie_policy": "stable_storage_index_with_explicit_self_exclusion",
        "k": 10,
        "n_permutations": EXPECTED_PERMUTATIONS,
        "deterministic_mean": primary["global_mean"],
        "deterministic_sd": primary["global_sd"],
        "deterministic_95_interval": primary["global_interval_95"],
        "items_changed": primary["items_changed"],
        "items_changed_pct": primary["items_changed_pct"],
        "fractional_soft_mean": 1.0,
        "fractional_soft_sd": 0.0,
        "scale_rows": rows,
    }

    audit_hash = _sha256(audit_path)
    provenance = release["provenance"]
    provenance["promotion_input_sha256"]["row_order_audit"] = audit_hash
    provenance.setdefault("component_promotions", {})["row_order_audit"] = {
        "generated_by": "research/chaosnli/manifests/promote_row_order_component.py",
        "input_sha256": audit_hash,
    }
    provenance["notes"].append(
        "The row-order audit explicitly excludes self-distance before stable storage-index tie resolution."
    )
    return release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path("results/canonical_results.json"),
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output or args.canonical
    release = promote_row_order_component(args.canonical, args.audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    print(f"Promoted corrected row-order audit to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
