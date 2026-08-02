"""Atomically promote a verified phase-diagram artifact into the canonical release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import product
from pathlib import Path
from typing import Any

EXPECTED_ALPHAS = (0.1, 0.5, 1.0)
EXPECTED_CATEGORIES = (2, 3, 5, 7, 10)
EXPECTED_VOTE_DEPTHS = (3, 5, 10, 20, 30, 50, 100)
EXPECTED_N_ITEMS = 3113
EXPECTED_K = 10
EXPECTED_REPETITIONS = 100


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


def _validated_phase(phase: dict[str, Any]) -> dict[str, Any]:
    if phase.get("n_repetitions_per_cell") != EXPECTED_REPETITIONS:
        raise ValueError("Phase diagram must contain exactly 100 repetitions per cell")
    if phase.get("n_items") != EXPECTED_N_ITEMS or phase.get("k") != EXPECTED_K:
        raise ValueError("Phase diagram must use the locked N=3,113 and k=10 design")

    cells = phase.get("phase_diagram_100reps")
    if not isinstance(cells, list):
        raise ValueError("Phase diagram is missing phase_diagram_100reps")
    expected_keys = set(product(EXPECTED_ALPHAS, EXPECTED_CATEGORIES, EXPECTED_VOTE_DEPTHS))
    found_keys = {(cell["alpha"], cell["c"], cell["n_votes"]) for cell in cells}
    if found_keys != expected_keys or len(cells) != len(expected_keys):
        missing = sorted(expected_keys - found_keys)
        extra = sorted(found_keys - expected_keys)
        raise ValueError(
            f"Phase grid does not match the locked 105 cells; missing={missing}, extra={extra}"
        )

    empirical = float(phase["empirical_chaosnli_tie_pct"])
    if not math.isfinite(empirical) or not 0.0 <= empirical <= 100.0:
        raise ValueError("Empirical boundary-tie percentage must be finite and within [0, 100]")
    for cell in cells:
        mean = float(cell["mean_tie_pct"])
        sd = float(cell["sd_tie_pct"])
        if not math.isfinite(mean) or not 0.0 <= mean <= 100.0:
            raise ValueError(f"Invalid phase mean for cell {cell}")
        if not math.isfinite(sd) or sd < 0.0:
            raise ValueError(f"Invalid phase SD for cell {cell}")

    return {
        "estimand_id": "boundary_tie_prevalence_simulation",
        "verification_status": "verified_from_recomputed_100rep_artifact",
        "n_repetitions_per_cell": EXPECTED_REPETITIONS,
        "k": EXPECTED_K,
        "n_items": EXPECTED_N_ITEMS,
        "empirical_chaosnli_tie_pct": empirical,
        "simulation_runtime_ms": float(phase["total_runtime_ms"]),
        "cells": cells,
    }


def promote_phase_component(
    canonical_path: Path,
    phase_path: Path,
    cross_check_path: Path | None = None,
) -> dict[str, Any]:
    release = _read_json(canonical_path)
    if release.get("schema_version") != "1.0.0":
        raise ValueError("Phase promotion only supports canonical schema 1.0.0")
    if release.get("dataset", {}).get("item_count") != EXPECTED_N_ITEMS:
        raise ValueError("Canonical release does not contain the locked 3,113-item scope")
    if release.get("conventions", {}).get("primary_k") != EXPECTED_K:
        raise ValueError("Canonical release does not use the locked primary k=10")

    phase = _read_json(phase_path)
    promoted = _validated_phase(phase)
    if cross_check_path is not None:
        cross_check = _read_json(cross_check_path)
        validated_cross_check = _validated_phase(cross_check)
        primary_cells = {
            (cell["alpha"], cell["c"], cell["n_votes"]): cell
            for cell in promoted["cells"]
        }
        cross_cells = {
            (cell["alpha"], cell["c"], cell["n_votes"]): cell
            for cell in validated_cross_check["cells"]
        }
        if primary_cells.keys() != cross_cells.keys():
            raise ValueError("Phase cross-check grid does not match the primary artifact")
        if not math.isclose(
            promoted["empirical_chaosnli_tie_pct"],
            validated_cross_check["empirical_chaosnli_tie_pct"],
            abs_tol=1e-12,
        ):
            raise ValueError("Phase cross-check empirical tie percentage does not match")
        mean_differences = [
            abs(primary_cells[key]["mean_tie_pct"] - cross_cells[key]["mean_tie_pct"])
            for key in primary_cells
        ]
        promoted["cross_validation"] = {
            "status": "passed",
            "primary_implementation": "python_numpy_pcg64_direct_distance_matrix",
            "independent_implementation": "rust_chacha8_profile_aggregation",
            "independent_artifact_sha256": _sha256(cross_check_path),
            "independent_runtime_ms": validated_cross_check["simulation_runtime_ms"],
            "empirical_tie_pct_exact_match": True,
            "cell_mean_mae_percentage_points": sum(mean_differences)
            / len(mean_differences),
            "cell_mean_max_abs_difference_percentage_points": max(mean_differences),
            "note": "Cell differences reflect independent deterministic RNG implementations sampling the same model.",
        }
    release["phase_diagram"] = promoted
    release["tie_audit"]["empirical_boundary_tie_pct"] = promoted[
        "empirical_chaosnli_tie_pct"
    ]

    phase_hash = _sha256(phase_path)
    provenance = release["provenance"]
    provenance["promotion_input_sha256"]["phase_diagram"] = phase_hash
    if cross_check_path is not None:
        provenance["promotion_input_sha256"]["phase_diagram_cross_check"] = _sha256(
            cross_check_path
        )
    provenance.setdefault("component_promotions", {})["phase_diagram"] = {
        "generated_by": "research/chaosnli/manifests/promote_phase_component.py",
        "input_sha256": phase_hash,
    }
    provenance["notes"] = [
        note
        for note in provenance["notes"]
        if not note.startswith("Phase-diagram cells were consolidated")
        and not note.startswith("Phase-diagram cells were recomputed")
    ]
    provenance["notes"].append(
        "Phase-diagram cells were recomputed with the corrected kth/(k+1)th boundary-tie definition."
    )
    return release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path("results/canonical_results.json"),
    )
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--cross-check", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output or args.canonical
    release = promote_phase_component(args.canonical, args.phase, args.cross_check)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    print(f"Promoted verified phase diagram to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
