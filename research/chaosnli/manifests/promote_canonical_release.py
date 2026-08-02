"""Promote audited ChaosNLI results into the single canonical release artifact.

The inputs are recomputation artifacts, not competing release files. This command
validates their cross-file invariants and emits ``results/canonical_results.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DISPLAY_TO_SLUG = {
    "ALBERT-xxLarge": "albert-xxlarge",
    "BART-Large": "bart-large",
    "BERT-Base": "bert-base",
    "BERT-Large": "bert-large",
    "DistilBERT": "distilbert",
    "RoBERTa-Base": "roberta-base",
    "RoBERTa-Large": "roberta-large",
    "XLNet-Base": "xlnet-base",
    "XLNet-Large": "xlnet-large",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _read_mapping(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.suffix.lower() == ".json" else _read_yaml(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_close(name: str, actual: float, expected: float, tolerance: float = 5e-6) -> None:
    if abs(actual - expected) > tolerance:
        raise ValueError(f"{name}: expected {expected}, got {actual}")


def _reference_cell(rows: list[dict[str, Any]], n_votes: int, k: int) -> dict[str, Any]:
    matches = [row for row in rows if row["n_votes"] == n_votes and row["k"] == k]
    if len(matches) != 1:
        raise ValueError(f"Expected one reference cell for n={n_votes}, k={k}; got {len(matches)}")
    return matches[0]


def build_release(
    *,
    canonical_core_path: Path,
    reference_surface_path: Path,
    geometry_path: Path,
    phase_path: Path,
    study_manifest_path: Path,
    model_manifest_path: Path,
) -> dict[str, Any]:
    canonical = _read_mapping(canonical_core_path)
    surface = _read_json(reference_surface_path)
    geometry = _read_mapping(geometry_path)
    phase = _read_json(phase_path)
    study_manifest = _read_json(study_manifest_path)
    model_manifest = _read_json(model_manifest_path)

    h1 = canonical["h1_bootstrap_paired"]
    legacy_h1 = canonical["h1_bootstrap"]
    for key in ("hh100_bootstrap_mean", "hh100_bootstrap_95ci"):
        if h1[key] != legacy_h1[key]:
            raise ValueError(f"Duplicate H1 sections disagree on {key}")
    for model_name, values in h1["models"].items():
        for key in (
            "q_paired_hm_mean",
            "delta_m_mean",
            "delta_m_95ci",
            "replicates_gt_zero",
            "q_fixed_reference",
        ):
            if values[key] != legacy_h1["models"][model_name][key]:
                raise ValueError(f"Duplicate H1 sections disagree on {model_name}.{key}")

    _require_close("HH100 paired bootstrap mean", h1["hh100_bootstrap_mean"], 0.07549)
    _require_close("HH100 direct-pair mean", canonical["hh100_simulation"]["mean"], 0.07550)

    expected_models = set(model_manifest["models"])
    found_models = {DISPLAY_TO_SLUG[name] for name in h1["models"]}
    if found_models != expected_models:
        raise ValueError(
            f"Canonical model set mismatch: expected {sorted(expected_models)}, got {sorted(found_models)}"
        )

    model_recovery: dict[str, Any] = {}
    for display_name, values in h1["models"].items():
        slug = DISPLAY_TO_SLUG[display_name]
        model_recovery[slug] = {"display_name": display_name, **values}

    _require_close(
        "BART-Large paired score",
        model_recovery["bart-large"]["q_paired_hm_mean"],
        0.01572,
    )

    surface_cells = surface["cells"]
    expected_surface_cells = len(surface["n_depths"]) * len(surface["k_list"])
    if len(surface_cells) != expected_surface_cells:
        raise ValueError(
            f"Reference surface has {len(surface_cells)} cells; expected {expected_surface_cells}"
        )
    surface_keys = {(cell["n_votes"], cell["k"]) for cell in surface_cells}
    if len(surface_keys) != expected_surface_cells:
        raise ValueError("Reference surface contains duplicate (n_votes, k) cells")
    _require_close(
        "Plug-in reference surface n=100, k=10",
        _reference_cell(surface_cells, 100, 10)["mean"],
        0.1391,
    )

    geometry_table = geometry["geometry_table_Q_G_m_G_emp"]
    if set(geometry_table) != expected_models:
        raise ValueError("Geometry table model set does not match the locked model artifact")
    for model_name, row in geometry_table.items():
        if set(row) != {"Hellinger", "JSD", "TV", "Euclidean", "Aitchison"}:
            raise ValueError(f"Geometry row for {model_name} has an unexpected metric set")

    phase_cells = phase["phase_diagram_100reps"]
    if phase.get("n_repetitions_per_cell") != 100:
        raise ValueError("Phase diagram must contain exactly 100 repetitions per cell")
    if phase.get("n_items") != 3113 or phase.get("k") != 10:
        raise ValueError("Phase diagram must use the locked N=3,113 and k=10 design")
    if len(phase_cells) != 105:
        raise ValueError(f"Expected 105 phase-diagram cells, got {len(phase_cells)}")
    phase_keys = {(cell["alpha"], cell["c"], cell["n_votes"]) for cell in phase_cells}
    if len(phase_keys) != len(phase_cells):
        raise ValueError("Phase diagram contains duplicate (alpha, c, n_votes) cells")

    varierr = canonical["varierr_validation"]
    sd_reduction_vs_null = 100.0 * (1.0 - varierr["within_profile_sd"] / varierr["null_sd_mean"])
    sd_reduction_vs_overall = 100.0 * (1.0 - varierr["within_profile_sd"] / varierr["overall_sd"])
    _require_close("VariErr permutation p-value", varierr["permutation_p_value"], 0.2045)

    ladder_cell = _reference_cell(canonical["annotation_budget_r_reference"], 100, 10)

    source_files = study_manifest["files"]
    dataset_sources = {
        key: {
            "filename": value["filename"],
            "path": value["path"],
            "sha256": value["sha256"],
            "size_bytes": value["size_bytes"],
            "row_count": value["row_count"],
        }
        for key, value in source_files.items()
    }

    promotion_inputs = {
        "canonical_core": canonical_core_path,
        "reference_surface": reference_surface_path,
        "geometry_audit": geometry_path,
        "phase_diagram": phase_path,
        "study_manifest": study_manifest_path,
        "model_manifest": model_manifest_path,
    }

    return {
        "schema_version": "1.0.0",
        "release_id": "chaosnli-canonical-2026-08-02",
        "release_status": {
            "study_1": "canonical_locked",
            "study_2a": "exploratory",
            "study_2b": "exploratory_inconclusive",
            "studies_3_and_4": "planned_not_results",
        },
        "dataset": {
            "name": "ChaosNLI-SNLI+MNLI",
            "item_count": 3113,
            "snli_item_count": 1514,
            "mnli_item_count": 1599,
            "judgments_per_item": 100,
            "canonical_label_order": ["entailment", "neutral", "contradiction"],
            "selection_scope": "selected low-original-agreement NLI items",
            "source_files": dataset_sources,
        },
        "conventions": {
            "primary_metric": "hellinger",
            "primary_k": 10,
            "tie_tolerance": 1e-7,
            "human_prior": {"family": "dirichlet", "alpha": [0.5, 0.5, 0.5]},
            "primary_overlap": "fractional_tie_aware_fuzzy_qnx",
        },
        "tie_audit": {
            "empirical_boundary_tie_pct": canonical["phase_diagram"]["empirical_chaosnli_tie_pct"],
            "row_order_experiment": canonical["row_order_experiment"],
        },
        "human_reference": {
            "hh100_direct_pair": {
                "estimand_id": "hh100_direct_pair",
                **canonical["hh100_simulation"],
            },
            "hh100_paired_focal_bootstrap": {
                "estimand_id": "hh100_paired_focal_bootstrap",
                "mean": h1["hh100_bootstrap_mean"],
                "interval_95": h1["hh100_bootstrap_95ci"],
                "n_bootstrap": 1000,
            },
            "posterior100_vs_observed_seed0": {
                "estimand_id": "posterior100_vs_observed_seed0",
                "value": ladder_cell["r_reference"],
                "note": "Single posterior-predictive 100-vote cohort versus observed graph; not the plug-in multi-seed surface.",
            },
        },
        "model_recovery": {
            "estimand_id": "model_vs_paired_posterior_cohorts",
            "k": 10,
            "metric": "hellinger",
            "empirical_stratified_null_mean": 0.00354,
            "theoretical_chance": round(10 / (3113 - 1), 8),
            "models": model_recovery,
        },
        "reference_surface": surface,
        "geometry_sensitivity": {
            "estimand_id": "model_vs_observed_empirical_graph",
            "k": 10,
            "models": geometry_table,
        },
        "phase_diagram": {
            "estimand_id": "boundary_tie_prevalence_simulation",
            "verification_status": "verified_from_promoted_100rep_artifact",
            "n_repetitions_per_cell": phase["n_repetitions_per_cell"],
            "k": phase["k"],
            "n_items": phase["n_items"],
            "empirical_chaosnli_tie_pct": phase["empirical_chaosnli_tie_pct"],
            "cells": phase_cells,
        },
        "varierr_validation": {
            "status": "exploratory_inconclusive",
            "matched_items": varierr["matched_items"],
            "multi_item_profiles": varierr["multi_item_profiles"],
            "overall_sd": varierr["overall_sd"],
            "within_profile_sd": varierr["within_profile_sd"],
            "null_sd_mean": varierr["null_sd_mean"],
            "sd_reduction_vs_null_pct": round(sd_reduction_vs_null, 1),
            "sd_reduction_vs_overall_pct": round(sd_reduction_vs_overall, 1),
            "n_permutations": varierr["n_permutations"],
            "permutation_p_value": varierr["permutation_p_value"],
        },
        "model_artifact": model_manifest,
        "provenance": {
            "generated_by": "research/chaosnli/manifests/promote_canonical_release.py",
            "promotion_input_sha256": {
                role: _sha256(path) for role, path in promotion_inputs.items()
            },
            "notes": [
                "The committed canonical JSON is the sole release-facing quantitative source.",
                "Recomputation intermediates belong under research/chaosnli/artifacts and are ignored.",
                "Exact checkpoint revisions for the supplied model logits remain unavailable.",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-core", type=Path, required=True)
    parser.add_argument("--reference-surface", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument(
        "--study-manifest",
        type=Path,
        default=Path("research/chaosnli/configs/study.yaml"),
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("research/chaosnli/configs/model_artifacts.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    release = build_release(
        canonical_core_path=args.canonical_core,
        reference_surface_path=args.reference_surface,
        geometry_path=args.geometry,
        phase_path=args.phase,
        study_manifest_path=args.study_manifest,
        model_manifest_path=args.model_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    print(f"Promoted canonical release to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
