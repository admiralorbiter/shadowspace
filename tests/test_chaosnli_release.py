"""Integrity tests for the single committed ChaosNLI result contract."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from research.chaosnli.manifests.promote_phase_component import promote_phase_component

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = REPO_ROOT / "results" / "canonical_results.json"


def _release() -> dict:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_only_one_canonical_result_artifact_is_committed() -> None:
    artifacts = sorted(path.name for path in RESULT_PATH.parent.glob("canonical_results.*"))
    assert artifacts == ["canonical_results.json"]


def test_canonical_release_has_locked_scope_and_estimands() -> None:
    release = _release()

    assert release["schema_version"] == "1.0.0"
    assert release["dataset"]["item_count"] == 3113
    assert release["dataset"]["snli_item_count"] == 1514
    assert release["dataset"]["mnli_item_count"] == 1599
    assert release["dataset"]["canonical_label_order"] == [
        "entailment",
        "neutral",
        "contradiction",
    ]
    assert release["conventions"]["primary_metric"] == "hellinger"
    assert release["conventions"]["primary_k"] == 10
    assert release["human_reference"]["hh100_paired_focal_bootstrap"]["mean"] == 0.07549
    assert release["model_recovery"]["models"]["bart-large"]["q_paired_hm_mean"] == 0.01572
    assert len(release["model_recovery"]["models"]) == 9


def test_canonical_release_surfaces_are_complete() -> None:
    release = _release()
    surface = release["reference_surface"]
    assert len(surface["cells"]) == len(surface["n_depths"]) * len(surface["k_list"]) == 40
    phase = release["phase_diagram"]
    assert phase["verification_status"] == "verified_from_recomputed_100rep_artifact"
    assert phase["empirical_chaosnli_tie_pct"] == 49.11660777385159
    assert phase["cross_validation"]["status"] == "passed"
    assert phase["cross_validation"]["empirical_tie_pct_exact_match"] is True
    assert round(phase["cross_validation"]["cell_mean_mae_percentage_points"], 3) == 0.079
    assert release["tie_audit"]["empirical_boundary_tie_pct"] == 49.11660777385159
    row_order = release["tie_audit"]["row_order_experiment"]
    assert round(row_order["deterministic_mean"], 4) == 0.9554
    assert round(row_order["deterministic_sd"], 4) == 0.0015
    assert round(row_order["items_changed_pct"], 1) == 49.1
    assert len(phase["cells"]) == 105
    assert len(release["geometry_sensitivity"]["models"]) == 9


def test_varierr_comparisons_are_explicit_and_inconclusive() -> None:
    varierr = _release()["varierr_validation"]

    assert varierr["status"] == "exploratory_inconclusive"
    assert varierr["sd_reduction_vs_null_pct"] == 7.8
    assert varierr["sd_reduction_vs_overall_pct"] == 25.0
    assert varierr["permutation_p_value"] == 0.2045
    assert "sd_reduction_pct" not in varierr


def test_locked_paths_are_repository_relative_and_portable() -> None:
    release = _release()
    paths = [source["path"] for source in release["dataset"]["source_files"].values()] + [
        release["model_artifact"]["path"]
    ]

    assert all("\\" not in path for path in paths)
    assert all(not Path(path).is_absolute() for path in paths)


def test_phase_component_promotion_updates_phase_and_tie_audit(tmp_path: Path) -> None:
    phase_path = tmp_path / "phase.json"
    phase = {
        "n_repetitions_per_cell": 100,
        "n_items": 3113,
        "k": 10,
        "empirical_chaosnli_tie_pct": 49.11660777385159,
        "total_runtime_ms": 123.0,
        "phase_diagram_100reps": [
            {
                "alpha": alpha,
                "c": categories,
                "n_votes": n_votes,
                "mean_tie_pct": 50.0,
                "sd_tie_pct": 1.0,
            }
            for alpha, categories, n_votes in product(
                (0.1, 0.5, 1.0),
                (2, 3, 5, 7, 10),
                (3, 5, 10, 20, 30, 50, 100),
            )
        ],
    }
    phase_path.write_text(json.dumps(phase), encoding="utf-8")

    promoted = promote_phase_component(RESULT_PATH, phase_path)

    assert promoted["phase_diagram"]["verification_status"] == (
        "verified_from_recomputed_100rep_artifact"
    )
    assert promoted["tie_audit"]["empirical_boundary_tie_pct"] == phase[
        "empirical_chaosnli_tie_pct"
    ]
    assert promoted["provenance"]["component_promotions"]["phase_diagram"][
        "input_sha256"
    ]
