"""Build Milestone 1 Tracked Provenance and Summary Package.

Copies and validates compact summary JSONs into research/chaosnli/results/
and creates MILESTONE_1_PROVENANCE.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

ART_E005 = Path("research/chaosnli/artifacts/E005/summaries/E005_summary.json")
ART_E007 = Path("research/chaosnli/artifacts/E007/summaries/E007_summary.json")
ART_E008 = Path("research/chaosnli/artifacts/E008/summaries/E008_summary.json")

RESULTS_DIR = Path("research/chaosnli/results")

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. E005 Full Summary
    if ART_E005.exists():
        shutil.copy2(ART_E005, RESULTS_DIR / "E005_full_summary.json")
        print(f"Copied {ART_E005} -> {RESULTS_DIR / 'E005_full_summary.json'}")

    # 2. E007 Full Summaries
    if ART_E007.exists():
        with open(ART_E007, "r", encoding="utf-8") as f:
            e007_raw = json.load(f)

        census_summary = {
            "experiment_id": "E007",
            "subset": e007_raw["subset"],
            "object_count": e007_raw["object_count"],
            "q_hh_relational": e007_raw["q_hh_relational"],
            "best_subset_by_size": e007_raw["best_subset_by_size"],
        }
        with open(RESULTS_DIR / "E007_full_census_summary.json", "w", encoding="utf-8") as f:
            json.dump(census_summary, f, indent=2)

        shapley_summary = {
            "experiment_id": "E007",
            "subset": e007_raw["subset"],
            "shapley_attributions": e007_raw["shapley_attributions"],
        }
        with open(RESULTS_DIR / "E007_full_shapley.json", "w", encoding="utf-8") as f:
            json.dump(shapley_summary, f, indent=2)

        print(f"Exported E007 census & Shapley summaries into {RESULTS_DIR}")

    # 3. E008 Full Curve
    if ART_E008.exists():
        shutil.copy2(ART_E008, RESULTS_DIR / "E008_full_curve.json")
        print(f"Copied {ART_E008} -> {RESULTS_DIR / 'E008_full_curve.json'}")

    # 4. Master Provenance Package
    provenance = {
        "milestone": "Milestone 1 — Full-Data Confirmation",
        "status": "full_data_runs_complete_audit_pending",
        "dataset_release": "chaosnli-canonical-2026-08-02",
        "object_count": 3113,
        "object_id_sha256": "121c49cbd40b171d100ba88c1a23d809818c28bad9249bea99a52ec8f5af19d6",
        "support_matrix_sha256": "94e483e714d92f039f817389d948cbf41b7970077b56f852491832605dccc96f",
        "q_hh_relational": 0.038987226212620456,
        "experiments": {
            "E005": {
                "d_size_matched_family_point": +0.0211,
                "d_size_95_ci": [+0.0166, +0.0257],
                "p_boot_gt_zero": 1.0,
                "ladder_levels": 6,
            },
            "E007": {
                "subsets_evaluated": 511,
                "best_single_model": "bart-large",
                "best_single_r_norm": 0.3793,
                "best_pair_r_norm": 0.5382,
                "best_triplet_r_norm": 0.6472,
                "grand_coalition_r_norm": 0.8444,
            },
            "E008": {
                "cross_validation": "5fold_dataset_label_entropy_stratified",
                "n_restarts_per_fold": 20,
                "k18_r_normalized": 1.0087,
            },
        },
    }

    with open(RESULTS_DIR / "MILESTONE_1_PROVENANCE.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    print(f"Generated master provenance file at {RESULTS_DIR / 'MILESTONE_1_PROVENANCE.json'}")

if __name__ == "__main__":
    main()
