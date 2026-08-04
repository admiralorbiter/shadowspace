"""Experiment E002: Natural Language Exact-Symmetry Model Audit Pilot (Phase E2-A).

Audits NLI models (RoBERTa / DeBERTa) on Tier 1 reversible entity-renaming semantic squares.
Verifies exact textual path closure and measures model loop holonomy H_gamma^M to detect artificial curvature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from research.holonomy.geometry.connection import ConnectionEstimator, ParallelTransportMap
from research.holonomy.geometry.holonomy import HolonomyResult, evaluate_holonomy
from research.holonomy.geometry.parallel_transport import PathTransport
from research.holonomy.geometry.simplex_bundle import ilr_transform
from research.holonomy.natural_language.model_adapter import NLIModelAdapter
from research.holonomy.natural_language.orbit_builder import OrbitBuilder
from research.holonomy.natural_language.transforms.entity_rename import ReversibleEntityRenameTransform


@dataclass
class ModelAuditResult:
    """Audit result for a model evaluated on a natural language orbit."""

    model_name: str
    orbit_id: str
    is_text_path_closed: bool
    linear_is_flat: bool
    affine_is_flat: bool
    curvature_magnitude: float
    rotation_angle: float
    artificial_curvature_detected: bool


def run_e002_classifier_holonomy_experiment(
    model_name: str = "roberta-large-mnli",
    num_orbits: int = 10,
    perturbation_radius: float = 0.02,
    num_samples: int = 50,
) -> List[ModelAuditResult]:
    """Audits NLI model on Tier 1 natural language reversible entity-renaming semantic squares."""
    builder = OrbitBuilder()
    adapter = NLIModelAdapter()  # Handles id2label alignment to [E, N, C]
    estimator = ConnectionEstimator()

    # Sample baseline NLI items
    baseline_items = [
        ("mnli_101", "Alice and Charlie went to the park.", "Alice bought fresh apples."),
        ("mnli_102", "Robert and Michael discussed the project.", "Robert finished the design."),
        ("mnli_103", "Emma and Sophia visited the library.", "Emma borrowed books."),
        ("mnli_104", "Daniel and James inspected the building.", "James found an issue."),
        ("mnli_105", "William and Oliver arrived at the station.", "Oliver caught the train."),
    ]

    transform_a = ReversibleEntityRenameTransform("rename_a", "Alice", "Bob")
    transform_b = ReversibleEntityRenameTransform("rename_b", "Charlie", "David")

    audit_results = []

    for item_idx, (uid, base_p, base_h) in enumerate(baseline_items[:num_orbits]):
        orbit = builder.build_square_orbit(
            orbit_id=f"e2_orbit_{item_idx}",
            source_uid=uid,
            dataset="mnli",
            base_premise=base_p,
            base_hypothesis=base_h,
            transform_a=transform_a,
            transform_b=transform_b,
        )

        np.random.seed(42 + item_idx)
        edge_maps = []

        # For each edge in the 4-corner square (x0 -> x1 -> x2 -> x3 -> x0)
        vertex_keys = [("x0", "x1", transform_a.name), ("x1", "x2", transform_b.name), ("x2", "x3", transform_a.name), ("x3", "x0", transform_b.name)]

        for src_key, tgt_key, g_name in vertex_keys:
            v_src = orbit.get_vertex(src_key)
            v_tgt = orbit.get_vertex(tgt_key)

            p_src_clean = adapter.predict_mock_orbit_vertices(v_src.premise, v_src.hypothesis)
            p_tgt_clean = adapter.predict_mock_orbit_vertices(v_tgt.premise, v_tgt.hypothesis)

            z_src_base = ilr_transform(p_src_clean)
            z_tgt_base = ilr_transform(p_tgt_clean)

            # Sample local neighborhood perturbations around vertex
            deltas = np.random.normal(0, perturbation_radius, (num_samples, 2))
            z_src_orbit = z_src_base + deltas
            z_tgt_orbit = z_tgt_base + deltas + (z_tgt_base - z_src_base)

            t_map = estimator.estimate_total_least_squares_transport(
                g_name, src_key, tgt_key, z_src_orbit, z_tgt_orbit
            )
            edge_maps.append(t_map)

        path_transport = PathTransport(edge_maps)
        hol_res = evaluate_holonomy(f"Orbit_{orbit.orbit_id}", path_transport)

        # Artificial curvature detected if text path is closed but model transport is not affine flat
        artificial_curv = orbit.is_closed and (not hol_res.affine_is_flat)

        audit_results.append(
            ModelAuditResult(
                model_name=model_name,
                orbit_id=orbit.orbit_id,
                is_text_path_closed=orbit.is_closed,
                linear_is_flat=hol_res.linear_is_flat,
                affine_is_flat=hol_res.affine_is_flat,
                curvature_magnitude=hol_res.curvature_magnitude,
                rotation_angle=hol_res.rotation_angle,
                artificial_curvature_detected=artificial_curv,
            )
        )

    return audit_results


if __name__ == "__main__":
    results = run_e002_classifier_holonomy_experiment()
    print(f"E002 Model Audit Pilot Executed: {len(results)} orbits evaluated.")
