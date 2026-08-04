"""Experiment E002: Natural Language Exact-Symmetry Model Audit Pilot (Phase E2-A1).

Audits NLI models on Tier 1 reversible entity-renaming semantic squares across source-item-disjoint train/val/test orbit splits.
Estimates cross-orbit edge transport maps from actual model prediction pairs (X_g -> Y_g), eliminating artificial ILR perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from research.holonomy.geometry.connection import ConnectionEstimator, ParallelTransportMap
from research.holonomy.geometry.holonomy import HolonomyResult, evaluate_holonomy
from research.holonomy.geometry.parallel_transport import PathTransport
from research.holonomy.geometry.simplex_bundle import ilr_transform
from research.holonomy.natural_language.model_adapter import HuggingFaceNLIAdapter, NLIModelAdapter
from research.holonomy.natural_language.orbit_builder import OrbitBuilder
from research.holonomy.natural_language.orbit_schema import SemanticOrbit
from research.holonomy.natural_language.transforms.entity_rename import ReversibleEntityRenameTransform


from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from research.holonomy.geometry.connection import ConnectionEstimator, EstimatorIdentifiabilityError, ParallelTransportMap
from research.holonomy.geometry.holonomy import HolonomyResult, evaluate_holonomy
from research.holonomy.geometry.parallel_transport import PathTransport
from research.holonomy.geometry.simplex_bundle import ilr_transform
from research.holonomy.natural_language.model_adapter import HuggingFaceNLIAdapter, NLIModelAdapter
from research.holonomy.natural_language.orbit_builder import OrbitBuilder
from research.holonomy.natural_language.orbit_schema import SemanticOrbit
from research.holonomy.natural_language.transforms.entity_rename import ReversibleEntityRenameTransform


@dataclass
class ModelAuditResult:
    """Audit result for a model evaluated on natural language orbit splits."""

    model_name: str
    is_live_model: bool
    adapter_mode: str
    num_active_orbits: int
    text_path_closure_rate: float
    train_orbit_count: int
    test_orbit_count: int
    estimator_identifiable: bool
    min_edge_design_rank: int
    max_edge_condition_number: float
    max_transport_norm: float
    linear_is_flat: bool
    affine_is_flat: bool
    curvature_magnitude: float
    mean_held_out_return_residual: float
    artificial_curvature_detected: bool
    provenance: Dict[str, str | bool | None] = field(default_factory=dict)


def run_e002_classifier_holonomy_experiment(
    model_name: str = "roberta-large-mnli",
    use_live_model: bool = False,
    num_base_items: int = 15,
) -> List[ModelAuditResult]:
    """Audits NLI model on Tier 1 natural language reversible entity-renaming semantic squares."""
    builder = OrbitBuilder()
    transform_a = ReversibleEntityRenameTransform("rename_a", "Alice", "Bob")
    transform_b = ReversibleEntityRenameTransform("rename_b", "Charlie", "David")
    estimator = ConnectionEstimator()

    if use_live_model:
        hf_adapter = HuggingFaceNLIAdapter(model_name=model_name, use_mock_fallback=False)
        hf_adapter.load()
        predict_fn = hf_adapter.predict
        is_live = hf_adapter.is_loaded
        adapter_mode = "huggingface_live" if is_live else "failed"
        provenance = hf_adapter.get_provenance_metadata()
    else:
        mock_adapter = NLIModelAdapter()
        predict_fn = mock_adapter.predict_mock_orbit_vertices
        is_live = False
        adapter_mode = "deterministic_sha256_mock"
        provenance = {
            "model_requested": model_name,
            "model_resolved": None,
            "adapter_mode": adapter_mode,
            "is_loaded": False,
            "use_mock_fallback": True,
        }

    # Generate baseline NLI items containing active entities Alice & Charlie
    baseline_items = []
    names_a = ["Alice", "Bob"]
    names_b = ["Charlie", "David"]

    for i in range(num_base_items):
        na = names_a[i % 2]
        nb = names_b[i % 2]
        p = f"{na} and {nb} walked in section {i+1} of the park."
        h = f"{na} bought apples near {nb}'s desk."
        baseline_items.append((f"mnli_{1000+i}", p, h))

    # Construct and validate orbits
    active_orbits: List[SemanticOrbit] = []
    for uid, p, h in baseline_items:
        orbit = builder.build_square_orbit(
            orbit_id=f"orbit_{uid}",
            source_uid=uid,
            dataset="mnli",
            base_premise=p,
            base_hypothesis=h,
            transform_a=transform_a,
            transform_b=transform_b,
        )
        if orbit.is_closed:
            active_orbits.append(orbit)

    if not active_orbits:
        raise RuntimeError("No active closed orbits constructed.")

    # Split into source-item-disjoint Train (60%), Val (20%), Test (20%)
    n_total = len(active_orbits)
    n_train = max(2, int(0.6 * n_total))
    n_val = max(1, int(0.2 * n_total))

    train_orbits = active_orbits[:n_train]
    val_orbits = active_orbits[n_train : n_train + n_val]
    test_orbits = active_orbits[n_train + n_val :]

    # Gather paired model observation coordinates (X_g -> Y_g) across train orbits
    edge_pairs = {"rename_a": ([], []), "rename_b": ([], []), "rename_a_inv": ([], []), "rename_b_inv": ([], [])}

    for orb in train_orbits:
        v0 = orb.get_vertex("x0")
        v1 = orb.get_vertex("x1")
        v2 = orb.get_vertex("x2")
        v3 = orb.get_vertex("x3")

        z0 = ilr_transform(predict_fn(v0.premise, v0.hypothesis))
        z1 = ilr_transform(predict_fn(v1.premise, v1.hypothesis))
        z2 = ilr_transform(predict_fn(v2.premise, v2.hypothesis))
        z3 = ilr_transform(predict_fn(v3.premise, v3.hypothesis))

        edge_pairs["rename_a"][0].append(z0)
        edge_pairs["rename_a"][1].append(z1)

        edge_pairs["rename_b"][0].append(z1)
        edge_pairs["rename_b"][1].append(z2)

        edge_pairs["rename_a_inv"][0].append(z2)
        edge_pairs["rename_a_inv"][1].append(z3)

        edge_pairs["rename_b_inv"][0].append(z3)
        edge_pairs["rename_b_inv"][1].append(z0)

    # Estimate cross-orbit edge transport connections (Ta, Tb, Ta_inv, Tb_inv) with identifiability checks
    try:
        t_a = estimator.estimate_total_least_squares_transport("rename_a", "x0", "x1", np.array(edge_pairs["rename_a"][0]), np.array(edge_pairs["rename_a"][1]))
        t_b = estimator.estimate_total_least_squares_transport("rename_b", "x1", "x2", np.array(edge_pairs["rename_b"][0]), np.array(edge_pairs["rename_b"][1]))
        t_a_inv = estimator.estimate_total_least_squares_transport("rename_a_inv", "x2", "x3", np.array(edge_pairs["rename_a_inv"][0]), np.array(edge_pairs["rename_a_inv"][1]))
        t_b_inv = estimator.estimate_total_least_squares_transport("rename_b_inv", "x3", "x0", np.array(edge_pairs["rename_b_inv"][0]), np.array(edge_pairs["rename_b_inv"][1]))

        transports = [t_a, t_b, t_a_inv, t_b_inv]
        min_rank = min(t.metadata.get("design_rank", 2) for t in transports)
        max_cond = max(t.metadata.get("condition_number", 0.0) for t in transports)
        max_norm = max(t.metadata.get("matrix_norm", 0.0) for t in transports)

        path_transport = PathTransport(transports)
        hol_res = evaluate_holonomy("E2_A1_NaturalLanguage_Square", path_transport)

        # Evaluate held-out return residuals on test_orbits
        test_residuals = []
        A_gamma = path_transport.compute_composite_matrix()
        H_hom = path_transport.compute_homogeneous_matrix()
        b_gamma = H_hom[:2, 2]

        for orb in (test_orbits or val_orbits):
            v0 = orb.get_vertex("x0")
            z0 = ilr_transform(predict_fn(v0.premise, v0.hypothesis))
            z0_returned = np.dot(A_gamma, z0) + b_gamma
            res_norm = float(np.linalg.norm(z0_returned - z0))
            test_residuals.append(res_norm)

        mean_residual = float(np.mean(test_residuals)) if test_residuals else 0.0
        estimator_identifiable = True
        linear_flat = hol_res.linear_is_flat
        affine_flat = hol_res.affine_is_flat
        curvature_mag = hol_res.curvature_magnitude
        artif_curv = bool(not hol_res.affine_is_flat)

    except EstimatorIdentifiabilityError as err:
        estimator_identifiable = False
        min_rank = 1
        max_cond = float("inf")
        max_norm = float("nan")
        mean_residual = float("nan")
        linear_flat = False
        affine_flat = False
        curvature_mag = float("nan")
        artif_curv = False  # Rank deficiency is not evidence of artificial curvature
        provenance["identifiability_error"] = str(err)

    text_closure_rate = float(np.mean([o.is_closed for o in active_orbits]))

    result = ModelAuditResult(
        model_name=model_name,
        is_live_model=is_live,
        adapter_mode=adapter_mode,
        num_active_orbits=len(active_orbits),
        text_path_closure_rate=text_closure_rate,
        train_orbit_count=len(train_orbits),
        test_orbit_count=len(test_orbits or val_orbits),
        estimator_identifiable=estimator_identifiable,
        min_edge_design_rank=min_rank,
        max_edge_condition_number=max_cond,
        max_transport_norm=max_norm,
        linear_is_flat=linear_flat,
        affine_is_flat=affine_flat,
        curvature_magnitude=curvature_mag,
        mean_held_out_return_residual=mean_residual,
        artificial_curvature_detected=artif_curv,
        provenance=provenance,
    )

    return [result]


if __name__ == "__main__":
    res_list = run_e002_classifier_holonomy_experiment()
    print(f"E2-A1 Model Audit Pilot Executed: {res_list[0]}")

