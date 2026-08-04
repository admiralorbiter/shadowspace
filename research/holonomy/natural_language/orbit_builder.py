"""Orbit Builder for constructing and validating closed natural language semantic squares."""

from __future__ import annotations

from typing import Tuple
from research.holonomy.natural_language.orbit_schema import SemanticEdge, SemanticOrbit, SemanticVertex
from research.holonomy.natural_language.transforms.base import BaseTransform


class OrbitBuilder:
    """Constructs and strictly validates closed natural language 4-corner semantic squares."""

    def build_square_orbit(
        self,
        orbit_id: str,
        source_uid: str,
        dataset: str,
        base_premise: str,
        base_hypothesis: str,
        transform_a: BaseTransform,
        transform_b: BaseTransform,
    ) -> SemanticOrbit:
        """Builds 4-corner semantic square x0 -a-> x1 -b-> x2 -a^-1-> x3 -b^-1-> x0."""
        # Vertex 0
        v0 = SemanticVertex("x0", base_premise, base_hypothesis)

        # Path 1: x0 -a-> x1 -b-> x2_ab
        p1, h1 = transform_a.apply(base_premise, base_hypothesis)
        v1 = SemanticVertex("x1", p1, h1)

        p2_ab, h2_ab = transform_b.apply(p1, h1)
        v2 = SemanticVertex("x2", p2_ab, h2_ab)

        # Path 2: x0 -b-> x1_b -a-> x2_ba (Commutativity check: x2_ab == x2_ba)
        p1_b, h1_b = transform_b.apply(base_premise, base_hypothesis)
        p2_ba, h2_ba = transform_a.apply(p1_b, h1_b)
        path_commutes = (p2_ab.strip() == p2_ba.strip()) and (h2_ab.strip() == h2_ba.strip())

        # Vertex 3 = a^-1(x2)
        p3, h3 = transform_a.invert(p2_ab, h2_ab)
        v3 = SemanticVertex("x3", p3, h3)

        # Closing verification: b^-1(x3) == x0
        p0_check, h0_check = transform_b.invert(p3, h3)
        closure_valid = (p0_check.strip() == base_premise.strip()) and (h0_check.strip() == base_hypothesis.strip())

        # Check generator activity if supported
        act_a = getattr(transform_a, "is_active_on_pair", lambda p, h: True)(base_premise, base_hypothesis)
        act_b = getattr(transform_b, "is_active_on_pair", lambda p, h: True)(base_premise, base_hypothesis)

        # Orbit is valid and closed iff path_commutes, closure_valid, and both transforms active
        is_closed = bool(path_commutes and closure_valid and act_a and act_b)

        orbit = SemanticOrbit(
            orbit_id=orbit_id,
            source_uid=source_uid,
            dataset=dataset,
            base_premise=base_premise,
            base_hypothesis=base_hypothesis,
            vertices={"x0": v0, "x1": v1, "x2": v2, "x3": v3},
            edges=[
                SemanticEdge("x0", "x1", transform_a.name, False),
                SemanticEdge("x1", "x2", transform_b.name, False),
                SemanticEdge("x2", "x3", transform_a.name, True),
                SemanticEdge("x3", "x0", transform_b.name, True),
            ],
            is_closed=is_closed,
        )
        return orbit
