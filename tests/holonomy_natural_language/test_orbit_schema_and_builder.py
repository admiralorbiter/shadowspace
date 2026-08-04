"""Unit tests for Natural Language Orbit Schema & Reversible Entity Rename Builder."""

import pytest
from research.holonomy.natural_language.orbit_builder import OrbitBuilder
from research.holonomy.natural_language.transforms.entity_rename import ReversibleEntityRenameTransform


def test_reversible_entity_rename_orbit_closure():
    builder = OrbitBuilder()
    transform_a = ReversibleEntityRenameTransform("rename_a", "Alice", "Bob")
    transform_b = ReversibleEntityRenameTransform("rename_b", "Charlie", "David")

    base_p = "Alice and Charlie went to the store."
    base_h = "Alice bought apples."

    orbit = builder.build_square_orbit(
        orbit_id="orbit_001",
        source_uid="mnli_101",
        dataset="mnli",
        base_premise=base_p,
        base_hypothesis=base_h,
        transform_a=transform_a,
        transform_b=transform_b,
    )

    assert orbit.is_closed is True
    assert len(orbit.vertices) == 4
    assert len(orbit.edges) == 4

    # Vertex 1: Alice -> Bob
    assert orbit.vertices["x1"].premise == "Bob and Charlie went to the store."
    # Vertex 2: Charlie -> David
    assert orbit.vertices["x2"].premise == "Bob and David went to the store."
    # Vertex 3: Bob -> Alice
    assert orbit.vertices["x3"].premise == "Alice and David went to the store."
