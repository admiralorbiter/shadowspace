"""Cayley Complex (Semantic Transformation Complex).

Builds a 2-dimensional complex where:
- 0-Cells (Vertices): Semantic states x
- 1-Cells (Edges): Elementary transformation edges g: x -> gx
- 2-Cells (Faces): Polygons representing closed loops arising from path equivalences
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from research.holonomy.algebra.generators import Generator, TransformationWord
from research.holonomy.algebra.relations import Relation


@dataclass(frozen=True)
class Edge:
    """Directed edge in the Cayley Complex."""

    source_id: str
    target_id: str
    generator: Generator

    def __repr__(self) -> str:
        return f"Edge({self.source_id} --{self.generator.name}--> {self.target_id})"


@dataclass
class ClosedLoop:
    """A closed 1-cycle (sequence of directed edges starting and ending at the same vertex)."""

    name: str
    start_vertex: str
    edges: List[Edge]

    def is_valid(self) -> bool:
        if not self.edges:
            return False
        curr = self.start_vertex
        for edge in self.edges:
            if edge.source_id != curr:
                return False
            curr = edge.target_id
        return curr == self.start_vertex


class CayleyComplex:
    """2-dimensional Cayley Complex representing semantic states, transformations, and closed loops."""

    def __init__(self) -> None:
        self.vertices: Dict[str, Any] = {}  # vertex_id -> semantic state object
        self.edges: List[Edge] = []
        self.adjacency: Dict[str, List[Edge]] = {}
        self.relations: List[Relation] = []
        self.loops: List[ClosedLoop] = []

    def add_vertex(self, vertex_id: str, state: Any) -> None:
        """Add a semantic state vertex."""
        if vertex_id not in self.vertices:
            self.vertices[vertex_id] = state
            self.adjacency[vertex_id] = []

    def add_edge(self, source_id: str, target_id: str, generator: Generator) -> Edge:
        """Add a transformation edge."""
        edge = Edge(source_id, target_id, generator)
        self.edges.append(edge)
        if source_id in self.adjacency:
            self.adjacency[source_id].append(edge)
        else:
            self.adjacency[source_id] = [edge]
        return edge

    def add_relation_loop(self, loop: ClosedLoop) -> None:
        """Register a closed 2-cell loop."""
        if loop.is_valid():
            self.loops.append(loop)

    def build_from_orbit(
        self,
        initial_states: Dict[str, Any],
        generators: Sequence[Generator],
        max_depth: int = 2,
    ) -> None:
        """Generates the transformation orbit graph up to max_depth."""
        for v_id, state in initial_states.items():
            self.add_vertex(v_id, state)

        frontier = list(initial_states.keys())

        for _ in range(max_depth):
            next_frontier = []
            for curr_id in frontier:
                curr_state = self.vertices[curr_id]
                for g in generators:
                    next_state = g(curr_state)
                    # Simple state representation string ID if available
                    next_id = getattr(next_state, "id", None) or f"{curr_id}_{g.name}"
                    self.add_vertex(next_id, next_state)
                    self.add_edge(curr_id, next_id, g)
                    next_frontier.append(next_id)
            frontier = next_frontier
