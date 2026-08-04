"""Schema definitions for Natural Language Semantic Orbits & Vertices (Phase E2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SemanticVertex:
    """A natural language vertex in a semantic transformation orbit."""

    vertex_id: str
    premise: str
    hypothesis: str
    canonical_formula_id: str = ""

    @property
    def key(self) -> str:
        return f"[{self.premise} |= {self.hypothesis}]"


@dataclass(frozen=True)
class SemanticEdge:
    """An elementary semantic transformation edge between vertices."""

    source_id: str
    target_id: str
    generator_name: str
    is_inverse: bool = False


@dataclass
class SemanticOrbit:
    """A closed natural language semantic square or orbit."""

    orbit_id: str
    source_uid: str
    dataset: str
    base_premise: str
    base_hypothesis: str
    vertices: Dict[str, SemanticVertex] = field(default_factory=dict)
    edges: List[SemanticEdge] = field(default_factory=list)
    is_closed: bool = True

    def get_vertex(self, vertex_id: str) -> SemanticVertex:
        return self.vertices[vertex_id]
