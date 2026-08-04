"""Natural Language Semantic Orbits & Classifier Audits (Phase E2)."""

from research.holonomy.natural_language.human_posterior import sample_dirichlet_human_posterior
from research.holonomy.natural_language.model_adapter import NLIModelAdapter
from research.holonomy.natural_language.orbit_builder import OrbitBuilder
from research.holonomy.natural_language.orbit_schema import SemanticEdge, SemanticOrbit, SemanticVertex
from research.holonomy.natural_language.transforms.entity_rename import ReversibleEntityRenameTransform

__all__ = [
    "SemanticVertex",
    "SemanticEdge",
    "SemanticOrbit",
    "OrbitBuilder",
    "ReversibleEntityRenameTransform",
    "sample_dirichlet_human_posterior",
    "NLIModelAdapter",
]
