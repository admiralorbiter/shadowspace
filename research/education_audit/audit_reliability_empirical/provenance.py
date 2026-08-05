"""ER-0: Evaluator Provenance Schema & Dependency Graph Validation."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


class EvaluatorProvenance:
    """Provenance record declaring evaluator properties, SHA-256 hashes, and independence dependencies."""

    def __init__(
        self,
        evaluator_id: str,
        evaluator_name: str,
        model_family: str,
        checkpoint_revision: str,
        checkpoint_sha256: str,
        training_data_revision: str,
        score_scale: List[float],
        threshold: float,
        threshold_source: str,
        is_independent: bool,
        independent_of: Optional[List[str]] = None,
    ):
        self.evaluator_id = evaluator_id
        self.evaluator_name = evaluator_name
        self.model_family = model_family
        self.checkpoint_revision = checkpoint_revision
        self.checkpoint_sha256 = checkpoint_sha256
        self.training_data_revision = training_data_revision
        self.score_scale = score_scale
        self.threshold = threshold
        self.threshold_source = threshold_source
        self.is_independent = is_independent
        self.independent_of = independent_of or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluator_id": self.evaluator_id,
            "evaluator_name": self.evaluator_name,
            "model_family": self.model_family,
            "checkpoint_revision": self.checkpoint_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "training_data_revision": self.training_data_revision,
            "score_scale": self.score_scale,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "is_independent": self.is_independent,
            "independent_of": self.independent_of,
        }


def validate_evaluator_dependency_graph(provenance_records: List[EvaluatorProvenance]) -> bool:
    """Fails closed if derived evaluators contain cyclic dependencies or claim independence incorrectly."""
    eval_map = {p.evaluator_id: p for p in provenance_records}

    for p in provenance_records:
        if not p.is_independent and not p.independent_of:
            raise ValueError(f"Non-independent evaluator {p.evaluator_id} must declare independent_of parent IDs.")

        # Check for cycles
        visited = set()
        stack = [p.evaluator_id]

        while stack:
            curr = stack.pop()
            if curr in visited:
                raise ValueError(f"Cyclic dependency detected in evaluator graph involving {curr}.")
            visited.add(curr)

            if curr in eval_map:
                for parent_id in eval_map[curr].independent_of:
                    stack.append(parent_id)

    return True
