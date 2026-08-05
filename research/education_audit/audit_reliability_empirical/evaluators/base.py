"""Base protocol and provenance registration for evaluators."""

from __future__ import annotations

from typing import Protocol
from research.education_audit.audit_reliability_empirical.provenance import EvaluatorProvenance


class EmpiricalAgencyEvaluator(Protocol):
    """Protocol for empirical agency evaluators."""
    provenance: EvaluatorProvenance

    def predict_score(self, text: str) -> float:
        ...
