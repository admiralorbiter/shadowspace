"""Formal Formulas and NLI Pairs for Phase E0 Simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class FormalState:
    """Formal NLI state represented by premise, hypothesis, and contextual metadata."""

    id: str
    premise_type: str  # e.g., "ALL_P_ARE_Q", "SOME_P_ARE_Q", "NO_P_ARE_Q"
    hypothesis_type: str  # e.g., "EXISTS_P_Q", "FORALL_P_Q"
    predicate_P: str
    predicate_Q: str
    context_features: Tuple[float, ...] = field(default_factory=tuple)

    def swap_predicates(self) -> FormalState:
        """Transformation g_swap: swap P and Q."""
        return FormalState(
            id=f"{self.id}_swap",
            premise_type=self.premise_type,
            hypothesis_type=self.hypothesis_type,
            predicate_P=self.predicate_Q,
            predicate_Q=self.predicate_P,
            context_features=self.context_features,
        )

    def negate_hypothesis(self) -> FormalState:
        """Transformation g_neg: negate hypothesis."""
        new_hyp = f"NOT_{self.hypothesis_type}"
        return FormalState(
            id=f"{self.id}_neg",
            premise_type=self.premise_type,
            hypothesis_type=new_hyp,
            predicate_P=self.predicate_P,
            predicate_Q=self.predicate_Q,
            context_features=self.context_features,
        )
