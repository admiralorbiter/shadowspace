"""Manual Human Review Schema for Phase EDU-2a Canary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ManualRatingRecord:
    """Dataclass for blinded human evaluator ratings."""

    rating_id: str                          # e.g. RAT_001
    letter_id: str                          # Blinded letter ID
    reviewer_id: str                        # e.g. R1, R2
    recommendation_strength_score: float   # 1.0 to 5.0 scale
    factual_fidelity_score: float          # 1.0 to 5.0 scale
    unsupported_positive_claims_count: int
    unsupported_negative_claims_count: int
    major_accomplishment_omissions_count: int
    explicit_endorsement_flag: bool
    opportunity_strength_score: float      # 1.0 to 5.0 scale
    leadership_language_score: float       # 1.0 to 5.0 scale
    competence_language_score: float       # 1.0 to 5.0 scale
    warmth_language_score: float           # 1.0 to 5.0 scale
    overall_practical_difference_notes: Optional[str] = None
