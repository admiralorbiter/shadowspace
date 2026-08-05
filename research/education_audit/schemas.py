"""Data Schemas for Educational Counterfactual AI Audit (Phase EDU-1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AuditCase:
    """A synthetic base student profile fact sheet."""

    case_id: str
    domain: str                           # technology, math_data, humanities, leadership
    achievement_band: str                 # qualified, exceptional
    academic_level: str                   # high_school, undergraduate
    facts: List[str]                      # Ground-truth fact sheet
    target_opportunity: str
    source_hash: str


@dataclass(frozen=True)
class CounterfactualVariant:
    """A counterfactual identity variant of a base audit case."""

    variant_id: str
    case_id: str
    condition: str                        # anonymous, pronoun_masc, pronoun_fem, name_masc, name_fem
    student_name: str                     # e.g. "Student A" or "Alex"
    pronoun_subject: str                  # e.g. "they", "he", "she"
    pronoun_object: str                   # e.g. "them", "him", "her"
    pronoun_possessive: str              # e.g. "their", "his", "her"
    rendered_input: str                   # Full prompt context payload
    input_hash: str


@dataclass(frozen=True)
class GenerationRecord:
    """Record of a generated text output from a model adapter."""

    generation_id: str
    case_id: str
    variant_id: str
    condition: str
    prompt_id: str
    prompt_hash: str
    model_id: str
    model_revision: str
    parameters: Dict[str, Any]
    repeat_index: int
    output_text: str
    output_hash: str


@dataclass(frozen=True)
class EvaluationRecord:
    """Structured evaluation metrics for a single generation record."""

    generation_id: str
    case_id: str
    variant_id: str
    condition: str
    prompt_id: str
    evaluator_type: str
    rubric_version: str
    word_count: int
    recommendation_strength_score: float  # 1.0 to 5.0 scale
    factual_hallucination_count: int      # Unsupported claims beyond fact sheet
    hallucinations_per_100_words: float
    opportunity_endorsement_flag: bool    # Explicit top-tier selection recommendation
    leadership_word_count: int
    competence_word_count: int
    warmth_word_count: int


@dataclass
class AuditManifest:
    """Execution and analysis manifest for an educational counterfactual audit run."""

    timestamp_utc: str
    run_id: str
    phase: str
    git_commit_sha: str
    execution_status: str
    mock_validation_status: str
    summary: Dict[str, Any]
