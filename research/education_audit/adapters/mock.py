"""Planted-Bias Mock Adapter for Educational Counterfactual Audit Validation."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List
from research.education_audit.schemas import AuditCase, CounterfactualVariant, GenerationRecord


class MockEducationAdapter:
    """Deterministic Mock Adapter with planted null and planted bias signals for unit testing."""

    def __init__(self, model_id: str = "mock-education-llm-v1") -> None:
        self.model_id = model_id
        self.revision = "mock-rev-1.0"

    def generate(
        self,
        case: AuditCase,
        variant: CounterfactualVariant,
        prompt_id: str,
        prompt_template: str,
        repeat_index: int = 0,
    ) -> GenerationRecord:
        """Generates deterministic mock text with planted bias signals based on variant and prompt."""
        name = variant.student_name
        subj = variant.pronoun_subject
        poss = variant.pronoun_possessive
        cond = variant.condition

        # Base text template
        base_body = (
            f"It is my distinct pleasure to recommend {name} for the {case.target_opportunity}. "
            f"During {poss} time in {case.domain}, {name} has demonstrated great commitment and skill. "
            f"{subj.capitalize()} has consistently delivered solid results across all assigned projects. "
        )


        # Planted Signal 1: Planted Recommendation Strength Shift for masculine cues in structured prompt
        strength_adjective = "strong"
        if cond in ["pronoun_masc", "name_masc"] and prompt_id == "structured_prompt":
            strength_adjective = "extraordinary and unreservedly outstanding"
        elif cond in ["pronoun_fem", "name_fem"]:
            strength_adjective = "satisfactory and competent"

        recommendation_sentence = (
            f"I give {name} my {strength_adjective} recommendation for admission into the program. "
        )

        # Planted Signal 2: Planted Hallucination Asymmetry for feminine cues in minimal prompt
        hallucination_text = ""
        if cond in ["pronoun_fem", "name_fem"] and prompt_id == "minimal_prompt" and case.achievement_band == "exceptional":
            hallucination_text = (
                f"Additionally, {name} previously won the International Rhodes Science Olympiad and single-handedly "
                f"secured a $500,000 national grant for the department."
            )

        full_output = base_body + recommendation_sentence + hallucination_text

        prompt_content = prompt_template.format(rendered_input=variant.rendered_input, target_opportunity=case.target_opportunity)
        p_hash = hashlib.sha256((prompt_id + "||" + prompt_content).encode("utf-8")).hexdigest()
        out_hash = hashlib.sha256(full_output.encode("utf-8")).hexdigest()
        gen_id = f"gen_{variant.variant_id}_{prompt_id}_r{repeat_index}"

        return GenerationRecord(
            generation_id=gen_id,
            case_id=case.case_id,
            variant_id=variant.variant_id,
            condition=cond,
            prompt_id=prompt_id,
            prompt_hash=p_hash,
            model_id=self.model_id,
            model_revision=self.revision,
            parameters={"temperature": 0.0, "top_p": 1.0, "repeat_index": repeat_index},
            repeat_index=repeat_index,
            output_text=full_output,
            output_hash=out_hash,
        )
