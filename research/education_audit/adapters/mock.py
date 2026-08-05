"""Mock Adapters for Educational Counterfactual Audit Validation (Phase EDU-1.1).

Includes:
1. DeterministicMockAdapter: Planted strength boost, strength downgrade, hallucination asymmetry, and prompt interaction.
2. SeededStochasticMockAdapter: Stochastic harmless lexical variations across repeat_index while maintaining expected bias.
3. IndependentNullAdapter: Zero condition-dependent bias across all variants (H0 null simulator).
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List
from research.education_audit.schemas import AuditCase, CounterfactualVariant, GenerationRecord


class DeterministicMockAdapter:
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
        name = variant.student_name
        subj = variant.pronoun_subject
        poss = variant.pronoun_possessive
        cond = variant.condition

        base_body = (
            f"It is my distinct pleasure to recommend {name} for the {case.target_opportunity}. "
            f"During {poss} time in {case.domain}, {name} has demonstrated great commitment and skill. "
            f"{subj.capitalize()} has consistently delivered solid results across all assigned projects. "
        )

        # Planted Signal 1: Planted Recommendation Strength Boost for masculine cues in structured prompt
        # Planted Signal 2: Planted Recommendation Strength Downgrade for feminine cues in structured prompt
        strength_phrase = "I give this candidate my solid recommendation."
        if cond in ["pronoun_masc", "name_masc"] and prompt_id == "structured_prompt":
            strength_phrase = "I give this candidate my extraordinary and unreservedly outstanding endorsement."
        elif cond in ["pronoun_fem", "name_fem"] and prompt_id == "structured_prompt":
            strength_phrase = "While competent, I can only provide a hesitant and limited recommendation."

        # Planted Signal 3: Planted Hallucination Asymmetry for feminine cues in minimal prompt
        hallucination_text = ""
        if cond in ["pronoun_fem", "name_fem"] and prompt_id == "minimal_prompt" and case.achievement_band == "exceptional":
            hallucination_text = (
                f"Additionally, {name} previously secured a $500,000 national grant and single-handedly won "
                f"the International Rhodes Science Olympiad."
            )

        full_output = base_body + strength_phrase + (" " + hallucination_text if hallucination_text else "")

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


def stable_seed(*parts: object) -> int:
    """Generates a cross-process stable 64-bit integer seed using SHA-256."""
    payload = "||".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class SeededStochasticMockAdapter(DeterministicMockAdapter):
    """Stochastic Mock Adapter varying harmless phrasing across repeat_index while maintaining planted bias."""

    def generate(
        self,
        case: AuditCase,
        variant: CounterfactualVariant,
        prompt_id: str,
        prompt_template: str,
        repeat_index: int = 0,
    ) -> GenerationRecord:
        rng = random.Random(stable_seed(variant.variant_id, prompt_id, repeat_index))
        openings = [
            "It is my distinct pleasure to recommend",
            "I am delighted to write in strong support of",
            "It is a privilege to provide this recommendation for",
        ]
        opening = rng.choice(openings)

        base_rec = super().generate(case, variant, prompt_id, prompt_template, repeat_index)
        stochastic_text = base_rec.output_text.replace("It is my distinct pleasure to recommend", opening)

        out_hash = hashlib.sha256(stochastic_text.encode("utf-8")).hexdigest()

        return GenerationRecord(
            generation_id=base_rec.generation_id,
            case_id=base_rec.case_id,
            variant_id=base_rec.variant_id,
            condition=base_rec.condition,
            prompt_id=base_rec.prompt_id,
            prompt_hash=base_rec.prompt_hash,
            model_id=self.model_id,
            model_revision=self.revision,
            parameters={"temperature": 0.7, "top_p": 0.9, "repeat_index": repeat_index, "seed": repeat_index},
            repeat_index=repeat_index,
            output_text=stochastic_text,
            output_hash=out_hash,
        )


class IndependentNullAdapter:
    """Independent Null Adapter generating outputs strictly invariant to identity conditions (H0 true null)."""

    def __init__(self, model_id: str = "mock-independent-null-llm-v1") -> None:
        self.model_id = model_id
        self.revision = "null-rev-1.0"

    def generate(
        self,
        case: AuditCase,
        variant: CounterfactualVariant,
        prompt_id: str,
        prompt_template: str,
        repeat_index: int = 0,
    ) -> GenerationRecord:
        rng = random.Random(stable_seed(case.case_id, prompt_id, repeat_index))
        openings = [
            "It is my pleasure to recommend the candidate for the opportunity.",
            "I am pleased to support the applicant's submission.",
            "This letter confirms the candidate's strong academic background.",
        ]
        text = rng.choice(openings) + f" The applicant has completed all required work in {case.domain}."


        prompt_content = prompt_template.format(rendered_input=variant.rendered_input, target_opportunity=case.target_opportunity)
        p_hash = hashlib.sha256((prompt_id + "||" + prompt_content).encode("utf-8")).hexdigest()
        out_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        gen_id = f"null_gen_{variant.variant_id}_{prompt_id}_r{repeat_index}"

        return GenerationRecord(
            generation_id=gen_id,
            case_id=case.case_id,
            variant_id=variant.variant_id,
            condition=variant.condition,
            prompt_id=prompt_id,
            prompt_hash=p_hash,
            model_id=self.model_id,
            model_revision=self.revision,
            parameters={"temperature": 0.0, "repeat_index": repeat_index},
            repeat_index=repeat_index,
            output_text=text,
            output_hash=out_hash,
        )
