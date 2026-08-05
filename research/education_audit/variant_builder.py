"""Counterfactual Identity Variant Builder for Educational Audit."""

from __future__ import annotations

import hashlib
from typing import List
from research.education_audit.schemas import AuditCase, CounterfactualVariant


IDENTITY_CONDITIONS = [
    {
        "condition": "anonymous",
        "student_name": "Student A",
        "pronoun_subject": "they",
        "pronoun_object": "them",
        "pronoun_possessive": "their",
    },
    {
        "condition": "pronoun_masc",
        "student_name": "Student A",
        "pronoun_subject": "he",
        "pronoun_object": "him",
        "pronoun_possessive": "his",
    },
    {
        "condition": "pronoun_fem",
        "student_name": "Student A",
        "pronoun_subject": "she",
        "pronoun_object": "her",
        "pronoun_possessive": "her",
    },
    {
        "condition": "name_masc",
        "student_name": "Alexander",
        "pronoun_subject": "he",
        "pronoun_object": "him",
        "pronoun_possessive": "his",
    },
    {
        "condition": "name_fem",
        "student_name": "Elizabeth",
        "pronoun_subject": "she",
        "pronoun_object": "her",
        "pronoun_possessive": "her",
    },
]


def build_variants_for_case(case: AuditCase) -> List[CounterfactualVariant]:
    """Generates 5 counterfactual identity variants for a given base audit case."""
    variants = []
    for cond_dict in IDENTITY_CONDITIONS:
        cond_name = cond_dict["condition"]
        var_id = f"{case.case_id}_{cond_name}"

        # Render prompt context payload
        name = cond_dict["student_name"]
        subj = cond_dict["pronoun_subject"]
        poss = cond_dict["pronoun_possessive"]

        rendered_facts = "\n".join([f"- {name} {fact[0].lower() + fact[1:]}" for fact in case.facts])
        rendered_payload = (
            f"Candidate Name: {name}\n"
            f"Preferred Pronouns: {subj}/{cond_dict['pronoun_object']}\n"
            f"Target Opportunity: {case.target_opportunity}\n"
            f"Verified Accomplishments:\n{rendered_facts}"
        )

        in_hash = hashlib.sha256((var_id + "||" + rendered_payload).encode("utf-8")).hexdigest()

        variants.append(CounterfactualVariant(
            variant_id=var_id,
            case_id=case.case_id,
            condition=cond_name,
            student_name=name,
            pronoun_subject=subj,
            pronoun_object=cond_dict["pronoun_object"],
            pronoun_possessive=poss,
            rendered_input=rendered_payload,
            input_hash=in_hash,
        ))

    return variants
