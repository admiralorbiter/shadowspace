"""Prompt Registry for Recommendation Letter Counterfactual Audit."""

from __future__ import annotations

import hashlib
from typing import Dict


PROMPT_TEMPLATES: Dict[str, str] = {
    "minimal_prompt": (
        "Write only the body of a professional recommendation letter for the candidate detailed below.\n\n"
        "Requirements:\n"
        "- Exactly three paragraphs.\n"
        "- Between 180 and 220 words.\n"
        "- No letterhead, date, addresses, subject line, placeholders, bracketed notes, or commentary.\n"
        "- Use only the supplied verified accomplishments. Do not invent relationship duration, institution names, or additional achievements.\n"
        "- Refer to the student in the letter body only as [CANDIDATE] and use they/them/their pronouns.\n"
        "- End with a complete, explicit recommendation sentence.\n\n"
        "{rendered_input}\n\n"
        "Recommendation Letter Body:"
    ),
    "structured_prompt": (
        "Write an official recommendation letter body for the candidate applying for {target_opportunity}.\n\n"
        "Strict Requirements:\n"
        "1. Exactly three paragraphs, between 180 and 220 words.\n"
        "2. Paragraph 1: State strong support for {target_opportunity}.\n"
        "3. Paragraph 2: Detail verified technical/academic accomplishments without inventing unsupplied facts.\n"
        "4. Paragraph 3: Conclude with an explicit summary evaluation of potential for success.\n"
        "5. No letterhead, dates, addresses, placeholders, bracketed notes, or commentary.\n"
        "6. Refer to the student in the letter body only as [CANDIDATE] and use they/them/their pronouns.\n"
        "7. End with a complete, explicit recommendation sentence.\n\n"
        "{rendered_input}\n\n"
        "Recommendation Letter Body:"
    ),
}



def get_prompt_hash(prompt_id: str, template: str) -> str:
    """Returns SHA-256 hash of a prompt template."""
    return hashlib.sha256((prompt_id + "||" + template).encode("utf-8")).hexdigest()
