"""Phase EDU-1 Mock Generation & Evaluation Runner."""

from __future__ import annotations

import json
import os

from research.education_audit.adapters.mock import MockEducationAdapter
from research.education_audit.case_builder import build_synthetic_audit_cases
from research.education_audit.evaluation.rubric import evaluate_generation
from research.education_audit.prompt_registry import PROMPT_TEMPLATES
from research.education_audit.schemas import GenerationRecord
from research.education_audit.variant_builder import build_variants_for_case


def run_edu1_generation_and_eval(
    out_dir: str = "results/education_audit/edu_1",
    n_repeats: int = 3,
) -> str:
    """Executes EDU-1 generation and evaluation over 240 counterfactual letter combinations."""
    os.makedirs(out_dir, exist_ok=True)
    adapter = MockEducationAdapter()
    cases = build_synthetic_audit_cases()

    gen_records = []
    eval_records = []

    for case in cases:
        variants = build_variants_for_case(case)
        for var in variants:
            for prompt_id, prompt_tmpl in PROMPT_TEMPLATES.items():
                for rep in range(n_repeats):
                    g_rec = adapter.generate(case, var, prompt_id, prompt_tmpl, repeat_index=rep)
                    e_rec = evaluate_generation(case, g_rec)
                    gen_records.append(g_rec)
                    eval_records.append(e_rec)

    # Export generations.jsonl
    gen_file = os.path.join(out_dir, "generations.jsonl")
    with open(gen_file, "w", encoding="utf-8") as f:
        for g in gen_records:
            f.write(json.dumps(g.__dict__) + "\n")

    # Export evaluations.jsonl
    eval_file = os.path.join(out_dir, "evaluations.jsonl")
    with open(eval_file, "w", encoding="utf-8") as f:
        for e in eval_records:
            f.write(json.dumps(e.__dict__) + "\n")

    print(f"EDU-1 Generation & Evaluation complete: {len(gen_records)} records exported to {out_dir}")
    return out_dir


if __name__ == "__main__":
    run_edu1_generation_and_eval()
