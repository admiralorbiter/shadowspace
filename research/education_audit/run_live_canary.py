"""Phase EDU-2a Live Canary Execution Runner.

Executes 5-letter preflight verification, followed by 60-letter randomized live generation,
screening evaluation, and blinded rating packet export.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List

import numpy as np

from research.education_audit.adapters.ollama import OllamaEducationAdapter
from research.education_audit.case_builder import build_synthetic_audit_cases
from research.education_audit.evaluation.rubric import evaluate_generation
from research.education_audit.prompt_registry import PROMPT_TEMPLATES
from research.education_audit.reporting.rating_packet import generate_blinded_rating_packet
from research.education_audit.schemas import GenerationRecord
from research.education_audit.variant_builder import build_variants_for_case


def run_edu2a_canary(
    out_dir: str = "results/education_audit/edu_2a",
    model_name: str = "gemma:12b",
    use_mock_fallback_for_offline: bool = True,
    run_order_seed: int = 42,
) -> str:
    """Executes EDU-2a Canary: Preflight + 60-Letter Randomized Execution + Blinding Packet Export."""
    os.makedirs(out_dir, exist_ok=True)

    adapter = OllamaEducationAdapter(
        model_name=model_name,
        use_mock_fallback=use_mock_fallback_for_offline,
    )
    is_live = adapter.ping_and_inspect()
    print(f"Ollama Live Adapter Status for '{model_name}': loaded={is_live}, digest={adapter.model_digest}, version={adapter.ollama_version}")

    all_cases = {c.case_id: c for c in build_synthetic_audit_cases()}
    # Select Canary Profiles: tech_qual_001 and hum_excep_002
    canary_cases = [all_cases["tech_qual_001"], all_cases["hum_excep_002"]]

    # 1. Five-Letter Preflight Check (1 case x 1 prompt x 5 conditions x 1 seed)
    print("\n--- Running 5-Letter Preflight Check ---")
    preflight_case = canary_cases[0]
    preflight_variants = build_variants_for_case(preflight_case)
    preflight_records = []

    for var in preflight_variants:
        g = adapter.generate(preflight_case, var, "minimal_prompt", PROMPT_TEMPLATES["minimal_prompt"], repeat_index=0, seed=101)
        assert len(g.output_text) > 50, f"Preflight generation too short or empty for {var.condition}!"
        assert g.parameters.get("done_reason") in ["stop", "length", "mock"], f"Preflight done_reason error for {var.condition}!"
        preflight_records.append(g)

    print(f"Preflight Check Passed! 5/5 letters completed cleanly (Mean len: {int(np.mean([len(g.output_text) for g in preflight_records]))} chars).")

    # 2. Build 60-Run Task Combinations
    task_grid = []
    variants_map = {}

    for c in canary_cases:
        vars_for_c = build_variants_for_case(c)
        for v in vars_for_c:
            variants_map[v.variant_id] = v
            for p_id in ["minimal_prompt", "structured_prompt"]:
                for r_idx, seed_val in enumerate([101, 202, 303]):
                    task_grid.append({
                        "case": c,
                        "variant": v,
                        "prompt_id": p_id,
                        "prompt_template": PROMPT_TEMPLATES[p_id],
                        "repeat_index": r_idx,
                        "seed": seed_val,
                    })

    assert len(task_grid) == 60, f"Task grid size error: expected 60, got {len(task_grid)}"

    # Randomize execution order with stable run-order seed
    rng = random.Random(run_order_seed)
    grid_shuffled = list(task_grid)
    rng.shuffle(grid_shuffled)

    print(f"\n--- Running Frozen 60-Letter Canary Execution ({len(grid_shuffled)} runs) ---")
    canary_generations: List[GenerationRecord] = []
    canary_evaluations = []

    for idx, task in enumerate(grid_shuffled, start=1):
        g = adapter.generate(
            task["case"],
            task["variant"],
            task["prompt_id"],
            task["prompt_template"],
            repeat_index=task["repeat_index"],
            seed=task["seed"],
        )
        e = evaluate_generation(task["case"], g)
        canary_generations.append(g)
        canary_evaluations.append(e)

        if idx % 15 == 0 or idx == len(grid_shuffled):
            print(f"Completed {idx}/60 generations...")

    # 3. Export Generations & Screening Evaluations
    gen_path = os.path.join(out_dir, "generations.jsonl")
    with open(gen_path, "w", encoding="utf-8") as f:
        for g in canary_generations:
            f.write(json.dumps(g.__dict__) + "\n")

    eval_path = os.path.join(out_dir, "screening_evaluations.jsonl")
    with open(eval_path, "w", encoding="utf-8") as f:
        for e in canary_evaluations:
            f.write(json.dumps(e.__dict__) + "\n")

    # 4. Generate Blinded Rating Packet
    generate_blinded_rating_packet(canary_generations, variants_map, out_dir=out_dir)

    # 5. Export Sub-Manifests
    with open(os.path.join(out_dir, "case_manifest.json"), "w", encoding="utf-8") as f:
        json.dump([c.__dict__ for c in canary_cases], f, indent=2)

    with open(os.path.join(out_dir, "variant_manifest.json"), "w", encoding="utf-8") as f:
        json.dump([v.__dict__ for v in variants_map.values()], f, indent=2)

    with open(os.path.join(out_dir, "prompt_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(PROMPT_TEMPLATES, f, indent=2)

    with open(os.path.join(out_dir, "generation_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "total_generations": len(canary_generations),
            "model_name": model_name,
            "model_digest": adapter.model_digest,
            "is_live_adapter": is_live,
            "seeds_used": [101, 202, 303],
            "run_order_seed": run_order_seed,
        }, f, indent=2)

    print(f"\nEDU-2a Canary Execution Complete! All artifacts exported to {out_dir}")
    return out_dir


if __name__ == "__main__":
    run_edu2a_canary()
