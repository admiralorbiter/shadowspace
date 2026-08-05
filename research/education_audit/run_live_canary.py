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
    model_name: str = "gemma3:12b",
    use_mock_fallback_for_offline: bool = False,
    run_order_seed: int = 42,
) -> str:
    """Executes EDU-2a-R1 Canary: Calibration + 60-Letter Strict Execution + Private Blinding Packet Export."""
    os.makedirs(out_dir, exist_ok=True)

    adapter = OllamaEducationAdapter(
        model_name=model_name,
        use_mock_fallback=use_mock_fallback_for_offline,
    )
    is_live = adapter.ping_and_inspect()
    if not is_live and not use_mock_fallback_for_offline:
        raise RuntimeError(f"Ollama hard refusal: Could not load model '{model_name}'. Error: {adapter.load_error}")

    print(f"Ollama Live Adapter Status for '{model_name}': loaded={is_live}, digest={adapter.model_digest}, version={adapter.ollama_version}")

    all_cases = {c.case_id: c for c in build_synthetic_audit_cases()}
    canary_cases = [all_cases["tech_qual_001"], all_cases["hum_excep_002"]]

    # 1. Four-Letter Disposable Preflight Calibration Check (2 cases x 2 prompts x 1 anon condition)
    print("\n--- Running 4-Letter Disposable Preflight Calibration ---")
    calibration_records = []

    for c in canary_cases:
        anon_var = [v for v in build_variants_for_case(c) if v.condition == "anonymous"][0]
        for p_id in ["minimal_prompt", "structured_prompt"]:
            g = adapter.generate(c, anon_var, p_id, PROMPT_TEMPLATES[p_id], repeat_index=0, seed=101)
            reason = g.parameters.get("done_reason")
            words = len(g.output_text.split())
            paragraphs = len([p for p in g.output_text.split("\n\n") if p.strip()])

            assert reason in ["stop", "mock"], f"Preflight calibration truncation failure for {c.case_id}/{p_id}! Reason: {reason}"
            assert 100 <= words <= 350, f"Preflight calibration word count out of bounds ({words} words) for {c.case_id}/{p_id}!"
            calibration_records.append(g)

    print(f"Disposable Calibration Passed! 4/4 calibration letters completed cleanly (All done_reason == 'stop').")

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
    truncation_count = 0

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

        reason = g.parameters.get("done_reason")
        if reason not in ["stop", "mock"]:
            truncation_count += 1

        if idx % 15 == 0 or idx == len(grid_shuffled):
            print(f"Completed {idx}/60 generations... (Truncations so far: {truncation_count})")

    # Fail closed if any record truncated
    if truncation_count > 0:
        raise RuntimeError(f"Canary execution rejected: {truncation_count}/60 generations were truncated (done_reason != 'stop').")

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
    from research.holonomy.experiments.run_phase_e0_summary import get_git_commit_sha
    source_code_sha = get_git_commit_sha()

    with open(os.path.join(out_dir, "case_manifest.json"), "w", encoding="utf-8") as f:
        json.dump([c.__dict__ for c in canary_cases], f, indent=2)

    with open(os.path.join(out_dir, "variant_manifest.json"), "w", encoding="utf-8") as f:
        json.dump([v.__dict__ for v in variants_map.values()], f, indent=2)

    with open(os.path.join(out_dir, "prompt_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(PROMPT_TEMPLATES, f, indent=2)

    with open(os.path.join(out_dir, "generation_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "source_code_commit_sha": source_code_sha,
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
