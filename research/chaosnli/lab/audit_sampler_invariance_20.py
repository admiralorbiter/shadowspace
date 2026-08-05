"""Fast & Optimized Distributional Sampler & Endpoint Invariance Gate Audit (Unbuffered).

Audits candidate logprob numerical distribution invariance across 5 items x 6 permutations = 30 prompts per model.

Evaluates 3 core conditions:
  Condition 1: OpenAI-compatible /v1/chat/completions (current runner)
  Condition 2: Native /api/chat (Packaged defaults)
  Condition 3: Native /api/chat (Matched sampler: top_k=0, top_p=1.0, min_p=0.0, repeat_penalty=1.0, think=False)

Reports for all condition pairs (1 vs 2, 2 vs 3):
  - max |Delta logprob| for candidate tokens A, B, C
  - mean and max JSD between normalized candidate distributions
  - max ||p_a - p_b||_1 (L1 probability distance)
  - sampled-symbol agreement rate
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]

API_BASE = "http://localhost:11434"
API_V1_CHAT = f"{API_BASE}/v1/chat/completions"
API_NATIVE_CHAT = f"{API_BASE}/api/chat"

SYSTEM_PROMPT = """Assume the premise is true.

Determine the relationship of the hypothesis to the premise."""

USER_PROMPT_TEMPLATE = """Premise:
{premise}

Hypothesis:
{hypothesis}

Labels:
{s1} = {l1}: the hypothesis must be true if the premise is true.
{s2} = {l2}: the premise does not determine whether the hypothesis is true or false.
{s3} = {l3}: the hypothesis must be false if the premise is true.

Respond with exactly one symbol:
{s1}, {s2}, or {s3}"""

LABEL_SETS = {"ABC": ["A", "B", "C"]}
NLI_LABELS = ["Entailment", "Neutral", "Contradiction"]
S3_PERMUTATIONS = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]

def query_v1_api(model_tag: str, premise: str, hypothesis: str, perm_tuple: Tuple[int, int, int]) -> Dict:
    symbols = LABEL_SETS["ABC"]
    user_prompt = USER_PROMPT_TEMPLATE.format(
        premise=premise,
        hypothesis=hypothesis,
        s1=symbols[perm_tuple[0]], l1=NLI_LABELS[perm_tuple[0]],
        s2=symbols[perm_tuple[1]], l2=NLI_LABELS[perm_tuple[1]],
        s3=symbols[perm_tuple[2]], l3=NLI_LABELS[perm_tuple[2]],
    )
    payload = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 1.0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
        "reasoning_effort": "none",
    }
    r = requests.post(API_V1_CHAT, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    
    top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    sampled = data["choices"][0]["message"]["content"].strip()
    
    res = {}
    for item in top:
        tok = item["token"].strip()
        if tok in symbols and tok not in res:
            res[tok] = item["logprob"]
            
    return {"sampled": sampled, "logprobs": res}

def query_native_api(model_tag: str, premise: str, hypothesis: str, perm_tuple: Tuple[int, int, int], options_dict: Dict) -> Dict:
    symbols = LABEL_SETS["ABC"]
    user_prompt = USER_PROMPT_TEMPLATE.format(
        premise=premise,
        hypothesis=hypothesis,
        s1=symbols[perm_tuple[0]], l1=NLI_LABELS[perm_tuple[0]],
        s2=symbols[perm_tuple[1]], l2=NLI_LABELS[perm_tuple[1]],
        s3=symbols[perm_tuple[2]], l3=NLI_LABELS[perm_tuple[2]],
    )
    payload = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,
        "logprobs": True,
        "top_logprobs": 20,
        "options": {
            **options_dict,
            "num_predict": 1,
            "seed": 20260803,
        }
    }
    r = requests.post(API_NATIVE_CHAT, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()

    assert "logprobs" in data, f"Native API response missing 'logprobs' field for {model_tag}!"
    raw_logprobs = data.get("logprobs", [])
    assert len(raw_logprobs) > 0, f"Native API returned empty 'logprobs' list for {model_tag}!"

    content = data.get("message", {}).get("content", "").strip()
    
    res = {}
    first = raw_logprobs[0]
    top = first.get("top_logprobs", []) if isinstance(first, dict) else []
    for item in top:
        tok = item.get("token", "").strip() if isinstance(item, dict) else ""
        if tok in symbols and tok not in res:
            res[tok] = item.get("logprob", -40.0)
                
    return {"sampled": content, "logprobs": res}

def compute_prob_dist(lp_dict: Dict) -> np.ndarray:
    symbols = ["A", "B", "C"]
    lps = [lp_dict.get(s, -40.0) for s in symbols]
    max_lp = max(lps)
    unnorm = [math.exp(lp - max_lp) for lp in lps]
    denom = sum(unnorm)
    return np.array([u / denom for u in unnorm], dtype=np.float64)

def jsd_between(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm)

def compare_conditions(runs_a: List[Dict], runs_b: List[Dict], pair_name: str) -> Dict:
    total_prompts = len(runs_a)
    max_delta_logp = 0.0
    jsds = []
    l1_dists = []
    symbol_agreements = 0

    for idx in range(total_prompts):
        pa = compute_prob_dist(runs_a[idx]["logprobs"])
        pb = compute_prob_dist(runs_b[idx]["logprobs"])

        for s in ["A", "B", "C"]:
            lpa = runs_a[idx]["logprobs"].get(s, -40.0)
            lpb = runs_b[idx]["logprobs"].get(s, -40.0)
            diff = abs(lpa - lpb)
            if diff > max_delta_logp and lpa > -35.0 and lpb > -35.0:
                max_delta_logp = diff

        jsd_val = jsd_between(pa, pb)
        jsds.append(jsd_val)
        l1_dists.append(float(np.sum(np.abs(pa - pb))))

        if runs_a[idx]["sampled"] == runs_b[idx]["sampled"]:
            symbol_agreements += 1

    mean_jsd = float(np.mean(jsds))
    max_jsd = float(np.max(jsds))
    max_l1 = float(np.max(l1_dists))
    sampled_agree_pct = float(symbol_agreements / total_prompts * 100.0)

    print(f"    Pair [{pair_name}]:", flush=True)
    print(f"      Sampled Symbol Agreement:     {sampled_agree_pct:.1f}% ({symbol_agreements}/{total_prompts})", flush=True)
    print(f"      Max |Delta logprob|:          {max_delta_logp:.6f}", flush=True)
    print(f"      Distributional JSD (Mean/Max): {mean_jsd:.6f} / {max_jsd:.6f}", flush=True)
    print(f"      Max L1 Distance:               {max_l1:.6f}", flush=True)

    return {
        "pair_name": pair_name,
        "sampled_symbol_agreement_pct": sampled_agree_pct,
        "max_delta_logprob": max_delta_logp,
        "mean_jsd": mean_jsd,
        "max_jsd": max_jsd,
        "max_l1_dist": max_l1
    }

def audit_model(model_tag: str, items: List[Dict]) -> Dict:
    print(f"\n============================================================", flush=True)
    print(f"  SAMPLER & ENDPOINT INVARIANCE AUDIT: {model_tag}", flush=True)
    print(f"============================================================", flush=True)
    
    c1_runs = []
    c2_runs = []
    c3_runs = []
    
    total_prompts = len(items) * 6
    prompt_count = 0
    
    for i, it in enumerate(items):
        premise = it["premise"]
        hypothesis = it["hypothesis"]
        for p_idx, perm in enumerate(S3_PERMUTATIONS):
            prompt_count += 1
            
            # Cond 1: /v1/chat/completions
            r1 = query_v1_api(model_tag, premise, hypothesis, perm)
            c1_runs.append(r1)
            
            # Cond 2: Native /api/chat packaged defaults
            opts_c2 = {"temperature": 1.0}
            r2 = query_native_api(model_tag, premise, hypothesis, perm, opts_c2)
            c2_runs.append(r2)
            
            # Cond 3: Native /api/chat matched sampler (top_k=0, top_p=1.0, min_p=0.0, repeat_penalty=1.0)
            opts_c3 = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0}
            r3 = query_native_api(model_tag, premise, hypothesis, perm, opts_c3)
            c3_runs.append(r3)
            
            print(f"  [{model_tag}] Processed prompt {prompt_count}/{total_prompts}...", flush=True)

    print(f"  Completed all {total_prompts} prompts across 3 conditions.\n", flush=True)

    # Evaluate Pairs
    pair_1_2 = compare_conditions(c1_runs, c2_runs, "Cond 1 (/v1) vs Cond 2 (Native Packaged)")
    pair_2_3 = compare_conditions(c2_runs, c3_runs, "Cond 2 (Native Packaged) vs Cond 3 (Native Matched)")

    is_passed = (pair_2_3["mean_jsd"] < 1e-4 and pair_2_3["max_l1_dist"] < 1e-3)

    return {
        "model_tag": model_tag,
        "num_prompts": total_prompts,
        "pair_1_2_v1_vs_native_packaged": pair_1_2,
        "pair_2_3_native_packaged_vs_matched": pair_2_3,
        "gate_status": "PASSED" if is_passed else "MATCHED_SAMPLER_CONTRACT_RECOMMENDED"
    }

def main():
    manifest_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "manifests" / "preflight_60.jsonl"
    items = [json.loads(line) for line in open(manifest_path, "r", encoding="utf-8") if line.strip()][:5]
    
    print(f"Loaded {len(items)} preflight items (30 prompts per model) from {manifest_path.name}", flush=True)

    gemma_gate = audit_model("gemma3:12b", items)
    qwen_gate = audit_model("qwen2.5:14b", items)

    summary = {
        "experiment_id": "E010_Sampler_Invariance_Gate_30Prompts",
        "gemma3_12b": gemma_gate,
        "qwen2_5_14b": qwen_gate,
        "overall_gate_status": "PASSED" if (gemma_gate["gate_status"] == "PASSED" and qwen_gate["gate_status"] == "PASSED") else "MATCHED_SAMPLER_CONTRACT_RECOMMENDED"
    }

    out_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E010_sampler_invariance_gate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved sampler invariance gate results to {out_path}", flush=True)

if __name__ == "__main__":
    main()
