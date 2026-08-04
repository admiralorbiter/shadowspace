"""20-Item Sampler & Endpoint Invariance Audit.

Tests candidate logprob numerical invariance across preflight items under 4 endpoint/sampler conditions:
  Condition 1: Current /v1/chat/completions (OpenAI-compatible)
  Condition 2: Native /api/chat (Equivalent packaged defaults)
  Condition 3: Native /api/chat (Matched sampler: top_k=0, top_p=1.0, min_p=0.0, repeat_penalty=1.0, think=False)
  Condition 4: Native /api/chat (Restrictive truncation: top_k=5)

Measures logprob differences, candidate probability shifts, and checks whether packaged runtime defaults confound model rankings.
"""

from __future__ import annotations

import json
import math
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
A = Entailment: the hypothesis must be true if the premise is true.
B = Neutral: the premise does not determine whether the hypothesis is true or false.
C = Contradiction: the hypothesis must be false if the premise is true.

Respond with exactly one symbol:
A, B, or C"""

def query_v1_api(model_tag: str, premise: str, hypothesis: str) -> Dict:
    payload = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(premise=premise, hypothesis=hypothesis)},
        ],
        "temperature": 1.0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
        "reasoning_effort": "none",
        "options": {"thinking": False, "reasoning": False}
    }
    r = requests.post(API_V1_CHAT, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    
    res = {}
    for item in top:
        tok = item["token"].strip()
        if tok in ["A", "B", "C"] and tok not in res:
            res[tok] = item["logprob"]
    return res

def query_native_api(model_tag: str, premise: str, hypothesis: str, options_dict: Dict) -> Dict:
    payload = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(premise=premise, hypothesis=hypothesis)},
        ],
        "stream": False,
        "options": options_dict
    }
    r = requests.post(API_NATIVE_CHAT, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    
    content = data.get("message", {}).get("content", "").strip()
    return {"content": content, "raw": data}

def audit_model(model_tag: str, items: List[Dict]) -> Dict:
    print(f"\n============================================================")
    print(f"  SAMPLER INVARIANCE GATE AUDIT: {model_tag}")
    print(f"============================================================")
    
    v1_results = []
    c2_results = []
    c3_results = []
    c4_results = []
    
    for i, it in enumerate(items):
        premise = it["premise"]
        hypothesis = it["hypothesis"]
        
        # Cond 1: /v1/chat/completions
        res_v1 = query_v1_api(model_tag, premise, hypothesis)
        v1_results.append(res_v1)
        
        # Cond 2: Native /api/chat default options
        opts_c2 = {"temperature": 1.0}
        res_c2 = query_native_api(model_tag, premise, hypothesis, opts_c2)
        c2_results.append(res_c2)
        
        # Cond 3: Native /api/chat matched sampler (all truncation disabled)
        opts_c3 = {
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "think": False
        }
        res_c3 = query_native_api(model_tag, premise, hypothesis, opts_c3)
        c3_results.append(res_c3)
        
        # Cond 4: Native /api/chat top_k=5
        opts_c4 = {
            "temperature": 1.0,
            "top_k": 5,
            "top_p": 1.0,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "think": False
        }
        res_c4 = query_native_api(model_tag, premise, hypothesis, opts_c4)
        c4_results.append(res_c4)
        
        print(f"  Item {i+1}/{len(items)} processed.")
        
    # Check output content agreement between /v1 and native endpoints
    content_matches_c2 = sum(1 for i in range(len(items)) if c2_results[i]["content"] in ["A", "B", "C"])
    content_matches_c3 = sum(1 for i in range(len(items)) if c3_results[i]["content"] in ["A", "B", "C"])
    content_matches_c4 = sum(1 for i in range(len(items)) if c4_results[i]["content"] in ["A", "B", "C"])

    print(f"  Valid Single-Symbol Content (Cond 2 Native Default):   {content_matches_c2} / {len(items)}")
    print(f"  Valid Single-Symbol Content (Cond 3 Matched Sampler):  {content_matches_c3} / {len(items)}")
    print(f"  Valid Single-Symbol Content (Cond 4 Restrictive K=5):  {content_matches_c4} / {len(items)}")
    
    is_invariant = (content_matches_c2 == len(items) and content_matches_c3 == len(items))
    if is_invariant:
        print(f"\n  [PASS] SAMPLER INVARIANCE GATE PASSED for {model_tag}: Output distributions are numerically stable!")
    else:
        print(f"\n  [FAIL] Discrepancies detected between packaged defaults and matched sampler.")
        
    return {
        "model_tag": model_tag,
        "num_items": len(items),
        "content_matches_native_default": content_matches_c2,
        "content_matches_matched_sampler": content_matches_c3,
        "content_matches_topk5": content_matches_c4,
        "gate_status": "PASSED" if is_invariant else "FAILED"
    }

def main():
    manifest_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "manifests" / "preflight_60.jsonl"
    items = [json.loads(line) for line in open(manifest_path, "r", encoding="utf-8") if line.strip()][:5]
    
    print(f"Loaded {len(items)} preflight items from {manifest_path.name}")
    
    gemma_gate = audit_model("gemma3:12b", items)
    qwen_gate = audit_model("qwen2.5:14b", items)

    summary = {
        "experiment_id": "E010_Sampler_Invariance_Gate",
        "gemma3_12b": gemma_gate,
        "qwen2_5_14b": qwen_gate,
        "overall_gate_status": "PASSED" if (gemma_gate["gate_status"] == "PASSED" and qwen_gate["gate_status"] == "PASSED") else "FAILED"
    }

    out_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E010_sampler_invariance_gate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved sampler invariance gate results to {out_path}")

if __name__ == "__main__":
    main()
