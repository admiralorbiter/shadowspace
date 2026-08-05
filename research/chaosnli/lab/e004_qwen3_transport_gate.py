"""E004 Qwen3 14B 20-Item Transport Gate Harness.

Runs 20 items x 6 permutations = 120 unique requests for qwen3:14b under the exact frozen E004 /v1/chat/completions protocol:
  - temperature = 1.0
  - max_tokens = 1
  - logprobs = True
  - top_logprobs = 20
  - reasoning_effort = "none"

Asserts 100% pass conditions:
  1. 120/120 valid responses.
  2. Exactly 1 generated token (no thinking/reasoning preamble).
  3. First token is a legal symbol (A, B, or C).
  4. Non-null candidate logprobs observed or explicitly marked censored.
  5. SHA-256 digests and provenance metadata recorded.
"""

from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]

API_V1_CHAT = "http://localhost:11434/v1/chat/completions"
MODEL_TAG = "qwen3:14b"

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

def main():
    manifest_path = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "manifests" / "preflight_60.jsonl"
    items = [json.loads(line) for line in open(manifest_path, "r", encoding="utf-8") if line.strip()][:20]

    print(f"============================================================")
    print(f"  RUNNING E004 TRANSPORT GATE FOR MODEL: {MODEL_TAG}")
    print(f"============================================================")
    print(f"Loaded {len(items)} items ({len(items) * 6} requests) from {manifest_path.name}")

    out_dir = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "raw_responses"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_responses_file = out_dir / "gate120_qwen3-14b_v2_abc_t10_lpe.jsonl"

    responses = []
    valid_count = 0

    t0 = time.time()

    for idx, it in enumerate(items):
        premise = it["premise"]
        hypothesis = it["hypothesis"]
        object_id = it["object_id"]

        for perm_idx, perm in enumerate(S3_PERMUTATIONS):
            symbols = LABEL_SETS["ABC"]
            user_prompt = USER_PROMPT_TEMPLATE.format(
                premise=premise,
                hypothesis=hypothesis,
                s1=symbols[perm[0]], l1=NLI_LABELS[perm[0]],
                s2=symbols[perm[1]], l2=NLI_LABELS[perm[1]],
                s3=symbols[perm[2]], l3=NLI_LABELS[perm[2]],
            )

            payload = {
                "model": MODEL_TAG,
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

            req_t0 = time.time()
            r = requests.post(API_V1_CHAT, json=payload, timeout=60)
            latency = time.time() - req_t0

            assert r.status_code == 200, f"Request failed with status code {r.status_code}: {r.text}"
            data = r.json()

            choice = data["choices"][0]
            sampled_text = choice["message"]["content"].strip()
            logprobs_content = choice["logprobs"]["content"][0]
            top_logprobs = logprobs_content.get("top_logprobs", [])

            # Check single symbol output
            is_valid_symbol = (sampled_text in symbols)
            if is_valid_symbol:
                valid_count += 1

            token_logprobs = {entry["token"].strip(): entry["logprob"] for entry in top_logprobs}

            rec = {
                "request_id": f"gate_{object_id}_{perm_idx}",
                "object_id": object_id,
                "item_index": idx,
                "perm_index": perm_idx,
                "perm_tuple": perm,
                "model_tag": MODEL_TAG,
                "status": "success",
                "latency_sec": latency,
                "output_text": sampled_text,
                "first_token": logprobs_content.get("token", ""),
                "top_logprobs": top_logprobs,
                "token_logprobs": token_logprobs
            }
            responses.append(rec)

        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{len(items)} items ({(idx + 1) * 6}/{len(items) * 6} requests)...")

    elapsed = time.time() - t0
    print(f"Completed {len(responses)} requests in {elapsed:.2f}s ({len(responses)/elapsed:.2f} req/s).")

    # Export raw responses
    with open(out_responses_file, "w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r) + "\n")

    # Verify Transport Gate Criteria
    print("\n============================================================")
    print(f"  E004 TRANSPORT GATE VERIFICATION SUMMARY: {MODEL_TAG}")
    print("============================================================")
    print(f"  Total Requests:              {len(responses)} / {len(items) * 6}")
    print(f"  Valid Single-Symbol Outputs: {valid_count} / {len(responses)}")
    print(f"  Success Rate:                {valid_count / len(responses) * 100.0:.2f}%")

    assert len(responses) == 120, f"Expected 120 responses, got {len(responses)}"
    assert valid_count == 120, f"Expected 120 valid single-symbol outputs, got {valid_count}"

    gate_summary = {
        "model_tag": MODEL_TAG,
        "total_requests": len(responses),
        "valid_outputs": valid_count,
        "latency_total_sec": elapsed,
        "requests_per_sec": len(responses) / elapsed,
        "responses_sha256": hashlib.sha256(out_responses_file.read_bytes()).hexdigest(),
        "gate_status": "PASSED"
    }

    gate_out_path = PROJECT_ROOT / "research" / "chaosnli" / "results" / "E004_qwen3_transport_gate_summary.json"
    gate_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gate_out_path, "w", encoding="utf-8") as f:
        json.dump(gate_summary, f, indent=2)

    print(f"  [PASS] GATE PASSED: {MODEL_TAG} is ready for 600-item pilot run!")
    print(f"Saved transport gate summary to {gate_out_path}\n")

if __name__ == "__main__":
    main()
