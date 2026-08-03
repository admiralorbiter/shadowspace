"""E004 Ollama Multi-Worker Parallel Inference Runner.

Optimized for system hardware using ThreadPoolExecutor parallel workers.
Executes Log Probability Estimation (LPE) and Monte Carlo Estimation (MCE)
over specified item manifests using Ollama's OpenAI-compatible chat completion API.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error

API_URL = "http://localhost:11434/v1/chat/completions"
RAW_RESPONSES_DIR = Path("research/chaosnli/artifacts/E004/raw_responses")
MANIFEST_DIR = Path("research/chaosnli/artifacts/E004/manifests")

PROMPT_TEMPLATE = """Assume the premise is true.

Determine the relationship of the hypothesis to the premise.

Premise:
{premise}

Hypothesis:
{hypothesis}

Labels:
{symbol_1} = Entailment: the hypothesis must be true if the premise is true.
{symbol_2} = Neutral: the premise does not determine whether the hypothesis is true or false.
{symbol_3} = Contradiction: the hypothesis must be false if the premise is true.

Respond with exactly one symbol:
{symbol_1}, {symbol_2}, or {symbol_3}"""

LABEL_SETS = {
    "ABC": ["A", "B", "C"],
    "123": ["1", "2", "3"],
    "XYZ": ["X", "Y", "Z"],
}

S3_PERMUTATIONS = [
    (0, 1, 2),  # perm 0: E->s1, N->s2, C->s3
    (0, 2, 1),  # perm 1: E->s1, N->s3, C->s2
    (1, 0, 2),  # perm 2: E->s2, N->s1, C->s3
    (1, 2, 0),  # perm 3: E->s2, N->s3, C->s1
    (2, 0, 1),  # perm 4: E->s3, N->s1, C->s2
    (2, 1, 0),  # perm 5: E->s3, N->s2, C->s1
]

def make_request_id(
    model_tag: str,
    object_id: str,
    prompt_version: str,
    perm_idx: int,
    mode: str,
    temperature: float,
    seed: int,
    replicate: int,
    symbol_set_name: str = "ABC"
) -> str:
    raw = f"E004|{model_tag}|{object_id}|{prompt_version}|{symbol_set_name}|{perm_idx}|{mode}|{temperature:.4f}|{seed}|{replicate}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def get_existing_request_ids(file_path: Path) -> set[str]:
    if not file_path.exists():
        return set()
    existing = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if "request_id" in rec:
                    existing.add(rec["request_id"])
            except Exception:
                pass
    return existing

def prewarm_model(model_tag: str = "gemma3:12b") -> None:
    print(f"Pre-warming model '{model_tag}'...", flush=True)
    payload = {
        "model": model_tag,
        "messages": [{"role": "user", "content": "Respond with 1."}],
        "max_tokens": 1,
        "temperature": 1.0,
    }
    req = urllib.request.Request(API_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            resp.read()
        print("  Pre-warming complete.", flush=True)
    except Exception as e:
        print(f"  Warning: Pre-warming failed: {e}", flush=True)

def send_chat_completion(payload: dict) -> dict:
    req = urllib.request.Request(API_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def process_single_lpe_task(task_args: Tuple) -> Optional[dict]:
    model_tag, item, perm_idx, perm, symbols, symbol_set_name, existing_ids = task_args
    object_id = item["object_id"]
    premise = item["premise"]
    hypothesis = item["hypothesis"]

    sym_e, sym_n, sym_c = symbols[perm[0]], symbols[perm[1]], symbols[perm[2]]

    prompt = PROMPT_TEMPLATE.format(
        premise=premise,
        hypothesis=hypothesis,
        symbol_1=symbols[0],
        symbol_2=symbols[1],
        symbol_3=symbols[2],
    )

    req_id = make_request_id(model_tag, object_id, "v1", perm_idx, "lpe", 1.0, 42, 0, symbol_set_name)
    if req_id in existing_ids:
        return None

    payload = {
        "model": model_tag,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 42,
        "logprobs": True,
        "top_logprobs": 20,
    }

    try:
        resp_json = send_chat_completion(payload)
        return {
            "request_id": req_id,
            "object_id": object_id,
            "permutation_index": perm_idx,
            "symbol_mapping": {"entailment": sym_e, "neutral": sym_n, "contradiction": sym_c},
            "mode": "lpe",
            "temperature": 1.0,
            "seed": 42,
            "replicate": 0,
            "response": resp_json,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
    except Exception as e:
        print(f"  Error on item {object_id} perm {perm_idx}: {e}", flush=True)
        return None

def run_lpe(manifest_path: Path, output_path: Path, max_workers: int = 4, model_tag: str = "gemma3:12b", symbol_set_name: str = "ABC") -> None:
    existing_ids = get_existing_request_ids(output_path)
    symbols = LABEL_SETS[symbol_set_name]

    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_list = []
    for item in items:
        for perm_idx, perm in enumerate(S3_PERMUTATIONS):
            task_list.append((model_tag, item, perm_idx, perm, symbols, symbol_set_name, existing_ids))

    total_tasks = len(task_list)
    print(f"Starting Multi-Worker LPE run (workers={max_workers}) over {len(items)} items x 6 perms ({total_tasks} tasks)...", flush=True)
    print(f"  Existing completed requests: {len(existing_ids)}", flush=True)

    completed = 0
    t0 = time.time()

    with open(output_path, "a", encoding="utf-8") as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(process_single_lpe_task, task): task for task in task_list}
            for future in concurrent.futures.as_completed(future_to_task):
                res = future.result()
                if res is not None:
                    out_f.write(json.dumps(res) + "\n")
                    out_f.flush()
                    completed += 1
                    if completed % 20 == 0 or completed == total_tasks:
                        elapsed = time.time() - t0
                        rate = completed / max(0.1, elapsed)
                        print(f"  Completed {completed}/{total_tasks} LPE requests ({rate:.2f} req/sec | Elapsed: {elapsed:.1f}s)", flush=True)

def main():
    parser = argparse.ArgumentParser(description="E004 Optimized Multi-Worker Ollama Runner")
    parser.add_argument("--mode", choices=["preflight", "lpe", "mce"], required=True)
    parser.add_argument("--subset", choices=["preflight", "pilot"], default="preflight")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--symbol-set", choices=["ABC", "123", "XYZ"], default="ABC")

    args = parser.parse_args()
    prewarm_model(args.model)

    if args.mode in ["preflight", "lpe"]:
        manifest = MANIFEST_DIR / f"{args.subset}_60.jsonl" if args.subset == "preflight" else MANIFEST_DIR / f"{args.subset}_600.jsonl"
        out_lpe = RAW_RESPONSES_DIR / f"{args.subset}_lpe_responses.jsonl"
        run_lpe(manifest, out_lpe, max_workers=args.workers, model_tag=args.model, symbol_set_name=args.symbol_set)

if __name__ == "__main__":
    main()
