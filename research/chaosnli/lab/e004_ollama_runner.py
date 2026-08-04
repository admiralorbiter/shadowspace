"""E004 Ollama High-Throughput Parallel Inference Runner.

Optimized with persistent HTTP connection pooling (requests.Session) and system-prompt
KV-cache separation for maximum throughput on local GPU/Ollama server.
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

import requests
from requests.adapters import HTTPAdapter

API_URL = "http://localhost:11434/v1/chat/completions"
RAW_RESPONSES_DIR = Path("research/chaosnli/artifacts/E004/raw_responses")
MANIFEST_DIR = Path("research/chaosnli/artifacts/E004/manifests")

SYSTEM_PROMPT = """Assume the premise is true.

Determine the relationship of the hypothesis to the premise."""

USER_PROMPT_TEMPLATE = """Premise:
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

# Thread-local HTTP session storage for connection pooling
_THREAD_LOCAL_SESSIONS: Dict[int, requests.Session] = {}

def get_session() -> requests.Session:
    import threading
    tid = threading.get_ident()
    if tid not in _THREAD_LOCAL_SESSIONS:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _THREAD_LOCAL_SESSIONS[tid] = session
    return _THREAD_LOCAL_SESSIONS[tid]

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
            except json.JSONDecodeError:
                pass
    return existing

def query_ollama(
    messages: List[Dict[str, str]],
    model_tag: str,
    temperature: float = 0.0,
    max_tokens: int = 1,
    logprobs: bool = True,
    top_logprobs: int = 20,
    seed: Optional[int] = None
) -> Dict:
    session = get_session()
    payload = {
        "model": model_tag,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs if logprobs else None,
    }
    if seed is not None:
        payload["seed"] = seed

    resp = session.post(API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

def process_single_lpe_task(task_tuple) -> Optional[Dict]:
    model_tag, item, perm_idx, perm, symbols, symbol_set_name, existing_ids = task_tuple
    object_id = item["object_id"]
    req_id = make_request_id(model_tag, object_id, "v1", perm_idx, "lpe", 0.0, 0, 0, symbol_set_name)

    if req_id in existing_ids:
        return None

    symbol_1, symbol_2, symbol_3 = symbols[perm[0]], symbols[perm[1]], symbols[perm[2]]
    user_content = USER_PROMPT_TEMPLATE.format(
        premise=item["premise"],
        hypothesis=item["hypothesis"],
        symbol_1=symbol_1,
        symbol_2=symbol_2,
        symbol_3=symbol_3,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    t0 = time.time()
    resp_json = query_ollama(messages, model_tag, temperature=0.0, max_tokens=1, logprobs=True, top_logprobs=20)
    dt = time.time() - t0

    choice = resp_json["choices"][0]
    logprobs_content = choice.get("logprobs", {}).get("content", [])

    return {
        "request_id": req_id,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "object_id": object_id,
        "row_index": item["row_index"],
        "prompt_version": "v1",
        "symbol_set": symbol_set_name,
        "perm_idx": perm_idx,
        "perm_tuple": list(perm),
        "symbol_mapping": {"E": symbol_1, "N": symbol_2, "C": symbol_3},
        "mode": "lpe",
        "temperature": 0.0,
        "max_tokens": 1,
        "response_text": choice["message"]["content"],
        "logprobs": logprobs_content,
        "latency_sec": round(dt, 4),
    }

def run_lpe(manifest_path: Path, output_path: Path, max_workers: int = 4, model_tag: str = "gemma3:12b", symbol_set_name: str = "ABC"):
    symbols = LABEL_SETS[symbol_set_name]
    existing_ids = get_existing_request_ids(output_path)

    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
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

def prewarm_model(model_tag: str):
    print(f"Pre-warming model '{model_tag}' on Ollama...", flush=True)
    session = get_session()
    payload = {
        "model": model_tag,
        "messages": [{"role": "system", "content": "Hi"}, {"role": "user", "content": "Ping"}],
        "max_tokens": 1,
    }
    resp = session.post(API_URL, json=payload, timeout=60)
    resp.raise_for_status()
    print("  Model ready.", flush=True)

def main():
    parser = argparse.ArgumentParser(description="E004 Optimized High-Throughput Ollama Runner")
    parser.add_argument("--mode", choices=["preflight", "lpe", "mce", "validate_20"], required=True)
    parser.add_argument("--subset", choices=["preflight", "pilot"], default="pilot")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--symbol-set", choices=["ABC", "123", "XYZ"], default="ABC")

    args = parser.parse_args()
    prewarm_model(args.model)

    if args.mode == "validate_20":
        print("\n--- Running 20-Item Deterministic Validation under Overnight Ollama Config ---")
        manifest = MANIFEST_DIR / "preflight_60.jsonl"
        out_val = RAW_RESPONSES_DIR / "val20_check_responses.jsonl"
        run_lpe(manifest, out_val, max_workers=args.workers, model_tag=args.model, symbol_set_name=args.symbol_set)
        print("Validation complete. Check logprobs in val20_check_responses.jsonl")

    elif args.mode in ["preflight", "lpe"]:
        manifest = MANIFEST_DIR / "preflight_60.jsonl" if args.subset == "preflight" else MANIFEST_DIR / "pilot_600.jsonl"
        out_lpe = RAW_RESPONSES_DIR / f"{args.subset}_lpe_responses.jsonl"
        run_lpe(manifest, out_lpe, max_workers=args.workers, model_tag=args.model, symbol_set_name=args.symbol_set)

if __name__ == "__main__":
    main()
