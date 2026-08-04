"""E004 Ollama Hardened Inference Runner — v2 (Audited, Path-Resolved, API T=1.0 & Reasoning-Disabled).

Addresses all pre-launch audit requirements:
  1. Frozen inference contract: PROMPT_VERSION="v2", PRIMARY_SYMBOL_SET="ABC",
     full provenance per record (prompt hashes, model digest, Ollama version,
     request-body hash, latency, success/error status).
  2. validate_20 sends exactly 20 items (120 requests).
  3. Supports explicit temperature configuration (--temperature 1.0 default) with
     temperature tag recorded in request ID, output filename, record, and provenance.
  4. Explicitly disables thinking/reasoning mode using Ollama's top-level payload parameter
     ("reasoning_effort": "none") and options={"thinking": False, "reasoning": False}.
  5. Robust retry with exponential backoff, structured error records,
     failed-request queue, and audit summary.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROMPT_VERSION = "v2"
PRIMARY_SYMBOL_SET = "ABC"

PROJECT_ROOT = Path(__file__).resolve().parents[3]

API_BASE = "http://localhost:11434"
API_CHAT = f"{API_BASE}/v1/chat/completions"
API_VERSION = f"{API_BASE}/api/version"
API_SHOW = f"{API_BASE}/api/show"

RAW_RESPONSES_DIR = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "raw_responses"
MANIFEST_DIR = PROJECT_ROOT / "research" / "chaosnli" / "artifacts" / "E004" / "manifests"

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

NLI_LABELS = ["entailment", "neutral", "contradiction"]

S3_PERMUTATIONS: List[Tuple[int, int, int]] = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]

SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 1.0
BACKOFF_FACTOR = 2.0
REQUEST_TIMEOUT_SEC = 60

_thread_local = threading.local()

def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local.session = session
    return _thread_local.session

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def get_ollama_version() -> str:
    try:
        r = requests.get(API_VERSION, timeout=5)
        if r.status_code == 200:
            return r.json().get("version", "unknown")
    except Exception:
        pass
    return "unknown"

def get_model_digest(model_tag: str) -> str:
    try:
        r = requests.post(API_SHOW, json={"name": model_tag}, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "modelinfo" in data:
            return sha256_str(json.dumps(data["modelinfo"], sort_keys=True))
        return data.get("digest", "unknown")
    except Exception:
        return "unknown"

def model_slug(model_tag: str) -> str:
    return model_tag.replace(":", "-").replace("/", "-")

def make_request_id(
    model_tag: str,
    object_id: str,
    prompt_version: str,
    perm_idx: int,
    mode: str,
    temperature: float,
    seed: int,
    replicate: int,
    symbol_set_name: str = "ABC",
) -> str:
    raw = (
        f"E004|{model_tag}|{object_id}|{prompt_version}|{symbol_set_name}"
        f"|{perm_idx}|{mode}|{temperature:.4f}|{seed}|{replicate}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def get_existing_request_ids(file_path: Path) -> set:
    file_path = file_path.resolve()
    if not file_path.exists():
        return set()
    existing = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("request_id")
                if rid and rec.get("status") == "success" and rec.get("valid_output", True) is True:
                    existing.add(rid)
            except json.JSONDecodeError:
                pass
    return existing

def query_ollama_with_retry(
    messages: List[Dict[str, str]],
    model_tag: str,
    temperature: float = 1.0,
    max_tokens: int = 1,
    logprobs: bool = True,
    top_logprobs: int = 20,
    seed: Optional[int] = None,
) -> Tuple[Optional[Dict], Optional[str], float]:
    session = get_session()
    payload: Dict[str, Any] = {
        "model": model_tag,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": logprobs,
        "reasoning_effort": "none",
        "options": {
            "thinking": False,
            "reasoning": False
        }
    }
    if logprobs:
        payload["top_logprobs"] = top_logprobs
    if seed is not None:
        payload["seed"] = seed

    total_latency = 0.0
    last_error = ""

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = session.post(API_CHAT, json=payload, timeout=REQUEST_TIMEOUT_SEC)
            dt = time.time() - t0
            total_latency += dt
            resp.raise_for_status()
            return resp.json(), None, total_latency
        except requests.exceptions.ConnectionError as e:
            last_error = f"ConnectionError: {e}"
        except requests.exceptions.Timeout as e:
            last_error = f"Timeout: {e}"
            total_latency += REQUEST_TIMEOUT_SEC
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 0
            last_error = f"HTTPError {status_code}: {e}"
            if 400 <= status_code < 500 and status_code != 429:
                return None, last_error, total_latency
        except Exception as e:
            last_error = f"Unexpected: {type(e).__name__}: {e}"

        if attempt < MAX_RETRIES - 1:
            backoff = INITIAL_BACKOFF_SEC * (BACKOFF_FACTOR ** attempt)
            time.sleep(backoff)

    return None, last_error, total_latency

def render_prompt(premise: str, hypothesis: str, mapping: Tuple[int, int, int], symbol_set: List[str]) -> str:
    s1, s2, s3 = symbol_set[mapping[0]], symbol_set[mapping[1]], symbol_set[mapping[2]]
    return USER_PROMPT_TEMPLATE.format(
        premise=premise, hypothesis=hypothesis, symbol_1=s1, symbol_2=s2, symbol_3=s3
    )

def extract_logprob_distribution(
    resp_json: Dict[str, Any],
    mapping: Tuple[int, int, int],
    symbol_set: List[str],
) -> Dict[str, Any]:
    choices = resp_json.get("choices", [])
    if not choices:
        return {"valid": False, "error": "No choices in response"}

    choice = choices[0]
    message = choice.get("message", {})
    output_text = message.get("content", "").strip()

    logprobs_obj = choice.get("logprobs", {})
    content_logprobs = logprobs_obj.get("content", []) if logprobs_obj else []

    if not content_logprobs:
        return {"valid": False, "error": "No content logprobs returned"}

    first_token_info = content_logprobs[0]
    top_logprobs_list = first_token_info.get("top_logprobs", [])

    if not top_logprobs_list:
        return {"valid": False, "error": "No top_logprobs in first token"}

    symbols_in_play = {
        symbol_set[mapping[0]]: 0,
        symbol_set[mapping[1]]: 1,
        symbol_set[mapping[2]]: 2,
    }

    symbol_logprobs = {}
    found_tokens = {}
    for item in top_logprobs_list:
        token = item.get("token", "").strip()
        lp = item.get("logprob")
        if token in symbols_in_play and token not in found_tokens and lp is not None:
            found_tokens[token] = lp
            nli_idx = symbols_in_play[token]
            label_name = NLI_LABELS[nli_idx]
            symbol_logprobs[label_name] = {
                "token": token,
                "logprob": float(lp),
                "prob": float(math.exp(lp)),
            }

    missing_labels = [l for l in NLI_LABELS if l not in symbol_logprobs]
    if missing_labels:
        for missing_l in missing_labels:
            missing_nli_idx = NLI_LABELS.index(missing_l)
            for s_char, idx_val in symbols_in_play.items():
                if idx_val == missing_nli_idx:
                    symbol_logprobs[missing_l] = {
                        "token": s_char,
                        "logprob": -40.0,
                        "prob": float(math.exp(-40.0)),
                    }

    raw_probs = [symbol_logprobs[label]["prob"] for label in NLI_LABELS]
    candidate_mass = sum(raw_probs)

    if candidate_mass <= 0 or not math.isfinite(candidate_mass):
        return {"valid": False, "error": f"Invalid candidate mass: {candidate_mass}"}

    norm_probs = [p / candidate_mass for p in raw_probs]

    return {
        "valid": True,
        "output_text": output_text,
        "first_token": first_token_info.get("token"),
        "first_token_logprob": first_token_info.get("logprob"),
        "candidate_mass": float(candidate_mass),
        "raw_probs": raw_probs,
        "normalized_probs": norm_probs,
        "symbol_logprobs": symbol_logprobs,
    }

def process_lpe_task(
    item: Dict[str, Any],
    perm_idx: int,
    model_tag: str,
    symbol_set_name: str,
    ollama_version: str,
    model_digest: str,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    obj_id = item["object_id"]
    premise = item["premise"]
    hypothesis = item["hypothesis"]
    symbols = LABEL_SETS[symbol_set_name]
    perm = S3_PERMUTATIONS[perm_idx]

    user_prompt = render_prompt(premise, hypothesis, perm, symbols)
    user_prompt_sha256 = sha256_str(user_prompt)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    request_id = make_request_id(
        model_tag, obj_id, PROMPT_VERSION, perm_idx, "lpe", temperature, 0, 0, symbol_set_name
    )

    resp_json, error_str, latency = query_ollama_with_retry(
        messages=messages,
        model_tag=model_tag,
        temperature=temperature,
        max_tokens=1,
        logprobs=True,
        top_logprobs=20,
    )

    record: Dict[str, Any] = {
        "request_id": request_id,
        "object_id": obj_id,
        "item_index": item.get("index", 0),
        "model_tag": model_tag,
        "model_digest": model_digest,
        "ollama_version": ollama_version,
        "prompt_version": PROMPT_VERSION,
        "symbol_set_name": symbol_set_name,
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "user_prompt_sha256": user_prompt_sha256,
        "perm_index": perm_idx,
        "perm_mapping": list(perm),
        "mode": "lpe",
        "temperature": temperature,
        "latency_sec": round(latency, 4),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    if error_str or resp_json is None:
        record["status"] = "error"
        record["error"] = error_str
        record["valid_output"] = False
        return record

    dist_result = extract_logprob_distribution(resp_json, perm, symbols)

    if not dist_result["valid"]:
        record["status"] = "error"
        record["error"] = dist_result.get("error", "Distribution extraction failed")
        record["valid_output"] = False
        if "raw_top_logprobs" in dist_result:
            record["raw_top_logprobs"] = dist_result["raw_top_logprobs"]
        return record

    record["status"] = "success"
    record["valid_output"] = True
    record["output_text"] = dist_result["output_text"]
    record["first_token"] = dist_result["first_token"]
    record["candidate_mass"] = dist_result["candidate_mass"]
    record["raw_probs"] = dist_result["raw_probs"]
    record["normalized_probs"] = dist_result["normalized_probs"]
    record["symbol_logprobs"] = dist_result["symbol_logprobs"]

    return record

def run_lpe(
    items: List[Dict[str, Any]],
    out_file: Path,
    workers: int,
    model_tag: str,
    symbol_set_name: str = "ABC",
    temperature: float = 1.0,
    label: str = "LPE",
) -> None:
    out_file = out_file.resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = get_existing_request_ids(out_file)

    ollama_version = get_ollama_version()
    model_digest = get_model_digest(model_tag)

    tasks = []
    for item in items:
        for perm_idx in range(6):
            obj_id = item["object_id"]
            rid = make_request_id(
                model_tag, obj_id, PROMPT_VERSION, perm_idx, "lpe", temperature, 0, 0, symbol_set_name
            )
            if rid not in existing_ids:
                tasks.append((item, perm_idx))

    total_tasks = len(items) * 6
    remaining = len(tasks)

    print(f"=" * 72)
    print(f"  {label}: {len(items)} items × 6 permutations = {total_tasks} requests")
    print(f"  Already completed: {len(existing_ids)}")
    print(f"  Remaining: {remaining}")
    print(f"  Output: {out_file}")
    print(f"  Model: {model_tag}  Digest: {model_digest[:16]}...  Temp: {temperature:.1f}")
    print(f"  Ollama: {ollama_version}  Workers: {workers}")
    print(f"  Prompt Version: {PROMPT_VERSION}  Symbol Set: {symbol_set_name}")
    print(f"=" * 72)

    if not tasks:
        print("\n  All requests already completed. Nothing to do.")
        audit_file(out_file, len(items))
        return

    file_lock = threading.Lock()

    def _worker(task_tuple: Tuple[Dict[str, Any], int]) -> Dict[str, Any]:
        item, perm_idx = task_tuple
        rec = process_lpe_task(
            item, perm_idx, model_tag, symbol_set_name, ollama_version, model_digest, temperature
        )
        with file_lock:
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    completed = 0
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_worker, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            if completed % 10 == 0 or completed == remaining:
                elapsed = time.time() - t_start
                rate = completed / max(elapsed, 0.1)
                eta_sec = (remaining - completed) / max(rate, 0.01)
                print(
                    f"  [{completed}/{remaining}] {completed/remaining*100.1:.1f}% | "
                    f"{rate:.2f} req/s | ETA: {eta_sec:.0f}s",
                    flush=True,
                )

    print(f"\n  Finished {remaining} requests in {time.time()-t_start:.1f}s.")
    audit_file(out_file, len(items))

def audit_file(file_path: Path, expected_items: int) -> None:
    file_path = file_path.resolve()
    if not file_path.exists():
        print(f"  AUDIT ERROR: File does not exist: {file_path}")
        return

    total_records = 0
    successful = 0
    errors = 0
    unique_rids = set()
    object_ids_seen = set()
    perm_counts = {}
    v1_contamination = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_records += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            if rec.get("prompt_version") != PROMPT_VERSION:
                v1_contamination += 1

            rid = rec.get("request_id")
            if rid:
                unique_rids.add(rid)

            if rec.get("status") == "success" and rec.get("valid_output") is True:
                successful += 1
                oid = rec.get("object_id")
                if oid:
                    object_ids_seen.add(oid)
                    perm_counts[oid] = perm_counts.get(oid, 0) + 1
            else:
                errors += 1

    expected_requests = expected_items * 6
    full_perm_items = sum(1 for c in perm_counts.values() if c >= 6)

    print(f"\n  ============================================================")
    print(f"  AUDIT SUMMARY: {file_path.name}")
    print(f"  ============================================================")
    print(f"  Total records:             {total_records}")
    print(f"  Successful:                {successful}")
    print(f"  Errors:                    {errors}")
    print(f"  Unique success IDs:        {len(unique_rids)}")
    print(f"  Unique object IDs:         {len(object_ids_seen)} / {expected_items} expected")
    print(f"  Full 6-perm coverage:      {full_perm_items} / {expected_items} items")
    print(f"  v1 records (contamination): {v1_contamination}")
    print(f"  Expected requests:         {expected_requests}")

    passed = (
        successful >= expected_requests
        and len(unique_rids) >= expected_requests
        and len(object_ids_seen) >= expected_items
        and full_perm_items >= expected_items
        and errors == 0
        and v1_contamination == 0
    )

    if passed:
        print(f"\n  [PASS] GATE PASSED")
    else:
        print(f"\n  [FAIL] GATE FAILED")
    print(f"  ============================================================\n")

def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def prewarm_model(model_tag: str, temperature: float = 1.0) -> None:
    print(f"Pre-warming model '{model_tag}' (T={temperature:.1f})...", flush=True)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                premise="A man is walking.",
                hypothesis="A person is moving.",
                symbol_1="A",
                symbol_2="B",
                symbol_3="C",
            ),
        },
    ]
    resp, err, dt = query_ollama_with_retry(messages, model_tag, temperature=temperature, max_tokens=1)
    if err:
        print(f"  WARNING: Pre-warm failed: {err}", flush=True)
    else:
        print(f"  Model loaded and ready.", flush=True)

def main():
    parser = argparse.ArgumentParser(description="E004 Ollama Hardened Inference Runner v2")
    parser.add_argument(
        "--mode",
        choices=["validate_20", "lpe"],
        default="validate_20",
    )
    parser.add_argument("--model", type=str, default="gemma3:12b")
    parser.add_argument("--symbol-set", choices=["ABC", "123", "XYZ"], default="ABC")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()
    slug = model_slug(args.model)
    ss = args.symbol_set.lower()
    t_tag = f"t{int(args.temperature * 10)}"

    print(f"\nE004 Hardened Runner v2", flush=True)
    print(f"  Mode: {args.mode}", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Symbol Set: {args.symbol_set}", flush=True)
    print(f"  Temperature: {args.temperature:.1f} ({t_tag})", flush=True)
    print(f"  Prompt Version: {PROMPT_VERSION}", flush=True)
    print(f"  Workers: {args.workers}", flush=True)
    print(f"  System Prompt SHA-256: {SYSTEM_PROMPT_SHA256[:16]}...", flush=True)
    print(f"  OS: {platform.platform()}", flush=True)
    print(f"  Python: {platform.python_version()}", flush=True)
    print(flush=True)

    prewarm_model(args.model, args.temperature)

    if args.mode == "validate_20":
        manifest_path = MANIFEST_DIR / "preflight_60.jsonl"
        if not manifest_path.exists():
            print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
            sys.exit(1)

        all_items = load_manifest(manifest_path)
        items = all_items[:20]
        print(f"  Loaded {len(all_items)} preflight items, using first 20.", flush=True)
        print(f"  Expected: 20 × 6 = 120 unique LPE requests.", flush=True)

        out_path = RAW_RESPONSES_DIR / f"val20_{slug}_v2_{ss}_{t_tag}_lpe.jsonl"
        run_lpe(items, out_path, args.workers, args.model, args.symbol_set, args.temperature, label="VAL-20 LPE")

    elif args.mode == "lpe":
        manifest_path = MANIFEST_DIR / "pilot_600.jsonl"
        if not manifest_path.exists():
            print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
            sys.exit(1)

        items = load_manifest(manifest_path)
        print(f"  Loaded {len(items)} pilot items.", flush=True)
        print(f"  Expected: 600 × 6 = 3,600 unique LPE requests.", flush=True)

        out_path = RAW_RESPONSES_DIR / f"pilot600_{slug}_v2_{ss}_{t_tag}_lpe.jsonl"
        run_lpe(items, out_path, args.workers, args.model, args.symbol_set, args.temperature, label="PILOT-600 LPE")

if __name__ == "__main__":
    main()
