"""E004 Ollama Hardened Inference Runner — v2.

Addresses all four blocking issues from the pre-launch audit:
  1. Frozen inference contract: PROMPT_VERSION="v2", PRIMARY_SYMBOL_SET="ABC",
     full provenance per record (prompt hashes, model digest, Ollama version,
     request-body hash, latency, success/error status).
  2. validate_20 truly sends only 20 items (120 requests), not 60.
  3. Robust retry with exponential backoff, structured error records,
     failed-request queue, and audit summary.
  4. Separate MCE execution path with deterministic per-item seeds.

Output files are named:
  val20_{model_slug}_v2_{symbol}_lpe.jsonl
  pilot600_{model_slug}_v2_{symbol}_lpe.jsonl
  pilot600_{model_slug}_v2_{symbol}_mce.jsonl

Old v1 response files are NEVER appended to.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ──────────────────────────────────────────────────────────────────────
# Frozen inference contract
# ──────────────────────────────────────────────────────────────────────

PROMPT_VERSION = "v2"
PRIMARY_SYMBOL_SET = "ABC"

API_BASE = "http://localhost:11434"
API_CHAT = f"{API_BASE}/v1/chat/completions"
API_VERSION = f"{API_BASE}/api/version"
API_SHOW = f"{API_BASE}/api/show"

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

NLI_LABELS = ["entailment", "neutral", "contradiction"]

# All 6 permutations of S3 (label-to-symbol assignment order)
S3_PERMUTATIONS: List[Tuple[int, int, int]] = [
    (0, 1, 2),  # perm 0: E→s1, N→s2, C→s3
    (0, 2, 1),  # perm 1: E→s1, N→s3, C→s2
    (1, 0, 2),  # perm 2: E→s2, N→s1, C→s3
    (1, 2, 0),  # perm 3: E→s2, N→s3, C→s1
    (2, 0, 1),  # perm 4: E→s3, N→s1, C→s2
    (2, 1, 0),  # perm 5: E→s3, N→s2, C→s1
]

# Pre-compute frozen prompt hashes (invariant across calls)
SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

# Retry configuration
MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 1.0
BACKOFF_FACTOR = 2.0
REQUEST_TIMEOUT_SEC = 60

# MCE configuration
MCE_TEMPERATURE = 1.0
MCE_REPLICATES_PER_MAPPING = 5  # 5 replicates × 6 mappings = 30 samples/item
MCE_BASE_SEED = 42

# ──────────────────────────────────────────────────────────────────────
# Thread-local HTTP sessions
# ──────────────────────────────────────────────────────────────────────

_THREAD_LOCAL_SESSIONS: Dict[int, requests.Session] = {}
_SESSION_LOCK = threading.Lock()


def get_session() -> requests.Session:
    """Get or create a thread-local HTTP session with connection pooling."""
    tid = threading.get_ident()
    if tid not in _THREAD_LOCAL_SESSIONS:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        with _SESSION_LOCK:
            _THREAD_LOCAL_SESSIONS[tid] = session
    return _THREAD_LOCAL_SESSIONS[tid]


# ──────────────────────────────────────────────────────────────────────
# Provenance helpers
# ──────────────────────────────────────────────────────────────────────

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def get_ollama_version() -> str:
    """Query Ollama server for its version string."""
    try:
        r = requests.get(API_VERSION, timeout=5)
        r.raise_for_status()
        return r.json().get("version", "unknown")
    except Exception:
        return "unknown"


def get_model_digest(model_tag: str) -> str:
    """Query Ollama for the model digest."""
    try:
        r = requests.post(API_SHOW, json={"name": model_tag}, timeout=30)
        r.raise_for_status()
        data = r.json()
        # Try to extract digest from modelinfo or details
        if "modelinfo" in data:
            return sha256_str(json.dumps(data["modelinfo"], sort_keys=True))
        return data.get("digest", "unknown")
    except Exception:
        return "unknown"


def model_slug(model_tag: str) -> str:
    """Convert model tag to safe filename slug."""
    return model_tag.replace(":", "-").replace("/", "-")


# ──────────────────────────────────────────────────────────────────────
# Request ID generation
# ──────────────────────────────────────────────────────────────────────

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
    """Deterministic request ID from all parameters."""
    raw = (
        f"E004|{model_tag}|{object_id}|{prompt_version}|{symbol_set_name}"
        f"|{perm_idx}|{mode}|{temperature:.4f}|{seed}|{replicate}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# Checkpoint / resume
# ──────────────────────────────────────────────────────────────────────

def get_existing_request_ids(file_path: Path) -> set:
    """Load request IDs already completed in the output file."""
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


# ──────────────────────────────────────────────────────────────────────
# Robust Ollama query with retry
# ──────────────────────────────────────────────────────────────────────

def query_ollama_with_retry(
    messages: List[Dict[str, str]],
    model_tag: str,
    temperature: float = 0.0,
    max_tokens: int = 1,
    logprobs: bool = True,
    top_logprobs: int = 20,
    seed: Optional[int] = None,
) -> Tuple[Optional[Dict], Optional[str], float]:
    """
    Send a chat completion request with exponential-backoff retry.

    Returns (response_json, error_message, latency_sec).
    On success: (dict, None, latency).
    On final failure: (None, error_string, total_latency).
    """
    session = get_session()
    payload: Dict[str, Any] = {
        "model": model_tag,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": logprobs,
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
            total_latency += REQUEST_TIMEOUT_SEC  # approximate
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 0
            last_error = f"HTTPError {status_code}: {e}"
            # Don't retry 4xx client errors (except 429)
            if 400 <= status_code < 500 and status_code != 429:
                return None, last_error, total_latency
        except Exception as e:
            last_error = f"Unexpected: {type(e).__name__}: {e}"

        if attempt < MAX_RETRIES - 1:
            backoff = INITIAL_BACKOFF_SEC * (BACKOFF_FACTOR ** attempt)
            time.sleep(backoff)

    return None, last_error, total_latency


# ──────────────────────────────────────────────────────────────────────
# Render prompts and compute hashes
# ──────────────────────────────────────────────────────────────────────

def render_user_prompt(
    item: Dict, perm: Tuple[int, int, int], symbols: List[str]
) -> str:
    """Render the user prompt for a specific item and permutation."""
    s1, s2, s3 = symbols[perm[0]], symbols[perm[1]], symbols[perm[2]]
    return USER_PROMPT_TEMPLATE.format(
        premise=item["premise"],
        hypothesis=item["hypothesis"],
        symbol_1=s1,
        symbol_2=s2,
        symbol_3=s3,
    )


# ──────────────────────────────────────────────────────────────────────
# LPE task processor
# ──────────────────────────────────────────────────────────────────────

def process_lpe_task(task: Dict) -> Dict:
    """
    Process a single LPE request. Returns a complete record dict
    with status="success" or status="error".
    """
    model_tag = task["model_tag"]
    item = task["item"]
    perm_idx = task["perm_idx"]
    perm = task["perm"]
    symbols = task["symbols"]
    symbol_set_name = task["symbol_set_name"]
    ollama_version = task["ollama_version"]
    model_digest = task["model_digest"]
    temperature = task.get("temperature", 0.0)

    object_id = item["object_id"]
    row_index = item["row_index"]

    mode_name = "lpe_t10" if temperature == 1.0 else "lpe"
    req_id = make_request_id(
        model_tag, object_id, PROMPT_VERSION, perm_idx,
        mode_name, temperature, 0, 0, symbol_set_name
    )

    s1, s2, s3 = symbols[perm[0]], symbols[perm[1]], symbols[perm[2]]
    user_content = render_user_prompt(item, perm, symbols)
    user_prompt_sha256 = sha256_str(user_content)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # Compute full request body hash for provenance
    request_body = {
        "model": model_tag,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
    }
    request_body_hash = sha256_bytes(
        json.dumps(request_body, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )

    # Common record fields
    base_record = {
        "request_id": req_id,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "object_id": object_id,
        "row_index": row_index,
        "prompt_version": PROMPT_VERSION,
        "symbol_set": symbol_set_name,
        "perm_idx": perm_idx,
        "perm_tuple": list(perm),
        "symbol_mapping": {"E": s1, "N": s2, "C": s3},
        "mode": mode_name,
        "temperature": temperature,
        "max_tokens": 1,
        "seed": None,
        "replicate": 0,
        "model_tag": model_tag,
        "model_digest": model_digest,
        "ollama_version": ollama_version,
        "user_prompt_sha256": user_prompt_sha256,
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "request_body_hash": request_body_hash,
    }

    resp_json, error, latency = query_ollama_with_retry(
        messages, model_tag, temperature=temperature, max_tokens=1,
        logprobs=True, top_logprobs=20
    )

    if error is not None:
        base_record.update({
            "status": "error",
            "error": error,
            "latency_sec": round(latency, 4),
            "response_text": None,
            "logprobs": None,
        })
        return base_record

    choice = resp_json["choices"][0]
    logprobs_content = choice.get("logprobs", {}).get("content", [])

    base_record.update({
        "status": "success",
        "error": None,
        "latency_sec": round(latency, 4),
        "response_text": choice["message"]["content"],
        "logprobs": logprobs_content,
    })
    return base_record


# ──────────────────────────────────────────────────────────────────────
# MCE task processor
# ──────────────────────────────────────────────────────────────────────

def process_mce_task(task: Dict) -> Dict:
    """
    Process a single MCE request. Returns a complete record dict.
    MCE uses temperature sampling with logprobs=False, parsing a single symbol.
    """
    model_tag = task["model_tag"]
    item = task["item"]
    perm_idx = task["perm_idx"]
    perm = task["perm"]
    symbols = task["symbols"]
    symbol_set_name = task["symbol_set_name"]
    replicate = task["replicate"]
    ollama_version = task["ollama_version"]
    model_digest = task["model_digest"]

    object_id = item["object_id"]
    row_index = item["row_index"]

    # Deterministic seed: hash of (object_id, perm_idx, replicate)
    seed_raw = f"MCE|{object_id}|{perm_idx}|{replicate}"
    seed = int(hashlib.sha256(seed_raw.encode("utf-8")).hexdigest()[:8], 16)

    req_id = make_request_id(
        model_tag, object_id, PROMPT_VERSION, perm_idx,
        "mce", MCE_TEMPERATURE, seed, replicate, symbol_set_name
    )

    s1, s2, s3 = symbols[perm[0]], symbols[perm[1]], symbols[perm[2]]
    user_content = render_user_prompt(item, perm, symbols)
    user_prompt_sha256 = sha256_str(user_content)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    request_body = {
        "model": model_tag,
        "messages": messages,
        "temperature": MCE_TEMPERATURE,
        "max_tokens": 1,
        "logprobs": False,
        "seed": seed,
    }
    request_body_hash = sha256_bytes(
        json.dumps(request_body, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )

    base_record = {
        "request_id": req_id,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "object_id": object_id,
        "row_index": row_index,
        "prompt_version": PROMPT_VERSION,
        "symbol_set": symbol_set_name,
        "perm_idx": perm_idx,
        "perm_tuple": list(perm),
        "symbol_mapping": {"E": s1, "N": s2, "C": s3},
        "mode": "mce",
        "temperature": MCE_TEMPERATURE,
        "max_tokens": 1,
        "seed": seed,
        "replicate": replicate,
        "model_tag": model_tag,
        "model_digest": model_digest,
        "ollama_version": ollama_version,
        "user_prompt_sha256": user_prompt_sha256,
        "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
        "request_body_hash": request_body_hash,
    }

    resp_json, error, latency = query_ollama_with_retry(
        messages, model_tag, temperature=MCE_TEMPERATURE, max_tokens=1,
        logprobs=False, seed=seed
    )

    if error is not None:
        base_record.update({
            "status": "error",
            "error": error,
            "latency_sec": round(latency, 4),
            "response_text": None,
            "parsed_symbol": None,
            "parsed_label": None,
            "valid_output": False,
        })
        return base_record

    choice = resp_json["choices"][0]
    raw_text = choice["message"]["content"].strip()

    # Parse one-symbol response
    valid_symbols = set(symbols)
    parsed_symbol = raw_text if raw_text in valid_symbols else None
    parsed_label = None
    if parsed_symbol is not None:
        # Map back through permutation: which NLI label does this symbol represent?
        symbol_to_label = {symbols[perm[i]]: NLI_LABELS[i] for i in range(3)}
        parsed_label = symbol_to_label.get(parsed_symbol)

    base_record.update({
        "status": "success",
        "error": None,
        "latency_sec": round(latency, 4),
        "response_text": raw_text,
        "parsed_symbol": parsed_symbol,
        "parsed_label": parsed_label,
        "valid_output": parsed_symbol is not None,
    })
    return base_record


# ──────────────────────────────────────────────────────────────────────
# Manifest loading
# ──────────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> List[Dict]:
    """Load JSONL manifest into a list of item dicts."""
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ──────────────────────────────────────────────────────────────────────
# Run orchestrators
# ──────────────────────────────────────────────────────────────────────

def run_lpe(
    items: List[Dict],
    output_path: Path,
    max_workers: int,
    model_tag: str,
    symbol_set_name: str,
    label: str = "LPE",
    temperature: float = 0.0,
):
    """Execute LPE inference over items with full provenance and retry."""
    symbols = LABEL_SETS[symbol_set_name]
    existing_ids = get_existing_request_ids(output_path)

    # Capture provenance once
    ollama_version = get_ollama_version()
    m_digest = get_model_digest(model_tag)

    mode_name = "lpe_t10" if temperature == 1.0 else "lpe"
    # Build task list
    task_list = []
    for item in items:
        for perm_idx, perm in enumerate(S3_PERMUTATIONS):
            req_id = make_request_id(
                model_tag, item["object_id"], PROMPT_VERSION, perm_idx,
                mode_name, temperature, 0, 0, symbol_set_name
            )
            if req_id in existing_ids:
                continue
            task_list.append({
                "model_tag": model_tag,
                "item": item,
                "perm_idx": perm_idx,
                "perm": perm,
                "symbols": symbols,
                "symbol_set_name": symbol_set_name,
                "ollama_version": ollama_version,
                "model_digest": m_digest,
                "temperature": temperature,
            })

    total_new = len(task_list)
    total_expected = len(items) * 6
    print(f"\n{'='*72}", flush=True)
    print(f"  {label}: {len(items)} items × 6 permutations = {total_expected} requests", flush=True)
    print(f"  Already completed: {len(existing_ids)}", flush=True)
    print(f"  Remaining: {total_new}", flush=True)
    print(f"  Output: {output_path}", flush=True)
    print(f"  Model: {model_tag}  Digest: {m_digest[:16]}...", flush=True)
    print(f"  Ollama: {ollama_version}  Workers: {max_workers}", flush=True)
    print(f"  Prompt Version: {PROMPT_VERSION}  Symbol Set: {symbol_set_name}", flush=True)
    print(f"{'='*72}\n", flush=True)

    if total_new == 0:
        print("  All requests already completed. Nothing to do.", flush=True)
        _print_audit_summary(output_path, items, "lpe", symbol_set_name)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    errors = 0
    t0 = time.time()

    with open(output_path, "a", encoding="utf-8") as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_lpe_task, t): t for t in task_list
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    rec = future.result()
                except Exception as e:
                    # Should not happen — process_lpe_task handles errors internally
                    task_info = futures[future]
                    rec = {
                        "request_id": "EXECUTOR_ERROR",
                        "object_id": task_info["item"]["object_id"],
                        "status": "error",
                        "error": f"ExecutorError: {type(e).__name__}: {e}",
                        "timestamp_utc": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                    }

                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()

                if rec.get("status") == "success":
                    completed += 1
                else:
                    errors += 1

                done = completed + errors
                if done % 20 == 0 or done == total_new:
                    elapsed = time.time() - t0
                    rate = completed / max(0.1, elapsed)
                    print(
                        f"  [{done}/{total_new}] "
                        f"OK={completed} ERR={errors} "
                        f"({rate:.2f} req/s | {elapsed:.1f}s)",
                        flush=True,
                    )

    elapsed = time.time() - t0
    print(f"\n  Finished in {elapsed:.1f}s. Success={completed}, Errors={errors}", flush=True)
    _print_audit_summary(output_path, items, "lpe", symbol_set_name)


def run_mce(
    items: List[Dict],
    output_path: Path,
    max_workers: int,
    model_tag: str,
    symbol_set_name: str,
):
    """Execute MCE inference: items × 6 permutations × 5 replicates = 30 samples/item."""
    symbols = LABEL_SETS[symbol_set_name]
    existing_ids = get_existing_request_ids(output_path)

    ollama_version = get_ollama_version()
    m_digest = get_model_digest(model_tag)

    task_list = []
    for item in items:
        for perm_idx, perm in enumerate(S3_PERMUTATIONS):
            for replicate in range(MCE_REPLICATES_PER_MAPPING):
                seed_raw = f"MCE|{item['object_id']}|{perm_idx}|{replicate}"
                seed = int(hashlib.sha256(seed_raw.encode("utf-8")).hexdigest()[:8], 16)
                req_id = make_request_id(
                    model_tag, item["object_id"], PROMPT_VERSION, perm_idx,
                    "mce", MCE_TEMPERATURE, seed, replicate, symbol_set_name
                )
                if req_id in existing_ids:
                    continue
                task_list.append({
                    "model_tag": model_tag,
                    "item": item,
                    "perm_idx": perm_idx,
                    "perm": perm,
                    "symbols": symbols,
                    "symbol_set_name": symbol_set_name,
                    "replicate": replicate,
                    "ollama_version": ollama_version,
                    "model_digest": m_digest,
                })

    total_new = len(task_list)
    total_expected = len(items) * 6 * MCE_REPLICATES_PER_MAPPING
    print(f"\n{'='*72}", flush=True)
    print(f"  MCE: {len(items)} items × 6 perms × {MCE_REPLICATES_PER_MAPPING} replicates = {total_expected} requests", flush=True)
    print(f"  Already completed: {len(existing_ids)}", flush=True)
    print(f"  Remaining: {total_new}", flush=True)
    print(f"  Output: {output_path}", flush=True)
    print(f"  Model: {model_tag}  Temperature: {MCE_TEMPERATURE}", flush=True)
    print(f"  Prompt Version: {PROMPT_VERSION}  Symbol Set: {symbol_set_name}", flush=True)
    print(f"{'='*72}\n", flush=True)

    if total_new == 0:
        print("  All MCE requests already completed.", flush=True)
        _print_audit_summary(output_path, items, "mce", symbol_set_name)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    errors = 0
    invalid_outputs = 0
    t0 = time.time()

    with open(output_path, "a", encoding="utf-8") as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_mce_task, t): t for t in task_list
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    rec = future.result()
                except Exception as e:
                    task_info = futures[future]
                    rec = {
                        "request_id": "EXECUTOR_ERROR",
                        "object_id": task_info["item"]["object_id"],
                        "status": "error",
                        "error": f"ExecutorError: {type(e).__name__}: {e}",
                        "timestamp_utc": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                    }

                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()

                if rec.get("status") == "success":
                    completed += 1
                    if not rec.get("valid_output"):
                        invalid_outputs += 1
                else:
                    errors += 1

                done = completed + errors
                if done % 100 == 0 or done == total_new:
                    elapsed = time.time() - t0
                    rate = completed / max(0.1, elapsed)
                    print(
                        f"  [{done}/{total_new}] "
                        f"OK={completed} ERR={errors} Invalid={invalid_outputs} "
                        f"({rate:.2f} req/s | {elapsed:.1f}s)",
                        flush=True,
                    )

    elapsed = time.time() - t0
    print(f"\n  Finished in {elapsed:.1f}s. Success={completed}, Errors={errors}, Invalid={invalid_outputs}", flush=True)
    _print_audit_summary(output_path, items, "mce", symbol_set_name)


# ──────────────────────────────────────────────────────────────────────
# Audit summary
# ──────────────────────────────────────────────────────────────────────

def _print_audit_summary(
    output_path: Path,
    items: List[Dict],
    mode: str,
    symbol_set_name: str,
):
    """Print a comprehensive audit summary of completed output."""
    if not output_path.exists():
        print("\n  [AUDIT] No output file found.", flush=True)
        return

    records = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    success_recs = [r for r in records if r.get("status") == "success"]
    error_recs = [r for r in records if r.get("status") == "error"]

    unique_success_ids = set(r["request_id"] for r in success_recs)
    unique_object_ids = set(r["object_id"] for r in success_recs)
    expected_object_ids = set(item["object_id"] for item in items)

    # Check permutation coverage
    perm_coverage: Dict[str, set] = {}
    for r in success_recs:
        oid = r["object_id"]
        if oid not in perm_coverage:
            perm_coverage[oid] = set()
        perm_coverage[oid].add(r.get("perm_idx"))

    full_perm_coverage = sum(1 for s in perm_coverage.values() if s == {0, 1, 2, 3, 4, 5})

    # Check for v1 contamination
    v1_count = sum(1 for r in records if r.get("prompt_version") == "v1")

    print(f"\n  {'='*60}", flush=True)
    print(f"  AUDIT SUMMARY: {output_path.name}", flush=True)
    print(f"  {'='*60}", flush=True)
    print(f"  Total records:             {len(records)}", flush=True)
    print(f"  Successful:                {len(success_recs)}", flush=True)
    print(f"  Errors:                    {len(error_recs)}", flush=True)
    print(f"  Unique success IDs:        {len(unique_success_ids)}", flush=True)
    print(f"  Unique object IDs:         {len(unique_object_ids)} / {len(expected_object_ids)} expected", flush=True)
    print(f"  Full 6-perm coverage:      {full_perm_coverage} / {len(expected_object_ids)} items", flush=True)
    print(f"  v1 records (contamination): {v1_count}", flush=True)

    if mode == "lpe":
        expected_total = len(items) * 6
        print(f"  Expected requests:         {expected_total}", flush=True)
        gate_pass = (
            len(unique_success_ids) == expected_total
            and full_perm_coverage == len(expected_object_ids)
            and v1_count == 0
        )
    elif mode == "mce":
        expected_total = len(items) * 6 * MCE_REPLICATES_PER_MAPPING
        invalid_count = sum(1 for r in success_recs if not r.get("valid_output", True))
        print(f"  Expected requests:         {expected_total}", flush=True)
        print(f"  Invalid outputs:           {invalid_count}", flush=True)
        gate_pass = (
            len(unique_success_ids) == expected_total
            and v1_count == 0
        )
    else:
        gate_pass = False

    if gate_pass:
        print(f"\n  [PASS] GATE PASSED", flush=True)
    else:
        missing_objects = expected_object_ids - unique_object_ids
        if missing_objects:
            print(f"\n  Missing objects ({len(missing_objects)}): {list(missing_objects)[:5]}...", flush=True)
        if error_recs:
            print(f"\n  Failed requests ({len(error_recs)}):", flush=True)
            for er in error_recs[:5]:
                print(f"    {er.get('object_id', '?')}: {er.get('error', '?')[:80]}", flush=True)
        print(f"\n  [FAIL] GATE NOT PASSED -- review errors above", flush=True)

    print(f"  {'='*60}\n", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Model prewarm
# ──────────────────────────────────────────────────────────────────────

def prewarm_model(model_tag: str):
    """Send a trivial request to load the model into GPU memory."""
    print(f"Pre-warming model '{model_tag}'...", flush=True)
    session = get_session()
    payload = {
        "model": model_tag,
        "messages": [
            {"role": "system", "content": "Hi"},
            {"role": "user", "content": "Ping"},
        ],
        "max_tokens": 1,
    }
    try:
        resp = session.post(API_CHAT, json=payload, timeout=120)
        resp.raise_for_status()
        print("  Model loaded and ready.", flush=True)
    except Exception as e:
        print(f"  WARNING: Prewarm failed: {e}", flush=True)
        print("  Proceeding anyway — first real request will load the model.", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Main CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="E004 Hardened Ollama Inference Runner (v2)"
    )
    parser.add_argument(
        "--mode",
        choices=["validate_20", "validate_mce_20", "lpe", "lpe_t10", "mce"],
        required=True,
        help="validate_20: 20-item LPE gate. validate_mce_20: 20-item MCE gate. lpe: full LPE run (T=0.0). lpe_t10: full LPE run (T=1.0). mce: full MCE run.",
    )
    parser.add_argument(
        "--subset",
        choices=["pilot"],
        default="pilot",
        help="Item subset to use (pilot = 600 items).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent workers.",
    )
    parser.add_argument(
        "--model",
        default="gemma3:12b",
        help="Ollama model tag.",
    )
    parser.add_argument(
        "--symbol-set",
        choices=list(LABEL_SETS.keys()),
        default="ABC",
        help="Symbol set for label assignment.",
    )

    args = parser.parse_args()
    slug = model_slug(args.model)
    ss = args.symbol_set.lower()

    print(f"\nE004 Hardened Runner v2", flush=True)
    print(f"  Mode: {args.mode}", flush=True)
    print(f"  Model: {args.model}", flush=True)
    print(f"  Symbol Set: {args.symbol_set}", flush=True)
    print(f"  Prompt Version: {PROMPT_VERSION}", flush=True)
    print(f"  Workers: {args.workers}", flush=True)
    print(f"  System Prompt SHA-256: {SYSTEM_PROMPT_SHA256[:16]}...", flush=True)
    print(f"  OS: {platform.platform()}", flush=True)
    print(f"  Python: {platform.python_version()}", flush=True)
    print(flush=True)

    prewarm_model(args.model)

    if args.mode == "validate_20":
        # Gate 1: True 20-item validation — deterministic first 20 from preflight
        manifest_path = MANIFEST_DIR / "preflight_60.jsonl"
        if not manifest_path.exists():
            print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
            sys.exit(1)

        all_items = load_manifest(manifest_path)
        items = all_items[:20]  # Exactly 20 items — NOT all 60
        print(f"  Loaded {len(all_items)} preflight items, using first 20.", flush=True)
        print(f"  Expected: 20 × 6 = 120 unique LPE requests.", flush=True)

        out_path = RAW_RESPONSES_DIR / f"val20_{slug}_v2_{ss}_lpe.jsonl"
        run_lpe(items, out_path, args.workers, args.model, args.symbol_set, label="VAL-20 LPE")

    elif args.mode == "validate_mce_20":
        # 20-item MCE Validation — deterministic first 20 items from preflight
        manifest_path = MANIFEST_DIR / "preflight_60.jsonl"
        if not manifest_path.exists():
            print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
            sys.exit(1)

        all_items = load_manifest(manifest_path)
        items = all_items[:20]  # Exactly 20 items — 20 x 6 x 5 = 600 MCE calls
        print(f"  Loaded {len(all_items)} preflight items, using first 20.", flush=True)
        print(f"  Expected: 20 x 6 x {MCE_REPLICATES_PER_MAPPING} = 600 unique MCE requests.", flush=True)

        out_path = RAW_RESPONSES_DIR / f"val20_{slug}_v2_{ss}_mce.jsonl"
        run_mce(items, out_path, args.workers, args.model, args.symbol_set)

    elif args.mode in ["lpe", "lpe_t10"]:
        # Pilot LPE — 600 items
        manifest_path = MANIFEST_DIR / "pilot_600.jsonl"
        if not manifest_path.exists():
            print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
            sys.exit(1)

        items = load_manifest(manifest_path)
        temp = 1.0 if args.mode == "lpe_t10" else 0.0
        t_slug = "t10_" if temp == 1.0 else ""
        print(f"  Loaded {len(items)} pilot items.", flush=True)
        print(f"  Expected: {len(items)} × 6 = {len(items) * 6} unique LPE (T={temp}) requests.", flush=True)

        out_path = RAW_RESPONSES_DIR / f"pilot600_{slug}_v2_{ss}_{t_slug}lpe.jsonl"
        run_lpe(items, out_path, args.workers, args.model, args.symbol_set, label=f"PILOT-600 LPE (T={temp})", temperature=temp)

    elif args.mode == "mce":
        # Gate 3: Pilot MCE — 600 items × 6 × 5 = 18,000 requests
        manifest_path = MANIFEST_DIR / "pilot_600.jsonl"
        if not manifest_path.exists():
            print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
            sys.exit(1)

        items = load_manifest(manifest_path)
        print(f"  Loaded {len(items)} pilot items.", flush=True)
        print(f"  Expected: {len(items)} × 6 × {MCE_REPLICATES_PER_MAPPING} = {len(items) * 6 * MCE_REPLICATES_PER_MAPPING} unique MCE requests.", flush=True)

        out_path = RAW_RESPONSES_DIR / f"pilot600_{slug}_v2_{ss}_mce.jsonl"
        run_mce(items, out_path, args.workers, args.model, args.symbol_set)

    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
