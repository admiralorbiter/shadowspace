"""E004 Temperature vs Logprobs Diagnostic (Multi-Worker).

Tests whether Ollama's returned logprobs for candidate symbols (A, B, C)
are identical or scale-dependent across request temperatures T in {0.0, 0.5, 1.0}.
Executes 20 items x 6 perms x 3 temperatures = 360 LPE requests with logprobs=True.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter

API_CHAT = "http://localhost:11434/v1/chat/completions"
MANIFEST_PATH = Path("research/chaosnli/artifacts/E004/manifests/preflight_60.jsonl")

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

LABEL_SETS = {"ABC": ["A", "B", "C"]}
S3_PERMUTATIONS = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]

_THREAD_LOCAL_SESSIONS: Dict[int, requests.Session] = {}


def get_session() -> requests.Session:
    tid = threading.get_ident()
    if tid not in _THREAD_LOCAL_SESSIONS:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _THREAD_LOCAL_SESSIONS[tid] = session
    return _THREAD_LOCAL_SESSIONS[tid]


def process_task(task: Dict) -> Dict:
    session = get_session()
    item = task["item"]
    perm_idx = task["perm_idx"]
    perm = task["perm"]
    symbols = task["symbols"]
    temp = task["temp"]

    s1, s2, s3 = symbols[perm[0]], symbols[perm[1]], symbols[perm[2]]
    user_content = USER_PROMPT_TEMPLATE.format(
        premise=item["premise"],
        hypothesis=item["hypothesis"],
        symbol_1=s1,
        symbol_2=s2,
        symbol_3=s3,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    payload = {
        "model": "gemma3:12b",
        "messages": messages,
        "temperature": temp,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
    }

    resp = session.post(API_CHAT, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    lps = {e["token"]: e["logprob"] for e in top if e["token"] in symbols}

    return {
        "object_id": item["object_id"],
        "perm_idx": perm_idx,
        "temp": temp,
        "logprobs": lps,
    }


def main():
    print("=" * 72)
    print("   TEMPERATURE VS LOGPROBS DIAGNOSTIC TEST (T = 0.0 vs 0.5 vs 1.0)")
    print("=" * 72)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()][:20]

    symbols = LABEL_SETS["ABC"]
    temperatures = [0.0, 0.5, 1.0]

    task_list = []
    for item in items:
        for perm_idx, perm in enumerate(S3_PERMUTATIONS):
            for temp in temperatures:
                task_list.append({
                    "item": item,
                    "perm_idx": perm_idx,
                    "perm": perm,
                    "symbols": symbols,
                    "temp": temp,
                })

    total = len(task_list)
    results = {}
    completed = 0
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_task, t): t for t in task_list}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results[(res["object_id"], res["perm_idx"], res["temp"])] = res["logprobs"]
            completed += 1
            if completed % 30 == 0 or completed == total:
                elapsed = time.time() - t0
                print(f"  [{completed}/{total}] {completed / max(0.1, elapsed):.1f} req/s", flush=True)

    print("\n" + "=" * 72)
    print("   LOGPROBS COMPARISON ACROSS TEMPERATURES")
    print("=" * 72)

    diff_00_05 = []
    diff_00_10 = []

    for item in items:
        oid = item["object_id"]
        for perm_idx in range(6):
            lp_00 = results.get((oid, perm_idx, 0.0), {})
            lp_05 = results.get((oid, perm_idx, 0.5), {})
            lp_10 = results.get((oid, perm_idx, 1.0), {})

            for sym in symbols:
                val_00 = lp_00.get(sym)
                val_05 = lp_05.get(sym)
                val_10 = lp_10.get(sym)

                if val_00 is not None and val_05 is not None:
                    diff_00_05.append(abs(val_00 - val_05))
                if val_00 is not None and val_10 is not None:
                    diff_00_10.append(abs(val_00 - val_10))

    max_diff_05 = max(diff_00_05) if diff_00_05 else 0.0
    max_diff_10 = max(diff_00_10) if diff_00_10 else 0.0

    print(f"Max Logprob Absolute Difference (T=0.0 vs T=0.5): {max_diff_05:.8f}")
    print(f"Max Logprob Absolute Difference (T=0.0 vs T=1.0): {max_diff_10:.8f}")

    if max_diff_05 < 1e-5 and max_diff_10 < 1e-5:
        print("\n[DIAGNOSTIC RESULT]: Returned logprobs are EXACTLY IDENTICAL across temperatures.")
        print("  Temperature controls sampling choice only; logprobs represent true raw model logits.")
    else:
        print("\n[DIAGNOSTIC RESULT]: Returned logprobs DIFFER across temperatures.")
        print("  Ollama scales returned logprobs by request temperature before returning.")

    out_file = Path("research/chaosnli/artifacts/E004/summaries/E004_temperature_logprobs_diagnostic.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "num_items": len(items),
        "num_permutations": 6,
        "temperatures_tested": temperatures,
        "max_abs_diff_t00_vs_t05": max_diff_05,
        "max_abs_diff_t00_vs_t10": max_diff_10,
        "identical_logprobs": max_diff_05 < 1e-5 and max_diff_10 < 1e-5,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved diagnostic report to: {out_file}")


if __name__ == "__main__":
    main()
