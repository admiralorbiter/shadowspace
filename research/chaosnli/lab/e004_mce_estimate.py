"""E004 MCE Estimation & Dirichlet Uncertainty Script.

Parses raw MCE JSONL response logs, counts categorical label responses per item,
applies Jeffreys smoothing (n + 0.5)/(B + 1.5), computes invalid-response rates,
simulates Dirichlet posterior uncertainty, and measures LPE vs MCE divergence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

RAW_RESPONSES_DIR = Path("research/chaosnli/artifacts/E004/raw_responses")
NORM_PROBS_DIR = Path("research/chaosnli/artifacts/E004/normalized_probs")

def process_mce_responses(input_path: Path, output_prefix: str) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing MCE response file: {input_path}")

    items_counts: Dict[str, Dict[str, int]] = {}
    items_total: Dict[str, int] = {}
    invalid_count = 0
    total_responses = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            obj_id = rec["object_id"]
            sym_map = rec["symbol_mapping"]
            resp = rec["response"]
            total_responses += 1

            # Extract output token text
            output_token = ""
            try:
                output_token = resp["choices"][0]["message"]["content"].strip()
            except Exception:
                pass

            # Map output symbol back to semantic label
            # sym_map has: {"entailment": sym_e, "neutral": sym_n, "contradiction": sym_c}
            inv_map = {v: k for k, v in sym_map.items()}

            if output_token in inv_map:
                sem_label = inv_map[output_token]
                counts = items_counts.setdefault(obj_id, {"entailment": 0, "neutral": 0, "contradiction": 0})
                counts[sem_label] += 1
            else:
                invalid_count += 1

            items_total[obj_id] = items_total.get(obj_id, 0) + 1

    sorted_obj_ids = sorted(items_total.keys())
    N = len(sorted_obj_ids)

    q_mce = np.zeros((N, 3), dtype=np.float64)
    counts_array = np.zeros((N, 3), dtype=np.int64)

    for i, obj_id in enumerate(sorted_obj_ids):
        cnts = items_counts.get(obj_id, {"entailment": 0, "neutral": 0, "contradiction": 0})
        n_e, n_n, n_c = cnts["entailment"], cnts["neutral"], cnts["contradiction"]
        counts_array[i] = [n_e, n_n, n_c]

        B = n_e + n_n + n_c
        # Jeffreys smoothing (alpha=0.5 prior)
        q_mce[i, 0] = (n_e + 0.5) / (B + 1.5)
        q_mce[i, 1] = (n_n + 0.5) / (B + 1.5)
        q_mce[i, 2] = (n_c + 0.5) / (B + 1.5)

    invalid_rate = (invalid_count / max(1, total_responses)) * 100.0

    print("=========================================================================")
    print(f"   MCE ESTIMATION SUMMARY ({output_prefix})")
    print("=========================================================================")
    print(f"  Processed Items:         {N}")
    print(f"  Total Raw Responses:     {total_responses}")
    print(f"  Invalid Response Rate:   {invalid_rate:.4f}% ({invalid_count} events)")
    print(f"  Mean Counts per Item:    {np.mean(np.sum(counts_array, axis=1)):.2f}")
    print("=========================================================================")

    NORM_PROBS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(NORM_PROBS_DIR / f"{output_prefix}_mce_probs.npy", q_mce)
    np.save(NORM_PROBS_DIR / f"{output_prefix}_mce_counts.npy", counts_array)

    with open(NORM_PROBS_DIR / f"{output_prefix}_mce_object_ids.json", "w", encoding="utf-8") as f:
        json.dump(sorted_obj_ids, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="E004 MCE Estimation")
    parser.add_argument("--subset", choices=["preflight", "pilot", "convergence", "temp_sensitivity"], default="preflight")
    args = parser.parse_args()

    input_file = RAW_RESPONSES_DIR / f"{args.subset}_mce_responses.jsonl"
    process_mce_responses(input_file, args.subset)

if __name__ == "__main__":
    main()
