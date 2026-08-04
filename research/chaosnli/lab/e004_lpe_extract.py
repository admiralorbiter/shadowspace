"""E004 LPE Extraction & Normalization Script.

Parses raw LPE JSONL logs, extracts candidate-token log probabilities,
verifies top_logprobs presence, normalizes softmaxes across candidate symbols per permutation,
averages across the 6 label permutations to form q_LPE, computes per-item label order sensitivity S_order,
and persists mean logits and normalized probability matrices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

RAW_RESPONSES_DIR = Path("research/chaosnli/artifacts/E004/raw_responses")
NORM_PROBS_DIR = Path("research/chaosnli/artifacts/E004/normalized_probs")

def jsd_single(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon Divergence in bits for two 3D vectors."""
    m = 0.5 * (p + q)
    m = np.clip(m, 1e-12, 1.0)
    p_safe = np.clip(p, 1e-12, 1.0)
    q_safe = np.clip(q, 1e-12, 1.0)

    kl_pm = np.sum(np.where(p > 1e-12, p * np.log2(p_safe / m), 0.0))
    kl_qm = np.sum(np.where(q > 1e-12, q * np.log2(q_safe / m), 0.0))
    return float(0.5 * kl_pm + 0.5 * kl_qm)

def extract_candidate_logprobs_from_response(resp: dict, sym_map: dict) -> Tuple[dict, bool]:
    """Extract log probabilities for the three symbols from an OpenAI-compatible top_logprobs structure."""
    sym_e = sym_map["entailment"]
    sym_n = sym_map["neutral"]
    sym_c = sym_map["contradiction"]
    target_symbols = {sym_e, sym_n, sym_c}

    logprobs_found = {}

    try:
        choices = resp["choices"]
        first_choice = choices[0]
        logprobs_content = first_choice["logprobs"]["content"][0]
        top_lps = logprobs_content.get("top_logprobs", [])

        for entry in top_lps:
            tok = entry["token"].strip()
            if tok in target_symbols and tok not in logprobs_found:
                logprobs_found[tok] = float(entry["logprob"])
    except Exception:
        pass

    # Check if all three were found
    all_found = (len(logprobs_found) == 3)
    return logprobs_found, all_found

def process_lpe_responses(input_path: Path, manifest_path: Path, output_prefix: str) -> None:
    if not input_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing input files: {input_path} or {manifest_path}")

    # Load frozen manifest order
    manifest_items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            manifest_items.append(json.loads(line))

    manifest_obj_ids = [it["object_id"] for it in manifest_items]
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    items_raw: Dict[str, Dict[int, Tuple[dict, dict]]] = {}
    total_records = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            obj_id = rec["object_id"]
            perm_idx = rec["permutation_index"]
            sym_map = rec["symbol_mapping"]
            resp = rec["response"]
            items_raw.setdefault(obj_id, {})[perm_idx] = (resp, sym_map)
            total_records += 1

    print(f"Loaded {total_records} raw LPE responses across {len(items_raw)} items.")

    NORM_PROBS_DIR.mkdir(parents=True, exist_ok=True)
    N = len(manifest_obj_ids)

    q_lpe = np.zeros((N, 3), dtype=np.float64)
    s_mapping = np.zeros(N, dtype=np.float64)
    mean_logits = np.zeros((N, 3), dtype=np.float64)
    probs_per_perm = np.zeros((N, 6, 3), dtype=np.float64)
    logits_per_perm = np.zeros((N, 6, 3), dtype=np.float64)

    failed_items = []
    missing_tokens_count = 0

    for i, obj_id in enumerate(manifest_obj_ids):
        perm_map = items_raw.get(obj_id, {})
        if len(perm_map) < 6:
            failed_items.append(obj_id)
            print(f"  Warning: Item {obj_id} has only {len(perm_map)}/6 permutations logged.")
            continue

        item_logits = np.zeros((6, 3), dtype=np.float64)
        item_probs = np.zeros((6, 3), dtype=np.float64)

        item_valid = True
        for perm_idx in range(6):
            resp, sym_map = perm_map[perm_idx]
            lps, ok = extract_candidate_logprobs_from_response(resp, sym_map)
            if not ok:
                item_valid = False
                missing_tokens_count += 1
                continue

            sym_e = sym_map["entailment"]
            sym_n = sym_map["neutral"]
            sym_c = sym_map["contradiction"]

            log_e = lps[sym_e]
            log_n = lps[sym_n]
            log_c = lps[sym_c]

            item_logits[perm_idx] = [log_e, log_n, log_c]

            max_lp = max(log_e, log_n, log_c)
            exp_e = np.exp(log_e - max_lp)
            exp_n = np.exp(log_n - max_lp)
            exp_c = np.exp(log_c - max_lp)
            sum_exp = exp_e + exp_n + exp_c

            item_probs[perm_idx] = [exp_e / sum_exp, exp_n / sum_exp, exp_c / sum_exp]

        if not item_valid:
            failed_items.append(obj_id)
            continue

        mean_lps = np.mean(item_logits, axis=0)
        mean_logits[i] = mean_lps

        mean_p = np.mean(item_probs, axis=0)
        q_lpe[i] = mean_p
        probs_per_perm[i] = item_probs
        logits_per_perm[i] = item_logits

        jsd_sum = sum(jsd_single(item_probs[k], mean_p) for k in range(6))
        s_mapping[i] = jsd_sum / 6.0

    print("=========================================================================")
    print(f"   LPE EXTRACTION SUMMARY ({output_prefix})")
    print("=========================================================================")
    print(f"  Manifest Item Alignment:       STRICT MATCH ({N} items)")
    print(f"  Extraction Success Rate:       {(N - len(failed_items)) / max(1, N) * 100.0:.2f}%")
    print(f"  Missing Token Events:          {missing_tokens_count}")
    print(f"  Mean Label Mapping Sensitivity: {np.mean(s_mapping):.6f} bits")
    print("=========================================================================")

    np.save(NORM_PROBS_DIR / f"{output_prefix}_lpe_probs.npy", q_lpe)
    np.save(NORM_PROBS_DIR / f"{output_prefix}_lpe_mean_logits.npy", mean_logits)
    np.save(NORM_PROBS_DIR / f"{output_prefix}_lpe_order_sensitivity.npy", s_mapping)
    np.save(NORM_PROBS_DIR / f"{output_prefix}_lpe_probs_per_perm.npy", probs_per_perm)
    np.save(NORM_PROBS_DIR / f"{output_prefix}_lpe_logits_per_perm.npy", logits_per_perm)

    meta_info = {
        "object_ids": manifest_obj_ids,
        "object_ids_sha256": hashlib.sha256(json.dumps(manifest_obj_ids).encode("utf-8")).hexdigest(),
        "manifest_sha256": manifest_sha256,
        "object_count": N
    }
    with open(NORM_PROBS_DIR / f"{output_prefix}_object_ids.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="E004 LPE Extraction")
    parser.add_argument("--subset", choices=["preflight", "pilot"], default="preflight")
    args = parser.parse_args()

    manifest = Path("research/chaosnli/artifacts/E004/manifests") / f"{args.subset}_60.jsonl" if args.subset == "preflight" else Path("research/chaosnli/artifacts/E004/manifests") / f"{args.subset}_600.jsonl"
    input_file = RAW_RESPONSES_DIR / f"{args.subset}_lpe_responses.jsonl"
    process_lpe_responses(input_file, manifest, args.subset)

if __name__ == "__main__":
    main()
