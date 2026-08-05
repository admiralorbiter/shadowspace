"""Wan et al. EMNLP 2023 Reference Letter Bias Commit-Pinned Dataset Ingestion Module."""

from __future__ import annotations

import csv
import hashlib
import os
import urllib.request
from typing import Any, Dict, List

WAN2023_COMMIT_SHA = "1264990e5f55e46cb8b83d8bfe2749946008b4a8"
WAN2023_PINNED_CLG_URL = f"https://raw.githubusercontent.com/uclanlp/biases-llm-reference-letters/{WAN2023_COMMIT_SHA}/generated_letters/chatgpt/clg/clg_letters.csv"
EXPECTED_CLG_SHA256 = "c3ccc244b85a2e9ef9e671970a4f5cc41fc698b51770daa00d2a16df969f58be"


def download_and_verify_file(url: str, local_path: str, expected_sha256: str = None) -> str:
    """Downloads dataset file, verifies SHA-256 hash, and FAILS CLOSED by raising ValueError on mismatch."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        print(f"Downloading commit-pinned dataset from {url} ...")
        with urllib.request.urlopen(url) as resp:
            content = resp.read()
        with open(local_path, "wb") as f:
            f.write(content)

    with open(local_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    if expected_sha256 and file_hash != expected_sha256:
        raise ValueError(
            f"Dataset hash mismatch for {local_path}: expected {expected_sha256}, got {file_hash}"
        )

    return file_hash


def load_wan2023_dataset(data_dir: str = "data/external/wan2023") -> Dict[str, Any]:
    """Loads and validates the commit-pinned Wan et al. EMNLP 2023 published ChatGPT dataset (Fails Closed)."""
    local_clg_csv = os.path.join(data_dir, "clg_letters.csv")

    sha256_hash = download_and_verify_file(WAN2023_PINNED_CLG_URL, local_clg_csv, EXPECTED_CLG_SHA256)

    records: List[Dict[str, Any]] = []
    with open(local_clg_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "name": row.get("name", ""),
                "age": row.get("age", ""),
                "gender": "masculine" if row.get("gender", "").lower() in ["male", "m"] else "feminine",
                "occupation": row.get("occupation", ""),
                "prompt_text": row.get("prompts", ""),
                "generated_text": row.get("chatgpt_gen", "").replace("<return>", "\n"),
            })

    return {
        "dataset_name": "Wan et al. EMNLP 2023 (Commit-Pinned Real Dataset)",
        "status": "LOADED_PINNED_REAL_DATA",
        "commit_sha": WAN2023_COMMIT_SHA,
        "file_path": local_clg_csv,
        "sha256_hash": sha256_hash,
        "records_count": len(records),
        "sample_records": records[:5],
        "records": records,
    }
