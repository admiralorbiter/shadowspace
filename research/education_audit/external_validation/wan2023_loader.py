"""Wan et al. EMNLP 2023 Reference Letter Bias Real Dataset Ingestion & Caching Module."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import urllib.request
from typing import Any, Dict, List

WAN2023_CLG_URL = "https://raw.githubusercontent.com/uclanlp/biases-llm-reference-letters/main/generated_letters/chatgpt/clg/clg_letters.csv"


def download_and_cache_file(url: str, local_path: str) -> str:
    """Downloads a public dataset file, caches it locally, and calculates its SHA-256 hash."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        print(f"Downloading real dataset from {url} ...")
        with urllib.request.urlopen(url) as resp:
            content = resp.read()
        with open(local_path, "wb") as f:
            f.write(content)

    with open(local_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    return file_hash


def load_wan2023_dataset(data_dir: str = "data/external/wan2023") -> Dict[str, Any]:
    """Loads and validates the REAL Wan et al. EMNLP 2023 published ChatGPT reference letter dataset."""
    local_clg_csv = os.path.join(data_dir, "clg_letters.csv")

    try:
        sha256_hash = download_and_cache_file(WAN2023_CLG_URL, local_clg_csv)
    except Exception as e:
        print(f"Warning: Could not fetch online dataset ({e}). Falling back to cached or local copy.")
        if not os.path.exists(local_clg_csv):
            raise FileNotFoundError(f"Local file {local_clg_csv} not found and download failed.")
        with open(local_clg_csv, "rb") as f:
            sha256_hash = hashlib.sha256(f.read()).hexdigest()

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
        "dataset_name": "Wan et al. EMNLP 2023 (Real Published Dataset)",
        "status": "LOADED_REAL_DATA",
        "file_path": local_clg_csv,
        "sha256_hash": sha256_hash,
        "records_count": len(records),
        "sample_records": records[:5],
        "records": records,
    }
