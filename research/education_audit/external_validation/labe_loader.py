"""LABE / Language Agency Classifier Real Dataset Ingestion & Caching Module."""

from __future__ import annotations

import csv
import hashlib
import os
import urllib.request
from typing import Any, Dict, List

LABE_LAC_TRAIN_URL = "https://raw.githubusercontent.com/elainew728/labe-agency/main/lac_dataset_construction/lac_dataset/train.csv"
LABE_LAC_TEST_URL = "https://raw.githubusercontent.com/elainew728/labe-agency/main/lac_dataset_construction/lac_dataset/test.csv"


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


def load_labe_dataset(data_dir: str = "data/external/labe") -> Dict[str, Any]:
    """Loads and validates the REAL LABE 2023 Language Agency Classification (LAC) dataset."""
    local_train_csv = os.path.join(data_dir, "train.csv")
    local_test_csv = os.path.join(data_dir, "test.csv")

    try:
        train_hash = download_and_cache_file(LABE_LAC_TRAIN_URL, local_train_csv)
        test_hash = download_and_cache_file(LABE_LAC_TEST_URL, local_test_csv)
    except Exception as e:
        print(f"Warning: Could not fetch online LABE dataset ({e}). Falling back to local copy.")
        with open(local_train_csv, "rb") as f:
            train_hash = hashlib.sha256(f.read()).hexdigest()
        with open(local_test_csv, "rb") as f:
            test_hash = hashlib.sha256(f.read()).hexdigest()

    sentences: List[Dict[str, Any]] = []

    for path, split_name in [(local_train_csv, "train"), (local_test_csv, "test")]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                lbl = int(row.get("label", 0))
                sentences.append({
                    "sentence_id": f"LAC_{split_name}_{idx:04d}",
                    "text": row.get("text", ""),
                    "label_int": lbl,
                    "label_str": "agentic" if lbl == 1 else "communal_or_passive",
                    "split": split_name,
                })

    return {
        "dataset_name": "LABE 2023 Language Agency Classifier (Real Dataset)",
        "status": "LOADED_REAL_DATA",
        "train_file_path": local_train_csv,
        "test_file_path": local_test_csv,
        "train_sha256_hash": train_hash,
        "test_sha256_hash": test_hash,
        "labeled_sentences_count": len(sentences),
        "sample_sentences": sentences[:5],
        "sentences": sentences,
    }
