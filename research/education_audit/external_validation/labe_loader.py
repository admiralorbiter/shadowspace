"""LABE / Language Agency Classifier Commit-Pinned Ingestion Module with Fail-Closed Hash Verification."""

from __future__ import annotations

import csv
import hashlib
import os
import urllib.request
from typing import Any, Dict, List

LABE_COMMIT_SHA = "e8cc42d86df007fd05e3ae0c27c127b7a0a6165c"
BASE_URL = f"https://raw.githubusercontent.com/elainew728/labe-agency/{LABE_COMMIT_SHA}/lac_dataset_construction/lac_dataset"

SPLIT_URLS = {
    "train": f"{BASE_URL}/train.csv",
    "validation": f"{BASE_URL}/validation.csv",
    "test": f"{BASE_URL}/test.csv",
}

EXPECTED_LABE_HASHES = {
    "train": "abcc3ec6032e3b265cbf15c6d8a3da668a2a030675b00f0425b96698c8cd5b56",
    "validation": "b72ea51c5f723db24c25013ca22c9155eb4a3a0622ebcffb706436bbb11e8f5e",
    "test": "3fd3d732c8c41a1479c69c73714814181c2224a4fcf593ca500f152e29d287e6",
}




def download_and_verify_file(url: str, local_path: str, expected_sha256: str = None) -> str:
    """Downloads dataset split, verifies SHA-256 against expected registry value, and FAILS CLOSED on mismatch."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        print(f"Downloading commit-pinned LABE split from {url} ...")
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


def load_labe_dataset(data_dir: str = "data/external/labe") -> Dict[str, Any]:
    """Loads and validates the REAL LABE 2023 LAC dataset across Train, Validation, and Test splits (Fails Closed)."""
    sentences_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    split_hashes: Dict[str, str] = {}

    for split_name, url in SPLIT_URLS.items():
        local_csv = os.path.join(data_dir, f"{split_name}.csv")
        expected_h = EXPECTED_LABE_HASHES.get(split_name)

        split_hash = download_and_verify_file(url, local_csv, expected_sha256=expected_h)
        split_hashes[split_name] = split_hash

        if not os.path.exists(local_csv):
            raise FileNotFoundError(f"Required LABE split file {local_csv} missing.")

        with open(local_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                lbl = int(row.get("label", 0))
                sentences_by_split[split_name].append({
                    "sentence_id": f"LAC_{split_name}_{idx:04d}",
                    "text": row.get("text", ""),
                    "label_int": lbl,
                    "label_str": "agentic" if lbl == 1 else "communal_or_passive",
                    "split": split_name,
                })

    all_sentences = sentences_by_split["train"] + sentences_by_split["validation"] + sentences_by_split["test"]

    return {
        "dataset_name": "LABE 2023 Language Agency Classifier (Commit-Pinned Real Dataset)",
        "status": "LOADED_PINNED_REAL_DATA",
        "commit_sha": LABE_COMMIT_SHA,
        "split_hashes": split_hashes,
        "sentences_by_split": sentences_by_split,
        "train_count": len(sentences_by_split["train"]),
        "validation_count": len(sentences_by_split["validation"]),
        "test_count": len(sentences_by_split["test"]),
        "total_labeled_sentences_count": len(all_sentences),
        "all_sentences": all_sentences,
    }
