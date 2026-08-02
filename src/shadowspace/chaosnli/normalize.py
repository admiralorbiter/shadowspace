"""Normalization module to parse raw ChaosNLI JSONL records into canonical schema tables."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import polars as pl

CANONICAL_LABELS = ["entailment", "neutral", "contradiction"]
LABEL_MAP = {
    "entailment": "entailment",
    "e": "entailment",
    "neutral": "neutral",
    "n": "neutral",
    "contradiction": "contradiction",
    "c": "contradiction",
}


def compute_entropy_bits(p_e: float, p_n: float, p_c: float) -> float:
    """Compute Shannon entropy in bits for a 3-class distribution."""
    h = 0.0
    for p in (p_e, p_n, p_c):
        if p > 0.0:
            h -= p * math.log2(p)
    return float(h)


def normalize_record(raw: dict[str, Any], dataset_name: str) -> dict[str, Any] | None:
    """Normalize a single raw ChaosNLI JSONL dictionary into canonical schema fields."""
    # Pair ID extraction
    pair_id = str(raw.get("uid") or raw.get("pairID") or raw.get("id") or "")
    if not pair_id:
        return None

    object_id = f"{dataset_name}_{pair_id}"
    example_obj = raw.get("example", {}) if isinstance(raw.get("example"), dict) else {}
    premise = str(raw.get("premise") or example_obj.get("premise", "")).strip()
    hypothesis = str(raw.get("hypothesis") or example_obj.get("hypothesis", "")).strip()
    genre = str(raw.get("genre") or example_obj.get("genre", "")).strip() if (raw.get("genre") or example_obj.get("genre")) else None

    # Original gold label
    gold_raw = str(raw.get("old_label") or raw.get("gold_label") or raw.get("label") or "").lower()
    original_gold_label = LABEL_MAP.get(gold_raw, gold_raw)

    # Process 100 human judgments
    counts = {"entailment": 0, "neutral": 0, "contradiction": 0}
    raw_labels_list: list[str] = raw.get("labels", [])

    if raw_labels_list:
        for lbl in raw_labels_list:
            norm_lbl = LABEL_MAP.get(str(lbl).lower())
            if norm_lbl in counts:
                counts[norm_lbl] += 1
    elif "label_counter" in raw:
        lc = raw["label_counter"]
        for k, v in lc.items():
            norm_k = LABEL_MAP.get(str(k).lower())
            if norm_k in counts:
                counts[norm_k] += int(v)

    c_e = counts["entailment"]
    c_n = counts["neutral"]
    c_c = counts["contradiction"]
    total_count = c_e + c_n + c_c

    # Validate total count = 100 (or close if standard ChaosNLI)
    if total_count == 0:
        return None

    p_e = c_e / total_count
    p_n = c_n / total_count
    p_c = c_c / total_count

    entropy_bits = compute_entropy_bits(p_e, p_n, p_c)

    # Majority label logic
    majority_idx = max(range(3), key=lambda i: [c_e, c_n, c_c][i])
    majority_label = CANONICAL_LABELS[majority_idx]
    majority_count = max(c_e, c_n, c_c)
    agreement_rate = majority_count / total_count
    has_zero_count = (c_e == 0 or c_n == 0 or c_c == 0)

    return {
        "object_id": object_id,
        "source_dataset": dataset_name,
        "source_pair_id": pair_id,
        "premise": premise,
        "hypothesis": hypothesis,
        "genre": genre,
        "original_gold_label": original_gold_label,
        "original_labels_json": json.dumps(raw_labels_list),
        "human_count_entailment": c_e,
        "human_count_neutral": c_n,
        "human_count_contradiction": c_c,
        "human_p_entailment": p_e,
        "human_p_neutral": p_n,
        "human_p_contradiction": p_c,
        "human_entropy_bits": entropy_bits,
        "human_majority_label": majority_label,
        "human_majority_count": majority_count,
        "human_agreement_rate": agreement_rate,
        "has_zero_count": has_zero_count,
        "split": "unassigned",
    }


def normalize_dataset(
    raw_dir: Path = Path("data/chaosnli/raw"),
    output_dir: Path = Path("data/chaosnli/processed"),
    datasets: tuple[str, ...] = ("snli", "mnli"),
) -> dict[str, Any]:
    """Normalize raw ChaosNLI JSONL files into canonical Parquet tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_normalized: list[dict[str, Any]] = []

    file_mapping = {
        "snli": ("chaosNLI_snli.jsonl", "chaosnli_snli"),
        "mnli": ("chaosNLI_mnli_m.jsonl", "chaosnli_mnli"),
    }

    for key in datasets:
        if key not in file_mapping:
            continue
        filename, ds_name = file_mapping[key]
        filepath = raw_dir / filename
        if not filepath.exists():
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw_rec = json.loads(line)
                norm_rec = normalize_record(raw_rec, ds_name)
                if norm_rec is not None:
                    all_normalized.append(norm_rec)

    if not all_normalized:
        raise ValueError("No records normalized. Ensure raw source files exist.")

    df = pl.DataFrame(all_normalized)

    # Save to Parquet
    out_parquet = output_dir / "canonical_items.parquet"
    df.write_parquet(out_parquet)

    # Summary statistics
    total_items = len(df)
    zero_count_items = df.filter(pl.col("has_zero_count")).height
    mean_entropy = float(df["human_entropy_bits"].mean())

    summary = {
        "output_path": str(out_parquet),
        "total_items": total_items,
        "zero_count_items": zero_count_items,
        "zero_count_prevalence": zero_count_items / total_items if total_items > 0 else 0.0,
        "mean_entropy_bits": mean_entropy,
        "majority_counts": df["human_majority_label"].value_counts().to_dicts(),
    }

    return summary
