"""E004 Sample Design and Manifest Generator.

Performs 30-stratum sampling (2 datasets x 3 majority labels x 5 entropy quintiles),
selects preflight (60), pilot (600), MCE convergence (60), and temperature sensitivity (120) subsets.
Saves frozen manifest files under research/chaosnli/artifacts/E004/manifests/.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl

CANONICAL_PARQUET = Path("data/chaosnli/processed/canonical_items_posterior.parquet")
ARTIFACT_DIR = Path("research/chaosnli/artifacts/E004")
MANIFEST_DIR = ARTIFACT_DIR / "manifests"

def build_strata_mapping(df: pl.DataFrame) -> Dict[str, List[Dict]]:
    """Build 30 strata mapping from canonical dataframe."""
    p_human = df.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    
    # Entropy calculation
    entropy = -np.sum(np.where(p_human > 1e-12, p_human * np.log2(np.clip(p_human, 1e-12, 1.0)), 0.0), axis=1)
    
    datasets = df["source_dataset"].to_list()
    majority = np.argmax(p_human, axis=1)
    entropy_q = pd.qcut(entropy, q=5, labels=False, duplicates="drop")
    
    records = df.to_dicts()
    strata: Dict[str, List[Dict]] = {}
    for idx, (rec, d, m, eq) in enumerate(zip(records, datasets, majority, entropy_q)):
        key = f"{d}_{m}_{eq}"
        rec["row_index"] = idx
        rec["stratum_key"] = key
        strata.setdefault(key, []).append(rec)
        
    return strata

def create_sample_manifests(seed: int = 20260804) -> None:
    if not CANONICAL_PARQUET.exists():
        raise FileNotFoundError(f"Missing canonical data file: {CANONICAL_PARQUET}")
        
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(CANONICAL_PARQUET)
    strata = build_strata_mapping(df)
    
    rng = np.random.default_rng(seed)
    
    # 1. Preflight 60 (2 per stratum)
    preflight_items = []
    for key, items in sorted(strata.items()):
        shuffled = rng.permutation(len(items))
        selected_idx = shuffled[:min(2, len(items))]
        for i in selected_idx:
            preflight_items.append(items[i])
            
    # 2. Pilot 600 (target 20 per stratum with proportional redistribution for small strata)
    stratum_counts = {k: len(v) for k, v in strata.items()}
    stratum_targets = {}
    remaining_budget = 600
    
    # First pass: assign up to 20 or actual count
    for k, count in stratum_counts.items():
        alloc = min(20, count)
        stratum_targets[k] = alloc
        remaining_budget -= alloc
        
    # Second pass: distribute remaining budget proportionally among strata that have surplus items
    while remaining_budget > 0:
        eligible_strata = [k for k in stratum_counts if stratum_counts[k] > stratum_targets[k]]
        if not eligible_strata:
            break
        for k in eligible_strata:
            if remaining_budget == 0:
                break
            stratum_targets[k] += 1
            remaining_budget -= 1

    pilot_items = []
    for key, items in sorted(strata.items()):
        target = stratum_targets[key]
        shuffled = rng.permutation(len(items))
        selected_idx = shuffled[:target]
        for i in selected_idx:
            pilot_items.append(items[i])
            
    # 3. MCE convergence 60 (subset of pilot items, 2 per stratum where available)
    rng_conv = np.random.default_rng(seed + 1)
    pilot_by_stratum: Dict[str, List[Dict]] = {}
    for item in pilot_items:
        pilot_by_stratum.setdefault(item["stratum_key"], []).append(item)
        
    convergence_items = []
    for key, items in sorted(pilot_by_stratum.items()):
        shuffled = rng_conv.permutation(len(items))
        selected_idx = shuffled[:min(2, len(items))]
        for i in selected_idx:
            convergence_items.append(items[i])

    # 4. Temperature sensitivity 120 (subset of pilot items, 4 per stratum where available)
    rng_temp = np.random.default_rng(seed + 2)
    temp_sensitivity_items = []
    for key, items in sorted(pilot_by_stratum.items()):
        shuffled = rng_temp.permutation(len(items))
        selected_idx = shuffled[:min(4, len(items))]
        for i in selected_idx:
            temp_sensitivity_items.append(items[i])

    # Write manifests
    manifest_specs = [
        ("preflight_60.jsonl", preflight_items),
        ("pilot_600.jsonl", pilot_items),
        ("mce_convergence_60.jsonl", convergence_items),
        ("temp_sensitivity_120.jsonl", temp_sensitivity_items),
    ]

    print("=========================================================================")
    print("   EXPERIMENT E004 — MANIFEST GENERATION SUMMARY")
    print("=========================================================================")
    for filename, items in manifest_specs:
        out_path = MANIFEST_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        
        sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
        print(f"  {filename:<26} -> {len(items):>4} items | SHA-256: {sha256[:16]}...")
        
    print("=========================================================================")

if __name__ == "__main__":
    create_sample_manifests()
