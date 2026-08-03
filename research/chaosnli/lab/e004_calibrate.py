"""E004 Cross-Fitted Temperature Calibration Script.

Fits scalar temperature T strictly within the training fold of each of the 5 stratified CV folds
by minimizing soft-label NLL against empirical human distributions.
Applies each fold's T to construct coherent held-out predictions for pilot items.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.special import softmax

NORM_PROBS_DIR = Path("research/chaosnli/artifacts/E004/normalized_probs")
MANIFEST_DIR = Path("research/chaosnli/artifacts/E004/manifests")

TEMPERATURE_GRID = np.array([
    0.10, 0.125, 0.16, 0.20, 0.25,
    0.32, 0.40, 0.50, 0.63, 0.80,
    1.00,
    1.25, 1.60, 2.00, 2.50,
    3.20, 4.00, 5.00, 6.30, 8.00, 10.00
])

def soft_label_nll(p_human: np.ndarray, q_model: np.ndarray) -> float:
    """Compute mean soft-label cross-entropy NLL over items."""
    q_safe = np.clip(q_model, 1e-12, 1.0)
    return float(-np.mean(np.sum(p_human * np.log(q_safe), axis=1)))

def build_stratified_folds(items: List[Dict], n_folds: int = 5, seed: int = 20260803) -> List[np.ndarray]:
    """Build 5 stratified fold indices by (source_dataset, majority_label, entropy_quintile)."""
    p_human = np.array([[it["human_p_entailment"], it["human_p_neutral"], it["human_p_contradiction"]] for it in items])
    entropy = -np.sum(np.where(p_human > 1e-12, p_human * np.log2(np.clip(p_human, 1e-12, 1.0)), 0.0), axis=1)

    datasets = [it["source_dataset"] for it in items]
    majority = np.argmax(p_human, axis=1)
    entropy_q = pd.qcut(entropy, q=5, labels=False, duplicates="drop")

    strata_keys = [f"{d}_{m}_{eq}" for d, m, eq in zip(datasets, majority, entropy_q)]

    rng = np.random.default_rng(seed)
    strata_map: Dict[str, List[int]] = {}
    for idx, key in enumerate(strata_keys):
        strata_map.setdefault(key, []).append(idx)

    folds = [[] for _ in range(n_folds)]
    for key, indices in strata_map.items():
        shuffled = rng.permutation(indices)
        for i, idx in enumerate(shuffled):
            folds[i % n_folds].append(idx)

    return [np.array(sorted(fold), dtype=np.int64) for fold in folds]

def run_cross_fitted_calibration(subset: str = "pilot") -> None:
    manifest_file = MANIFEST_DIR / f"{subset}_600.jsonl" if subset == "pilot" else MANIFEST_DIR / f"{subset}_60.jsonl"
    mean_logits_file = NORM_PROBS_DIR / f"{subset}_lpe_mean_logits.npy"

    if not manifest_file.exists() or not mean_logits_file.exists():
        raise FileNotFoundError(f"Required inputs missing: {manifest_file} or {mean_logits_file}")

    items = []
    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    mean_logits = np.load(mean_logits_file)
    p_human = np.array([[it["human_p_entailment"], it["human_p_neutral"], it["human_p_contradiction"]] for it in items])

    folds = build_stratified_folds(items, n_folds=5)

    oof_calibrated_probs = np.zeros_like(p_human)
    fold_temperatures = []

    print("=========================================================================")
    print(f"   CROSS-FITTED TEMPERATURE CALIBRATION ({subset.upper()})")
    print("=========================================================================")

    for fold_idx in range(5):
        val_idx = folds[fold_idx]
        train_idx = np.setdiff1d(np.arange(len(items)), val_idx)

        train_logits = mean_logits[train_idx]
        train_p_human = p_human[train_idx]

        best_t = 1.0
        best_nll = float("inf")

        for t_cand in TEMPERATURE_GRID:
            q_train = softmax(train_logits / t_cand, axis=1)
            nll_val = soft_label_nll(train_p_human, q_train)
            if nll_val < best_nll:
                best_nll = nll_val
                best_t = t_cand

        fold_temperatures.append(best_t)

        val_logits = mean_logits[val_idx]
        oof_calibrated_probs[val_idx] = softmax(val_logits / best_t, axis=1)

        print(f"  Fold {fold_idx + 1}/5 -> Fitted T = {best_t:.4f} (Train NLL: {best_nll:.5f})")

    raw_nll = soft_label_nll(p_human, softmax(mean_logits, axis=1))
    calib_nll = soft_label_nll(p_human, oof_calibrated_probs)

    print(f"  Raw LPE NLL:         {raw_nll:.5f} nats")
    print(f"  Calibrated LPE NLL:  {calib_nll:.5f} nats (Delta NLL: {raw_nll - calib_nll:.5f})")
    print("=========================================================================")

    np.save(NORM_PROBS_DIR / f"{subset}_lpe_calibrated_probs.npy", oof_calibrated_probs)
    with open(NORM_PROBS_DIR / f"{subset}_lpe_fold_temperatures.json", "w", encoding="utf-8") as f:
        json.dump({"fold_temperatures": fold_temperatures, "mean_temperature": float(np.mean(fold_temperatures))}, f, indent=2)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["preflight", "pilot"], default="pilot")
    args = parser.parse_args()
    run_cross_fitted_calibration(args.subset)

if __name__ == "__main__":
    main()
