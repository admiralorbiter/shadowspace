"""E004 Cross-Fitted Temperature Calibration Script.

Fits scalar temperature T strictly within the training fold of each of the 5 stratified CV folds
by minimizing soft-label NLL against empirical human distributions.
Preserves raw estimand equality at T=1: q_i(T) = (1/6) sum_pi softmax(logits_i_pi / T).
Applies each fold's T to construct coherent full probability matrices Q^(f) for relational scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import softmax

NORM_PROBS_DIR = Path("research/chaosnli/artifacts/E004/normalized_probs")
MANIFEST_DIR = Path("research/chaosnli/artifacts/E004/manifests")

# Expanded logarithmic temperature grid T in [0.05, 100.0]
TEMPERATURE_GRID = np.logspace(np.log10(0.05), np.log10(100.0), num=500)

def compute_lpe_probs_at_temperature(logits_per_perm: np.ndarray, temp: float) -> np.ndarray:
    """Compute (N, 3) LPE probabilities for a given scalar temperature T across 6 permutations."""
    t_safe = max(1e-4, temp)
    probs_per_perm = softmax(logits_per_perm / t_safe, axis=2)
    return np.mean(probs_per_perm, axis=1)

def soft_label_nll(p_human: np.ndarray, q_model: np.ndarray) -> float:
    """Compute mean soft-label cross-entropy NLL over items."""
    q_safe = np.clip(q_model, 1e-12, 1.0)
    return float(-np.mean(np.sum(p_human * np.log(q_safe), axis=1)))

def find_best_temperature_vectorized(logits_perm: np.ndarray, p_human: np.ndarray, grid: np.ndarray) -> Tuple[float, float]:
    """Fully vectorized NLL evaluation over grid for (M, 6, 3) logit tensors."""
    # logits_perm: (M, 6, 3), grid: (G,)
    logits_expanded = logits_perm[np.newaxis, :, :, :] / grid[:, np.newaxis, np.newaxis, np.newaxis]
    max_l = np.max(logits_expanded, axis=3, keepdims=True)
    exp_l = np.exp(logits_expanded - max_l)
    probs_perm = exp_l / np.sum(exp_l, axis=3, keepdims=True)
    q_grid = np.mean(probs_perm, axis=2)  # (G, M, 3)

    q_safe = np.clip(q_grid, 1e-12, 1.0)
    nll_grid = -np.mean(np.sum(p_human[np.newaxis, :, :] * np.log(q_safe), axis=2), axis=1)

    best_idx = np.argmin(nll_grid)
    return float(grid[best_idx]), float(nll_grid[best_idx])

def build_stratified_folds(items: List[Dict], n_folds: int = 5, seed: int = 20260803) -> List[np.ndarray]:
    """Build 5 stratified fold indices by (source_dataset, majority_label)."""
    p_human = np.array([[it["human_p_entailment"], it["human_p_neutral"], it["human_p_contradiction"]] for it in items])
    datasets = [it["source_dataset"] for it in items]
    majority = np.argmax(p_human, axis=1)

    strata_keys = [f"{d}_{m}" for d, m in zip(datasets, majority)]

    rng = np.random.default_rng(seed)
    strata_map: Dict[str, List[int]] = {}
    for idx, key in enumerate(strata_keys):
        strata_map.setdefault(key, []).append(idx)

    folds = [[] for _ in range(n_folds)]
    for key, indices in sorted(strata_map.items()):
        shuffled = rng.permutation(indices)
        for i, idx in enumerate(shuffled):
            folds[i % n_folds].append(idx)

    fold_arrays = [np.array(sorted(fold), dtype=np.int64) for fold in folds]

    # Assertions for fold integrity
    assert all(len(f) > 0 for f in fold_arrays), "Found empty fold!"
    assert sorted(np.concatenate(fold_arrays).tolist()) == list(range(len(items))), "Fold index mismatch!"

    return fold_arrays

def run_cross_fitted_calibration(subset: str = "pilot") -> None:
    manifest_file = MANIFEST_DIR / f"{subset}_600.jsonl" if subset == "pilot" else MANIFEST_DIR / f"{subset}_60.jsonl"
    logits_per_perm_file = NORM_PROBS_DIR / f"{subset}_lpe_logits_per_perm.npy"

    if not manifest_file.exists() or not logits_per_perm_file.exists():
        raise FileNotFoundError(f"Required inputs missing: {manifest_file} or {logits_per_perm_file}")

    items = []
    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    N = len(items)
    logits_per_perm = np.load(logits_per_perm_file)  # (N, 6, 3)
    p_human = np.array([[it["human_p_entailment"], it["human_p_neutral"], it["human_p_contradiction"]] for it in items])

    # Verification: raw LPE at T=1.0 matches q_lpe_raw
    q_lpe_raw = np.load(NORM_PROBS_DIR / f"{subset}_lpe_probs.npy")
    q_lpe_t1 = compute_lpe_probs_at_temperature(logits_per_perm, 1.0)
    assert np.allclose(q_lpe_raw, q_lpe_t1, atol=1e-10), "q_raw and q(T=1.0) must match identically!"

    folds = build_stratified_folds(items, n_folds=5)

    oof_calibrated_probs = np.zeros_like(p_human)
    fold_full_probs = np.zeros((5, N, 3), dtype=np.float64)
    fold_temperatures = []

    print("=========================================================================")
    print(f"   CROSS-FITTED TEMPERATURE CALIBRATION ({subset.upper()})")
    print("=========================================================================")

    for fold_idx in range(5):
        val_idx = folds[fold_idx]
        train_idx = np.setdiff1d(np.arange(N), val_idx)

        train_logits_perm = logits_per_perm[train_idx]
        train_p_human = p_human[train_idx]

        best_t, best_nll = find_best_temperature_vectorized(train_logits_perm, train_p_human, TEMPERATURE_GRID)

        # Boundary check assertion
        assert min(TEMPERATURE_GRID) < best_t < max(TEMPERATURE_GRID), f"Fitted T={best_t} hit search boundary!"
        fold_temperatures.append(float(best_t))

        # Full coherent probability matrix for fold_idx
        q_full_fold = compute_lpe_probs_at_temperature(logits_per_perm, best_t)
        fold_full_probs[fold_idx] = q_full_fold

        # Held-out focal predictions for pointwise OOF NLL
        oof_calibrated_probs[val_idx] = q_full_fold[val_idx]

        print(f"  Fold {fold_idx + 1}/5 -> Fitted T = {best_t:.4f} (Train NLL: {best_nll:.5f})")

    raw_nll = soft_label_nll(p_human, q_lpe_raw)
    calib_nll = soft_label_nll(p_human, oof_calibrated_probs)

    print(f"  Raw LPE NLL:         {raw_nll:.5f} nats")
    print(f"  Calibrated LPE NLL:  {calib_nll:.5f} nats (Delta NLL: {raw_nll - calib_nll:.5f})")
    print("=========================================================================")

    np.save(NORM_PROBS_DIR / f"{subset}_lpe_calibrated_probs.npy", oof_calibrated_probs)
    np.save(NORM_PROBS_DIR / f"{subset}_lpe_fold_calibrated_probs.npy", fold_full_probs)

    # Save fold composition stats
    fold_stats = []
    for f_i, f_idx in enumerate(folds):
        f_items = [items[i] for i in f_idx]
        fold_stats.append({
            "fold_index": f_i,
            "count": len(f_items),
            "datasets": pd.Series([it["source_dataset"] for it in f_items]).value_counts().to_dict(),
            "majority_labels": pd.Series([np.argmax([it["human_p_entailment"], it["human_p_neutral"], it["human_p_contradiction"]]) for it in f_items]).value_counts().to_dict()
        })

    with open(NORM_PROBS_DIR / f"{subset}_lpe_fold_temperatures.json", "w", encoding="utf-8") as f:
        json.dump({
            "fold_temperatures": fold_temperatures,
            "mean_temperature": float(np.mean(fold_temperatures)),
            "fold_statistics": fold_stats
        }, f, indent=2)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["preflight", "pilot"], default="pilot")
    args = parser.parse_args()
    run_cross_fitted_calibration(args.subset)

if __name__ == "__main__":
    main()
