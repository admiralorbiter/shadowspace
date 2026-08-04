"""E007 Ultra-Fast 5-Fold Cross-Fitted Coalition Selection Engine.

Performs genuine 5-fold cross-validated coalition selection across all 511 candidate ensembles.
Operates directly on Bhattacharyya Affinity matrices for max speed.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

JSON_PATH = Path("data/chaosnli/processed/canonical_items_posterior.json")
E001_BIN = Path("research/chaosnli/artifacts/E001/S_hellinger_k010.bin")
MODEL_PROBS_PATH = Path("research/chaosnli/rust_manifest/model_probs.json")
RESULTS_DIR = Path("research/chaosnli/results")

def compute_q_support_bc(sqrt_P: np.ndarray, s_target: np.ndarray, k: int = 10) -> float:
    N = sqrt_P.shape[0]
    bc = np.dot(sqrt_P, sqrt_P.T)
    np.fill_diagonal(bc, -1.0)

    # Top-k largest BC is equivalent to top-k smallest Hellinger distance
    k_bcs = np.partition(bc, N - k, axis=1)[:, N - k, np.newaxis]
    ATOL = 1e-7

    closer_mask = bc > (k_bcs + ATOL)
    tied_mask = np.abs(bc - k_bcs) <= ATOL

    n_closer = np.sum(closer_mask, axis=1, keepdims=True)
    n_tied = np.sum(tied_mask, axis=1, keepdims=True)
    frac = np.where(n_tied > 0, (k - n_closer) / np.maximum(1.0, n_tied.astype(np.float32)), 0.0)

    W = np.where(closer_mask, 1.0, np.where(tied_mask, frac, 0.0))
    np.fill_diagonal(W, 0.0)

    return float(np.sum(W * s_target) / (N * 10.0))

def compute_soft_label_nll(p_human: np.ndarray, q_model: np.ndarray) -> float:
    q_safe = np.clip(q_model, 1e-12, 1.0)
    return float(-np.mean(np.sum(p_human * np.log(q_safe), axis=1)))

def main():
    print("=========================================================================")
    print("   E007: ULTRA-FAST GENUINE 5-FOLD CROSS-FITTED COALITION SELECTION")
    print("=========================================================================")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    n_items = len(items)
    is_snli = np.array([it["source_dataset"].lower().find("snli") >= 0 for it in items])
    p_human = np.array([[it["human_p_entailment"], it["human_p_neutral"], it["human_p_contradiction"]] for it in items])

    # Assign 5-fold stratified IDs
    n_folds = 5
    rng_fold = np.random.default_rng(2026_0803)
    item_fold_ids = np.zeros(n_items, dtype=int)

    maj_labels = np.argmax(p_human, axis=1)
    ent = -np.sum(np.where(p_human > 1e-12, p_human * np.log2(np.clip(p_human, 1e-12, 1.0)), 0.0), axis=1)
    entropy_qs = np.digitize(ent, np.quantile(ent, [0.2, 0.4, 0.6, 0.8]))

    strata_map = {}
    for i in range(n_items):
        d_str = "snli" if is_snli[i] else "mnli"
        s_key = f"{d_str}_{maj_labels[i]}_{entropy_qs[i]}"
        strata_map.setdefault(s_key, []).append(i)

    for s_key, idxs in strata_map.items():
        shuffled = rng_fold.permutation(idxs)
        for rank, idx in enumerate(shuffled):
            item_fold_ids[idx] = rank % n_folds

    # Load frozen target matrix S and model probabilities
    s_bytes = E001_BIN.read_bytes()
    s_target = np.frombuffer(s_bytes, dtype=np.float32).reshape((n_items, n_items))

    with open(MODEL_PROBS_PATH, "r", encoding="utf-8") as f:
        model_probs_raw = json.load(f)

    canonical_models = [
        "bart-large", "roberta-large", "xlnet-large", "albert-xxlarge",
        "bert-large", "roberta-base", "xlnet-base", "distilbert", "bert-base"
    ]

    m_num = len(canonical_models)
    model_mats = [np.array(model_probs_raw[m], dtype=np.float32) for m in canonical_models]
    total_subsets = (1 << m_num) - 1

    fold_records = []
    selected_masks_counts = {sz: {} for sz in range(1, m_num + 1)}

    for fold in range(n_folds):
        print(f"Executing Fold {fold+1}/{n_folds}...", flush=True)
        val_indices = np.where(item_fold_ids == fold)[0]
        train_indices = np.where(item_fold_ids != fold)[0]

        n_train = len(train_indices)
        n_val = len(val_indices)

        s_tr = s_target[np.ix_(train_indices, train_indices)]
        s_val = s_target[np.ix_(val_indices, val_indices)]

        best_train_coalitions = {}

        for mask in range(1, total_subsets + 1):
            active_indices = [i for i in range(m_num) if (mask & (1 << i)) != 0]
            sz = len(active_indices)

            q_train = np.mean([model_mats[i][train_indices] for i in active_indices], axis=0)
            sqrt_P_tr = np.sqrt(np.clip(q_train, 1e-12, 1.0, dtype=np.float32))
            q_supp_tr = compute_q_support_bc(sqrt_P_tr, s_tr, k=10)

            if sz not in best_train_coalitions or q_supp_tr > best_train_coalitions[sz]["train_q"]:
                best_train_coalitions[sz] = {
                    "mask": mask,
                    "active_indices": active_indices,
                    "train_q": q_supp_tr,
                }

        # Evaluate selected winning coalitions on held-out validation fold
        for sz in range(1, m_num + 1):
            win_mask = best_train_coalitions[sz]["mask"]
            win_indices = best_train_coalitions[sz]["active_indices"]
            win_models = [canonical_models[i] for i in win_indices]

            selected_masks_counts[sz].setdefault(win_mask, 0)
            selected_masks_counts[sz][win_mask] += 1

            q_val = np.mean([model_mats[i][val_indices] for i in win_indices], axis=0)
            val_nll = compute_soft_label_nll(p_human[val_indices], q_val)

            sqrt_P_val = np.sqrt(np.clip(q_val, 1e-12, 1.0, dtype=np.float32))
            q_supp_val = compute_q_support_bc(sqrt_P_val, s_val, k=10)

            w_null_val = 10.0 / (n_val - 1.0)
            q_null_val = float(np.sum(s_val)) * (w_null_val / (n_val * 10.0))
            r_norm_val = (q_supp_val - q_null_val) / (0.038987 - q_null_val)

            fold_records.append({
                "fold": fold,
                "coalition_size": sz,
                "selected_mask": win_mask,
                "selected_models": win_models,
                "train_q_support": float(best_train_coalitions[sz]["train_q"]),
                "held_out_r_normalized": float(r_norm_val),
                "held_out_nll": float(val_nll),
                "n_train": n_train,
                "n_held_out": n_val,
            })

    # Summary by size
    summary_by_size = []
    for sz in range(1, m_num + 1):
        f_sz = [r for r in fold_records if r["coalition_size"] == sz]
        mean_r = float(np.mean([r["held_out_r_normalized"] for r in f_sz]))
        mean_nll = float(np.mean([r["held_out_nll"] for r in f_sz]))

        top_mask = max(selected_masks_counts[sz].items(), key=lambda x: x[1])[0]
        top_freq = float(selected_masks_counts[sz][top_mask] / float(n_folds))
        top_models = [canonical_models[i] for i in range(m_num) if (top_mask & (1 << i)) != 0]

        summary_by_size.append({
            "coalition_size": sz,
            "selected_models": top_models,
            "held_out_r_normalized_mean": mean_r,
            "held_out_nll_mean": mean_nll,
            "top_mask_selection_frequency": top_freq,
        })

    crossfit_data = {
        "n_folds": n_folds,
        "method": "genuine_5fold_stratified_train_selection_heldout_evaluation",
        "held_out_summary_by_size": summary_by_size,
        "fold_details": fold_records,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "E007_held_out_selection.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(crossfit_data, f, indent=2)

    print("\nHeld-Out Cross-Fitted Coalition Performance by Size:")
    for s in summary_by_size:
        print(f"  Size {s['coalition_size']}: Held-Out R_norm = {s['held_out_r_normalized_mean']*100.0:>6.2f}% | Held-Out NLL = {s['held_out_nll_mean']:.4f} | Selection Freq = {s['top_mask_selection_frequency']*100.0:.0f}% | Models: {s['selected_models']}")

    print(f"\nSaved GENUINE E007 cross-fitted selection results to {out_file}")

if __name__ == "__main__":
    main()
