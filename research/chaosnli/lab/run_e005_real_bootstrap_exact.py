"""E005 Real Focal-Row 30-Stratum Paired Item Bootstrap Engine (Exact Rust Null Alignment).

Performs 1,000 genuine common 30-stratum focal-row item resamples using the exact
Rust N0 and N4 conditional null excess values for all 3,113 items.
No synthetic noise or Gaussian approximations are used.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

JSON_PATH = Path("data/chaosnli/processed/canonical_items_posterior.json")
E001_BIN = Path("research/chaosnli/artifacts/E001/S_hellinger_k010.bin")
MODEL_PROBS_PATH = Path("research/chaosnli/rust_manifest/model_probs.json")
E005_SUMMARY_PATH = Path("research/chaosnli/artifacts/E005/summaries/E005_summary.json")
RESULTS_DIR = Path("research/chaosnli/results")

def distance_hellinger_matrix(P: np.ndarray) -> np.ndarray:
    sqrt_P = np.sqrt(np.clip(P, 1e-12, 1.0))
    bc = np.dot(sqrt_P, sqrt_P.T)
    bc = np.clip(bc, 0.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - bc))

def compute_topk_weight_matrix(dist: np.ndarray, k: int = 10) -> np.ndarray:
    N = dist.shape[0]
    ATOL = 1e-7
    dist_self = dist.copy()
    np.fill_diagonal(dist_self, np.inf)

    k_dists = np.partition(dist_self, k - 1, axis=1)[:, k - 1, np.newaxis]
    closer_mask = dist_self < (k_dists - ATOL)
    tied_mask = np.abs(dist_self - k_dists) <= ATOL

    n_closer = np.sum(closer_mask, axis=1, keepdims=True)
    n_tied = np.sum(tied_mask, axis=1, keepdims=True)
    frac = np.where(n_tied > 0, (k - n_closer) / np.maximum(1.0, n_tied.astype(float)), 0.0)

    W = np.where(closer_mask, 1.0, np.where(tied_mask, frac, 0.0))
    np.fill_diagonal(W, 0.0)
    return W

def compute_entropy_quintiles(p_human: np.ndarray) -> np.ndarray:
    ent = -np.sum(np.where(p_human > 1e-12, p_human * np.log2(np.clip(p_human, 1e-12, 1.0)), 0.0), axis=1)
    q = np.quantile(ent, [0.2, 0.4, 0.6, 0.8])
    return np.digitize(ent, q)

def main():
    print("=========================================================================")
    print("   E005: REAL FOCAL-ROW ITEM BOOTSTRAP (EXACT RUST NULL ALIGNMENT)")
    print("=========================================================================")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    n_items = len(items)
    is_snli = np.array([it["source_dataset"].lower().find("snli") >= 0 for it in items])
    p_human = np.array([[it["human_p_entailment"], it["human_p_neutral"], it["human_p_contradiction"]] for it in items])

    maj_labels = np.argmax(p_human, axis=1)
    entropy_qs = compute_entropy_quintiles(p_human)

    strata_map = {}
    for i in range(n_items):
        d_str = "snli" if is_snli[i] else "mnli"
        s_key = f"{d_str}_{maj_labels[i]}_{entropy_qs[i]}"
        strata_map.setdefault(s_key, []).append(i)

    print(f"Created {len(strata_map)} sampling strata for common item resampling.")

    # Load frozen target matrix S
    s_bytes = E001_BIN.read_bytes()
    s_target = np.frombuffer(s_bytes, dtype=np.float32).reshape((n_items, n_items)).astype(np.float64)

    # Load Rust summary null values
    with open(E005_SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    cond_data = summary_data["conditions"]

    # Load model probability matrices
    with open(MODEL_PROBS_PATH, "r", encoding="utf-8") as f:
        model_probs_raw = json.load(f)

    model_names = [
        "model_roberta-large", "model_roberta-base",
        "model_xlnet-large", "model_xlnet-base",
        "model_bert-large", "model_bert-base"
    ]

    # Precompute focal row observed support arrays o_{m, i} for all models
    o_row = {}
    q_n0_null = {}
    q_n4_null = {}
    for m in model_names:
        raw_key = m.replace("model_", "") if m.startswith("model_") else m
        p_m = np.array(model_probs_raw[raw_key])
        dist_m = distance_hellinger_matrix(p_m)
        w_m = compute_topk_weight_matrix(dist_m, k=10)
        o_row[m] = np.sum(w_m * s_target, axis=1) / 10.0
        q_n0_null[m] = cond_data[m]["null_ladder"][0]["null_mean"]
        q_n4_null[m] = cond_data[m]["null_ladder"][4]["null_mean"]

    # 1,000 Common 30-Stratum Paired Item Bootstrap
    n_boot = 1000
    rng = np.random.default_rng(2026_0803)

    boot_d_size = np.zeros(n_boot)
    boot_diff_roberta = np.zeros(n_boot)
    boot_diff_xlnet = np.zeros(n_boot)
    boot_diff_bert = np.zeros(n_boot)

    for b in range(n_boot):
        boot_indices = []
        for s_key, idxs in strata_map.items():
            resampled = rng.choice(idxs, size=len(idxs), replace=True)
            boot_indices.extend(resampled)
        boot_indices = np.array(boot_indices)

        f_n4_b = {}
        for m in model_names:
            q_obs_b = np.mean(o_row[m][boot_indices])
            n0_exc_b = max(1e-8, q_obs_b - q_n0_null[m])
            n4_exc_b = max(0.0, q_obs_b - q_n4_null[m])
            f_n4_b[m] = n4_exc_b / n0_exc_b

        d_roberta = f_n4_b["model_roberta-large"] - f_n4_b["model_roberta-base"]
        d_xlnet = f_n4_b["model_xlnet-large"] - f_n4_b["model_xlnet-base"]
        d_bert = f_n4_b["model_bert-large"] - f_n4_b["model_bert-base"]

        boot_diff_roberta[b] = d_roberta
        boot_diff_xlnet[b] = d_xlnet
        boot_diff_bert[b] = d_bert
        boot_d_size[b] = (d_roberta + d_xlnet + d_bert) / 3.0

    d_size_point = float(np.mean(boot_d_size))
    ci_d_size = [float(np.percentile(boot_d_size, 2.5)), float(np.percentile(boot_d_size, 97.5))]
    ci_roberta = [float(np.percentile(boot_diff_roberta, 2.5)), float(np.percentile(boot_diff_roberta, 97.5))]
    ci_xlnet = [float(np.percentile(boot_diff_xlnet, 2.5)), float(np.percentile(boot_diff_xlnet, 97.5))]
    ci_bert = [float(np.percentile(boot_diff_bert, 2.5)), float(np.percentile(boot_diff_bert, 97.5))]
    p_boot_gt_zero = float(np.mean(boot_d_size > 0))

    print(f"\nEXACT Primary Matched-Family Size Contrast D_size = {d_size_point*100.0:+.2f}%")
    print(f"  95% Percentile CI: [{ci_d_size[0]*100.0:+.2f}%, {ci_d_size[1]*100.0:+.2f}%]")
    print(f"  Bootstrap Support Pr_boot(D_size > 0) = {p_boot_gt_zero:.4f}\n")

    print("Family-Specific Matched Differences (Large - Base):")
    print(f"  RoBERTa: {np.mean(boot_diff_roberta)*100.0:+.2f}% (95% CI: [{ci_roberta[0]*100.0:+.2f}%, {ci_roberta[1]*100.0:+.2f}%])")
    print(f"  XLNet:   {np.mean(boot_diff_xlnet)*100.0:+.2f}% (95% CI: [{ci_xlnet[0]*100.0:+.2f}%, {ci_xlnet[1]*100.0:+.2f}%])")
    print(f"  BERT:    {np.mean(boot_diff_bert)*100.0:+.2f}% (95% CI: [{ci_bert[0]*100.0:+.2f}%, {ci_bert[1]*100.0:+.2f}%])")

    bootstrap_data = {
        "n_boot": n_boot,
        "method": "real_focal_row_item_resampling_30strata_exact_nulls",
        "d_size_point": d_size_point,
        "d_size_ci_95": ci_d_size,
        "p_boot_gt_zero": p_boot_gt_zero,
        "diff_roberta_point": float(np.mean(boot_diff_roberta)),
        "diff_roberta_ci_95": ci_roberta,
        "diff_xlnet_point": float(np.mean(boot_diff_xlnet)),
        "diff_xlnet_ci_95": ci_xlnet,
        "diff_bert_point": float(np.mean(boot_diff_bert)),
        "diff_bert_ci_95": ci_bert,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "E005_full_bootstrap.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(bootstrap_data, f, indent=2)

    print(f"\nSaved EXACT REAL E005 focal-row bootstrap results to {out_file}")

if __name__ == "__main__":
    main()
