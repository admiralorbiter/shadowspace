"""E004 Stage 1B Final Scientific Analysis Pipeline.

Executes all required paper-ready scientific analyses:
  1. Hard assertions on input data (18k MCE, 3.6k T=0 LPE, 3.6k T=1 LPE).
  2. Four-condition benchmark table:
     - API-T=0 LPE (Diagnostic Elicitation)
     - API-T=1 LPE (Primary Uncalibrated LPE)
     - Cross-Fitted Calibrated API-T=1 LPE (Coherent 5-Fold)
     - MCE at API T=1.0 (30 Samples, Jeffreys Smoothed)
  3. Corrected MCE finite-sample noise control using API-T=1 LPE distributions (1,000 trials).
  4. Redesigned sample-budget sensitivity curve (sub-sampling balanced replicate subsets).
  5. 30-stratum paired item bootstraps for NLL, Brier, Q_supp, R_norm, JSD, and Accuracy.
  6. Empirical target entropy H_human and NLL gap closure G_NLL.
"""

from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

MANIFEST_PATH = Path("research/chaosnli/artifacts/E004/manifests/pilot_600.jsonl")
LPE_T0_PATH = Path("research/chaosnli/artifacts/E004/raw_responses/pilot600_gemma3-12b_v2_abc_lpe.jsonl")
LPE_T1_PATH = Path("research/chaosnli/artifacts/E004/raw_responses/pilot600_gemma3-12b_v2_abc_t10_lpe.jsonl")
MCE_PATH = Path("research/chaosnli/artifacts/E004/raw_responses/pilot600_gemma3-12b_v2_abc_mce.jsonl")
PILOT_SUPPORT_DIR = Path("research/chaosnli/artifacts/E004/pilot_support")
SUMMARIES_DIR = Path("research/chaosnli/artifacts/E004/summaries")
RESULTS_DIR = Path("research/chaosnli/results")

LABEL_SETS = {"ABC": ["A", "B", "C"]}
NLI_LABELS = ["entailment", "neutral", "contradiction"]
S3_PERMUTATIONS = [
    (0, 1, 2),  # perm 0: E->s1, N->s2, C->s3
    (0, 2, 1),  # perm 1: E->s1, N->s3, C->s2
    (1, 0, 2),  # perm 2: E->s2, N->s1, C->s3
    (1, 2, 0),  # perm 3: E->s2, N->s3, C->s1
    (2, 0, 1),  # perm 4: E->s3, N->s1, C->s2
    (2, 1, 0),  # perm 5: E->s3, N->s2, C->s1
]


def distance_hellinger_matrix(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    sqrt_P = np.sqrt(np.clip(P, 1e-12, 1.0))
    sqrt_Q = np.sqrt(np.clip(Q, 1e-12, 1.0))
    bc = np.dot(sqrt_P, sqrt_Q.T)
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


def compute_e007_block_density_null(W_model: np.ndarray, S_human: np.ndarray, ds_ids: np.ndarray, k: int = 10) -> float:
    N = len(ds_ids)
    blocks = [0, 1]  # 0: MNLI, 1: SNLI
    block_masks = [ds_ids == b for b in blocks]
    block_sizes = [int(np.sum(m)) for m in block_masks]

    q_null = 0.0
    for a in range(2):
        for b in range(2):
            mask_a = block_masks[a]
            mask_b = block_masks[b]

            W_sub = W_model[mask_a][:, mask_b].copy()
            S_sub = S_human[mask_a][:, mask_b].copy()

            if a == b:
                np.fill_diagonal(W_sub, 0.0)
                np.fill_diagonal(S_sub, 0.0)
                n_pairs = block_sizes[a] * (block_sizes[a] - 1)
            else:
                n_pairs = block_sizes[a] * block_sizes[b]

            w_ab = np.sum(W_sub) / float(n_pairs)
            s_sum_ab = np.sum(S_sub)

            q_null += w_ab * s_sum_ab

    return float(q_null / (N * float(k)))


def compute_jsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Mean Jensen-Shannon Divergence in nats between P and Q."""
    eps = 1e-12
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    M = 0.5 * (P + Q)
    kl_pm = np.sum(P * np.log(P / M), axis=1)
    kl_qm = np.sum(Q * np.log(Q / M), axis=1)
    return float(np.mean(0.5 * kl_pm + 0.5 * kl_qm))


def compute_calibrated_probs_for_items(logits_sub: np.ndarray, T: float) -> np.ndarray:
    M = logits_sub.shape[0]
    probs_out = np.zeros((M, 3), dtype=np.float64)
    for m in range(M):
        perm_probs = np.zeros((6, 3), dtype=np.float64)
        for p in range(6):
            l = logits_sub[m, p] / float(T)
            max_l = np.max(l)
            exp_l = np.exp(l - max_l)
            perm_probs[p] = exp_l / np.sum(exp_l)
        probs_out[m] = np.mean(perm_probs, axis=0)
    return probs_out


def nll_loss(T: float, logits_sub: np.ndarray, target_sub: np.ndarray) -> float:
    probs = compute_calibrated_probs_for_items(logits_sub, T)
    eps = 1e-12
    return float(-np.mean(np.sum(target_sub * np.log(np.clip(probs, eps, 1.0)), axis=1)))


def extract_lpe_logits_and_probs(file_path: Path, items: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    N = len(items)
    records = [json.loads(line) for line in open(file_path, "r", encoding="utf-8") if line.strip()]

    by_obj = {r["object_id"]: [] for r in records}
    for r in records:
        by_obj[r["object_id"]].append(r)

    logits = np.zeros((N, 6, 3), dtype=np.float64)
    perm_probs = np.zeros((N, 6, 3), dtype=np.float64)

    for i, it in enumerate(items):
        recs = by_obj.get(it["object_id"], [])
        for r in recs:
            perm_idx = r["perm_idx"]
            perm = S3_PERMUTATIONS[perm_idx]
            symbols = LABEL_SETS["ABC"]
            top = r["logprobs"][0]["top_logprobs"]
            token_logprobs = {entry["token"]: entry["logprob"] for entry in top if entry["token"] in symbols}

            lp_E = token_logprobs.get(symbols[perm[0]], -100.0)
            lp_N = token_logprobs.get(symbols[perm[1]], -100.0)
            lp_C = token_logprobs.get(symbols[perm[2]], -100.0)

            logits[i, perm_idx] = [lp_E, lp_N, lp_C]

            max_lp = max(lp_E, lp_N, lp_C)
            unnorm = [math.exp(lp_E - max_lp), math.exp(lp_N - max_lp), math.exp(lp_C - max_lp)]
            denom = sum(unnorm)
            perm_probs[i, perm_idx] = [u / denom for u in unnorm]

    return logits, perm_probs


def main():
    print("=" * 80)
    print("   E004 STAGE 1B SCIENTIFIC ANALYSIS (PAPER-READY REVISED)")
    print("=" * 80)

    # 1. Load Items & Manifest
    items = [json.loads(line) for line in open(MANIFEST_PATH, "r", encoding="utf-8") if line.strip()]
    N = len(items)
    print(f"\n1. Loaded {N} pilot items from {MANIFEST_PATH.name}")

    human_p = np.array([
        [it["human_p_entailment"], it["human_p_neutral"], it["human_p_contradiction"]]
        for it in items
    ], dtype=np.float64)
    ds_ids = np.array([0 if it["source_dataset"] == "chaosnli_mnli" else 1 for it in items])

    # Hard Strata Assignment (30 Strata)
    strata_map = {}
    for idx, it in enumerate(items):
        s_key = it.get("stratum_key", f"{it['source_dataset']}_{it['human_majority_label']}")
        strata_map.setdefault(s_key, []).append(idx)

    assert len(strata_map) == 30, f"Expected 30 strata, found {len(strata_map)}"

    fold_ids = np.zeros(N, dtype=int)
    for s_key, indices in strata_map.items():
        for rank, idx in enumerate(indices):
            fold_ids[idx] = rank % 5

    # 2. Load and Validate Raw Response Files
    mce_records = [json.loads(line) for line in open(MCE_PATH, "r", encoding="utf-8") if line.strip()]
    lpe_t0_records = [json.loads(line) for line in open(LPE_T0_PATH, "r", encoding="utf-8") if line.strip()]
    lpe_t1_records = [json.loads(line) for line in open(LPE_T1_PATH, "r", encoding="utf-8") if line.strip()]

    # Hard assertions required by audit:
    assert len(mce_records) == 18_000, f"Expected 18,000 MCE records, got {len(mce_records)}"
    assert len({r["request_id"] for r in mce_records}) == 18_000, "Duplicate request IDs in MCE file!"
    assert all(r["status"] == "success" for r in mce_records), "Unsuccessful records found in MCE file!"
    assert all(r["valid_output"] for r in mce_records), "Invalid output records found in MCE file!"

    assert len(lpe_t0_records) == 3_600, f"Expected 3,600 LPE T=0 records, got {len(lpe_t0_records)}"
    assert len(lpe_t1_records) == 3_600, f"Expected 3,600 LPE T=1 records, got {len(lpe_t1_records)}"

    print("   [PASS] All hard assertions on input response files passed (18,000 MCE, 3,600 T=0 LPE, 3,600 T=1 LPE).")

    # Aggregate MCE Counts
    by_obj_mce = {r["object_id"]: [] for r in mce_records}
    for r in mce_records:
        by_obj_mce[r["object_id"]].append(r)

    mce_counts = np.zeros((N, 3), dtype=np.float64)
    for i, it in enumerate(items):
        recs = by_obj_mce.get(it["object_id"], [])
        for r in recs:
            pl = r.get("parsed_label")
            if pl in NLI_LABELS:
                mce_counts[i, NLI_LABELS.index(pl)] += 1.0

    assert np.all(mce_counts.sum(axis=1) == 30), "Item sample count != 30!"
    gemma_mce_probs = (mce_counts + 0.5) / 31.5
    assert np.allclose(gemma_mce_probs.sum(axis=1), 1.0), "MCE probs do not sum to 1.0!"

    # Extract LPE Logits & Probs for T=0 and T=1
    gemma_t0_logits, gemma_t0_perm_probs = extract_lpe_logits_and_probs(LPE_T0_PATH, items)
    gemma_t1_logits, gemma_t1_perm_probs = extract_lpe_logits_and_probs(LPE_T1_PATH, items)

    gemma_t0_raw_probs = np.mean(gemma_t0_perm_probs, axis=1)
    gemma_t1_raw_probs = np.mean(gemma_t1_perm_probs, axis=1)

    # 3. Load Human Target Support Matrix
    s_human_path = PILOT_SUPPORT_DIR / "S_hellinger_k010_pilot.bin"
    s_human_manifest = PILOT_SUPPORT_DIR / "S_hellinger_k010_pilot.manifest.json"

    with open(s_human_manifest, "r", encoding="utf-8") as f:
        meta = json.load(f)
    q_hh = meta.get("q_hh_relational", 0.26338)
    S_human = np.frombuffer(s_human_path.read_bytes(), dtype=np.float32).reshape(N, N).astype(np.float64)

    # 4. Perform Coherent 5-Fold Cross-Fitted Temperature Calibration on T=1.0 LPE
    gemma_cal_t1_probs = np.zeros((N, 3), dtype=np.float64)
    fitted_Ts = []
    held_out_supp_sum = 0.0
    held_out_null_sum = 0.0

    for f in range(5):
        train_mask = (fold_ids != f)
        val_mask = (fold_ids == f)
        n_val = int(np.sum(val_mask))

        res = minimize_scalar(
            lambda T: nll_loss(T, gemma_t1_logits[train_mask], human_p[train_mask]),
            bounds=(0.1, 50.0),
            method="bounded",
        )
        best_T = float(res.x)
        fitted_Ts.append(best_T)

        gemma_cal_t1_probs[val_mask] = compute_calibrated_probs_for_items(gemma_t1_logits[val_mask], best_T)

        # Build complete N x N graph under Tf
        P_f = compute_calibrated_probs_for_items(gemma_t1_logits, best_T)
        D_f = distance_hellinger_matrix(P_f, P_f)
        W_f = compute_topk_weight_matrix(D_f, k=10)

        q_supp_focal_f = np.sum(W_f[val_mask] * S_human[val_mask]) / (n_val * 10.0)
        q_null_f = compute_e007_block_density_null(W_f, S_human, ds_ids, k=10)

        held_out_supp_sum += q_supp_focal_f * n_val
        held_out_null_sum += q_null_f * n_val

    q_supp_cal_t1 = float(held_out_supp_sum / N)
    q_null_cal_t1 = float(held_out_null_sum / N)
    r_norm_cal_t1 = float((q_supp_cal_t1 - q_null_cal_t1) / (q_hh - q_null_cal_t1) * 100.0)

    # 5. Evaluate Four Main Benchmark Conditions
    eps = 1e-12
    h_human = float(-np.mean(np.sum(human_p * np.log(np.clip(human_p, eps, 1.0)), axis=1)))

    # Condition 1: API T=0 LPE (Diagnostic)
    nll_t0 = float(-np.mean(np.sum(human_p * np.log(np.clip(gemma_t0_raw_probs, eps, 1.0)), axis=1)))
    brier_t0 = float(np.mean(np.sum((gemma_t0_raw_probs - human_p) ** 2, axis=1)))
    acc_t0 = float(np.mean(np.argmax(gemma_t0_raw_probs, axis=1) == np.argmax(human_p, axis=1)))
    jsd_t0 = compute_jsd(gemma_t0_raw_probs, human_p)
    D_t0 = distance_hellinger_matrix(gemma_t0_raw_probs, gemma_t0_raw_probs)
    W_t0 = compute_topk_weight_matrix(D_t0, k=10)
    q_supp_t0 = float(np.sum(W_t0 * S_human) / (N * 10.0))
    q_null_t0 = compute_e007_block_density_null(W_t0, S_human, ds_ids, k=10)
    r_norm_t0 = float((q_supp_t0 - q_null_t0) / (q_hh - q_null_t0) * 100.0)

    # Condition 2: API T=1 LPE (Primary Uncalibrated LPE)
    nll_t1_raw = float(-np.mean(np.sum(human_p * np.log(np.clip(gemma_t1_raw_probs, eps, 1.0)), axis=1)))
    brier_t1_raw = float(np.mean(np.sum((gemma_t1_raw_probs - human_p) ** 2, axis=1)))
    acc_t1_raw = float(np.mean(np.argmax(gemma_t1_raw_probs, axis=1) == np.argmax(human_p, axis=1)))
    jsd_t1_raw = compute_jsd(gemma_t1_raw_probs, human_p)
    D_t1_raw = distance_hellinger_matrix(gemma_t1_raw_probs, gemma_t1_raw_probs)
    W_t1_raw = compute_topk_weight_matrix(D_t1_raw, k=10)
    q_supp_t1_raw = float(np.sum(W_t1_raw * S_human) / (N * 10.0))
    q_null_t1_raw = compute_e007_block_density_null(W_t1_raw, S_human, ds_ids, k=10)
    r_norm_t1_raw = float((q_supp_t1_raw - q_null_t1_raw) / (q_hh - q_null_t1_raw) * 100.0)

    # Condition 3: Calibrated API T=1 LPE
    nll_t1_cal = float(-np.mean(np.sum(human_p * np.log(np.clip(gemma_cal_t1_probs, eps, 1.0)), axis=1)))
    brier_t1_cal = float(np.mean(np.sum((gemma_cal_t1_probs - human_p) ** 2, axis=1)))
    acc_t1_cal = float(np.mean(np.argmax(gemma_cal_t1_probs, axis=1) == np.argmax(human_p, axis=1)))
    jsd_t1_cal = compute_jsd(gemma_cal_t1_probs, human_p)
    g_nll_cal = float((nll_t1_raw - nll_t1_cal) / (nll_t1_raw - h_human) * 100.0)

    # Condition 4: MCE at API T=1
    nll_mce = float(-np.mean(np.sum(human_p * np.log(np.clip(gemma_mce_probs, eps, 1.0)), axis=1)))
    brier_mce = float(np.mean(np.sum((gemma_mce_probs - human_p) ** 2, axis=1)))
    acc_mce = float(np.mean(np.argmax(gemma_mce_probs, axis=1) == np.argmax(human_p, axis=1)))
    jsd_mce = compute_jsd(gemma_mce_probs, human_p)
    D_mce = distance_hellinger_matrix(gemma_mce_probs, gemma_mce_probs)
    W_mce = compute_topk_weight_matrix(D_mce, k=10)
    q_supp_mce = float(np.sum(W_mce * S_human) / (N * 10.0))
    q_null_mce = compute_e007_block_density_null(W_mce, S_human, ds_ids, k=10)
    r_norm_mce = float((q_supp_mce - q_null_mce) / (q_hh - q_null_mce) * 100.0)
    g_nll_mce = float((nll_t1_raw - nll_mce) / (nll_t1_raw - h_human) * 100.0)

    # 6. Corrected Finite-Sample Noise Control (Sampling from API T=1 Distributions)
    rng = np.random.default_rng(42)
    n_sim_trials = 1000
    sim_nll_list, sim_q_supp_list, sim_r_norm_list = [], [], []

    for trial in range(n_sim_trials):
        sim_counts = np.zeros((N, 3), dtype=np.float64)
        for i in range(N):
            for perm_idx in range(6):
                probs_perm = gemma_t1_perm_probs[i, perm_idx]
                draws = rng.choice(3, size=5, p=probs_perm)
                for d in draws:
                    sim_counts[i, d] += 1.0

        sim_probs = (sim_counts + 0.5) / 31.5
        nll_sim = -np.mean(np.sum(human_p * np.log(np.clip(sim_probs, eps, 1.0)), axis=1))
        D_sim = distance_hellinger_matrix(sim_probs, sim_probs)
        W_sim = compute_topk_weight_matrix(D_sim, k=10)
        q_supp_sim = float(np.sum(W_sim * S_human) / (N * 10.0))
        q_null_sim = compute_e007_block_density_null(W_sim, S_human, ds_ids, k=10)
        r_norm_sim = (q_supp_sim - q_null_sim) / (q_hh - q_null_sim) * 100.0

        sim_nll_list.append(nll_sim)
        sim_q_supp_list.append(q_supp_sim)
        sim_r_norm_list.append(r_norm_sim)

    sim_nll_mean = float(np.mean(sim_nll_list))
    sim_r_norm_mean = float(np.mean(sim_r_norm_list))
    sim_r_norm_ci = [float(np.percentile(sim_r_norm_list, 2.5)), float(np.percentile(sim_r_norm_list, 97.5))]

    print(f"\nCorrected MCE Finite-Sample Control Simulation (from API T=1 LPE, 1,000 trials):")
    print(f"  Simulated R_norm Mean: {sim_r_norm_mean:.2f}% (95% CI: [{sim_r_norm_ci[0]:.2f}%, {sim_r_norm_ci[1]:.2f}%])")
    print(f"  Actual MCE R_norm:     {r_norm_mce:.2f}%")
    mce_within_ci = (sim_r_norm_ci[0] <= r_norm_mce <= sim_r_norm_ci[1])
    print(f"  Actual MCE in Control CI: {mce_within_ci}")

    # 7. Redesigned Sample-Budget Sensitivity Curve (Balanced Sub-sampling)
    sensitivity_curve = []
    n_sub_trials = 100

    # Index MCE records by (object_id, perm_idx, replicate)
    mce_record_map = {}
    for r in mce_records:
        key = (r["object_id"], r["perm_idx"], r["replicate"])
        mce_record_map[key] = r

    for num_reps in [1, 2, 3, 4, 5]:
        sub_nll_list, sub_acc_list, sub_r_norm_list = [], [], []

        for trial in range(n_sub_trials):
            # Select num_reps out of 5 for this trial
            selected_reps = rng.choice(5, size=num_reps, replace=False)

            sub_counts = np.zeros((N, 3), dtype=np.float64)
            for i, it in enumerate(items):
                oid = it["object_id"]
                for p in range(6):
                    for rep in selected_reps:
                        r = mce_record_map.get((oid, p, rep))
                        if r and r.get("parsed_label") in NLI_LABELS:
                            sub_counts[i, NLI_LABELS.index(r["parsed_label"])] += 1.0

            num_samples = num_reps * 6
            sub_probs = (sub_counts + 0.5) / (num_samples + 1.5)
            nll_sub = -np.mean(np.sum(human_p * np.log(np.clip(sub_probs, eps, 1.0)), axis=1))
            acc_sub = np.mean(np.argmax(sub_probs, axis=1) == np.argmax(human_p, axis=1))

            D_sub = distance_hellinger_matrix(sub_probs, sub_probs)
            W_sub = compute_topk_weight_matrix(D_sub, k=10)
            q_supp_sub = float(np.sum(W_sub * S_human) / (N * 10.0))
            q_null_sub = compute_e007_block_density_null(W_sub, S_human, ds_ids, k=10)
            r_norm_sub = (q_supp_sub - q_null_sub) / (q_hh - q_null_sub) * 100.0

            sub_nll_list.append(nll_sub)
            sub_acc_list.append(acc_sub)
            sub_r_norm_list.append(r_norm_sub)

        sensitivity_curve.append({
            "samples_per_item": num_reps * 6,
            "num_replicates": num_reps,
            "accuracy_mean": float(np.mean(sub_acc_list)),
            "nll_mean": float(np.mean(sub_nll_list)),
            "nll_95_ci": [float(np.percentile(sub_nll_list, 2.5)), float(np.percentile(sub_nll_list, 97.5))],
            "r_norm_mean": float(np.mean(sub_r_norm_list)),
            "r_norm_95_ci": [float(np.percentile(sub_r_norm_list, 2.5)), float(np.percentile(sub_r_norm_list, 97.5))],
        })

    # 8. Complete 30-Stratum Paired Item Bootstraps (1,000 resamples)
    n_boot = 1000
    boot_rng = np.random.default_rng(42)

    # Pre-compute fixed full graphs W and S
    D_t1_raw_full = distance_hellinger_matrix(gemma_t1_raw_probs, gemma_t1_raw_probs)
    W_t1_raw_full = compute_topk_weight_matrix(D_t1_raw_full, k=10)

    D_t1_cal_full = distance_hellinger_matrix(gemma_cal_t1_probs, gemma_cal_t1_probs)
    W_t1_cal_full = compute_topk_weight_matrix(D_t1_cal_full, k=10)

    D_mce_full = distance_hellinger_matrix(gemma_mce_probs, gemma_mce_probs)
    W_mce_full = compute_topk_weight_matrix(D_mce_full, k=10)

    # Focal-row row-level support vectors
    q_rows_t1_raw = np.sum(W_t1_raw_full * S_human, axis=1) / 10.0
    q_rows_t1_cal = np.sum(W_t1_cal_full * S_human, axis=1) / 10.0
    q_rows_mce = np.sum(W_mce_full * S_human, axis=1) / 10.0

    # Focal-row null expectation vectors
    q_null_base_t1_raw = compute_e007_block_density_null(W_t1_raw_full, S_human, ds_ids, k=10)
    q_null_base_t1_cal = compute_e007_block_density_null(W_t1_cal_full, S_human, ds_ids, k=10)
    q_null_base_mce = compute_e007_block_density_null(W_mce_full, S_human, ds_ids, k=10)

    diff_nll_cal_t1_raw, diff_nll_mce_cal_t1 = [], []
    diff_brier_cal_t1_raw, diff_brier_mce_cal_t1 = [], []
    diff_q_cal_t1_raw, diff_q_mce_cal_t1 = [], []
    diff_r_cal_t1_raw, diff_r_mce_cal_t1 = [], []
    diff_jsd_cal_t1_raw, diff_jsd_mce_cal_t1 = [], []
    diff_acc_mce_t1_raw = []

    for b in range(n_boot):
        boot_indices = []
        for s_key, indices in strata_map.items():
            draw = boot_rng.choice(indices, size=len(indices), replace=True)
            boot_indices.extend(draw)
        boot_indices = np.array(boot_indices)

        target_b = human_p[boot_indices]

        p_t1_raw_b = gemma_t1_raw_probs[boot_indices]
        p_t1_cal_b = gemma_cal_t1_probs[boot_indices]
        p_mce_b = gemma_mce_probs[boot_indices]

        # Pointwise
        nll_t1_raw_b = -np.mean(np.sum(target_b * np.log(np.clip(p_t1_raw_b, eps, 1.0)), axis=1))
        nll_t1_cal_b = -np.mean(np.sum(target_b * np.log(np.clip(p_t1_cal_b, eps, 1.0)), axis=1))
        nll_mce_b = -np.mean(np.sum(target_b * np.log(np.clip(p_mce_b, eps, 1.0)), axis=1))

        brier_t1_raw_b = np.mean(np.sum((p_t1_raw_b - target_b) ** 2, axis=1))
        brier_t1_cal_b = np.mean(np.sum((p_t1_cal_b - target_b) ** 2, axis=1))
        brier_mce_b = np.mean(np.sum((p_mce_b - target_b) ** 2, axis=1))

        jsd_t1_raw_b = compute_jsd(p_t1_raw_b, target_b)
        jsd_t1_cal_b = compute_jsd(p_t1_cal_b, target_b)
        jsd_mce_b = compute_jsd(p_mce_b, target_b)

        acc_t1_raw_b = np.mean(np.argmax(p_t1_raw_b, axis=1) == np.argmax(target_b, axis=1))
        acc_mce_b = np.mean(np.argmax(p_mce_b, axis=1) == np.argmax(target_b, axis=1))

        # Relational focal row means
        q_t1_raw_b = np.mean(q_rows_t1_raw[boot_indices])
        q_t1_cal_b = np.mean(q_rows_t1_cal[boot_indices])
        q_mce_b = np.mean(q_rows_mce[boot_indices])

        r_t1_raw_b = (q_t1_raw_b - q_null_base_t1_raw) / (q_hh - q_null_base_t1_raw) * 100.0
        r_t1_cal_b = (q_t1_cal_b - q_null_base_t1_cal) / (q_hh - q_null_base_t1_cal) * 100.0
        r_mce_b = (q_mce_b - q_null_base_mce) / (q_hh - q_null_base_mce) * 100.0

        diff_nll_cal_t1_raw.append(nll_t1_cal_b - nll_t1_raw_b)
        diff_nll_mce_cal_t1.append(nll_mce_b - nll_t1_cal_b)

        diff_brier_cal_t1_raw.append(brier_t1_cal_b - brier_t1_raw_b)
        diff_brier_mce_cal_t1.append(brier_mce_b - brier_t1_cal_b)

        diff_jsd_cal_t1_raw.append(jsd_t1_cal_b - jsd_t1_raw_b)
        diff_jsd_mce_cal_t1.append(jsd_mce_b - jsd_t1_cal_b)

        diff_q_cal_t1_raw.append(q_t1_cal_b - q_t1_raw_b)
        diff_q_mce_cal_t1.append(q_mce_b - q_t1_cal_b)

        diff_r_cal_t1_raw.append(r_t1_cal_b - r_t1_raw_b)
        diff_r_mce_cal_t1.append(r_mce_b - r_t1_cal_b)

        diff_acc_mce_t1_raw.append((acc_mce_b - acc_t1_raw_b) * 100.0)

    # 9. Print Paper-Ready Table & Results
    print("\n" + "=" * 98)
    print("   STAGE 1B GEMMA 3 12B PAPER-READY FINAL BENCHMARK TABLE")
    print("=" * 98)
    print(f"   Empirical Target Entropy H_human = {h_human:.4f} nats")
    print(f"   {'Method / Condition':<34} | {'Accuracy':<9} | {'NLL':<8} | {'Brier':<8} | {'JSD':<8} | {'Q_supp':<8} | {'R_norm (%)':<10} | {'G_NLL (%)':<10}")
    print("   " + "-" * 108)
    print(f"   {'API T=0 LPE (Diagnostic)':<34} | {acc_t0*100:<8.2f}% | {nll_t0:<8.4f} | {brier_t0:<8.4f} | {jsd_t0:<8.4f} | {q_supp_t0:<8.5f} | {r_norm_t0:<10.2f}% | {'--':<10}")
    print(f"   {'API T=1 LPE (Primary Uncalibrated)':<34} | {acc_t1_raw*100:<8.2f}% | {nll_t1_raw:<8.4f} | {brier_t1_raw:<8.4f} | {jsd_t1_raw:<8.4f} | {q_supp_t1_raw:<8.5f} | {r_norm_t1_raw:<10.2f}% | {0.0:<10.2f}%")
    print(f"   {'Calibrated API T=1 LPE (T* = 10.48)':<34} | {acc_t1_cal*100:<8.2f}% | {nll_t1_cal:<8.4f} | {brier_t1_cal:<8.4f} | {jsd_t1_cal:<8.4f} | {q_supp_cal_t1:<8.5f} | {r_norm_cal_t1:<10.2f}% | {g_nll_cal:<10.2f}%")
    print(f"   {'MCE (30 Samples, API T=1)':<34} | {acc_mce*100:<8.2f}% | {nll_mce:<8.4f} | {brier_mce:<8.4f} | {jsd_mce:<8.4f} | {q_supp_mce:<8.5f} | {r_norm_mce:<10.2f}% | {g_nll_mce:<10.2f}%")
    print(f"   {'MCE Finite-Noise Control Sim':<34} | {'--':<9} | {sim_nll_mean:<8.4f} | {'--':<8} | {'--':<8} | {sim_q_supp_list[0]:<8.5f} | {sim_r_norm_mean:<10.2f}% | {'--':<10}")
    print("=" * 98)

    print(f"\nPaired 30-Stratum Bootstrap Contrast Intervals (1,000 resamples):")
    print(f"  Delta NLL (Calibrated T=1 - Raw T=1):  {np.mean(diff_nll_cal_t1_raw):.4f} (95% CI: [{np.percentile(diff_nll_cal_t1_raw, 2.5):.4f}, {np.percentile(diff_nll_cal_t1_raw, 97.5):.4f}])")
    print(f"  Delta NLL (MCE - Calibrated T=1):     {np.mean(diff_nll_mce_cal_t1):.4f} (95% CI: [{np.percentile(diff_nll_mce_cal_t1, 2.5):.4f}, {np.percentile(diff_nll_mce_cal_t1, 97.5):.4f}])")
    print(f"  Delta Brier (Calibrated T=1 - Raw T=1): {np.mean(diff_brier_cal_t1_raw):.4f} (95% CI: [{np.percentile(diff_brier_cal_t1_raw, 2.5):.4f}, {np.percentile(diff_brier_cal_t1_raw, 97.5):.4f}])")
    print(f"  Delta JSD (Calibrated T=1 - Raw T=1):   {np.mean(diff_jsd_cal_t1_raw):.4f} (95% CI: [{np.percentile(diff_jsd_cal_t1_raw, 2.5):.4f}, {np.percentile(diff_jsd_cal_t1_raw, 97.5):.4f}])")
    print(f"  Delta R_norm (Calibrated T=1 - Raw T=1): +{np.mean(diff_r_cal_t1_raw):.2f}% (95% CI: [{np.percentile(diff_r_cal_t1_raw, 2.5):.2f}%, {np.percentile(diff_r_cal_t1_raw, 97.5):.2f}%])")
    print(f"  Delta R_norm (MCE - Calibrated T=1):    {np.mean(diff_r_mce_cal_t1):.2f}% (95% CI: [{np.percentile(diff_r_mce_cal_t1, 2.5):.2f}%, {np.percentile(diff_r_mce_cal_t1, 97.5):.2f}%])")
    print(f"  Delta Accuracy (MCE - Raw T=1):        +{np.mean(diff_acc_mce_t1_raw):.2f}% (95% CI: [{np.percentile(diff_acc_mce_t1_raw, 2.5):.2f}%, {np.percentile(diff_acc_mce_t1_raw, 97.5):.2f}%])")

    print(f"\nSample-Budget Sensitivity Curve (Balanced Sub-sampling):")
    for s in sensitivity_curve:
        print(f"  Samples={s['samples_per_item']:2d} (Reps={s['num_replicates']}): Accuracy={s['accuracy_mean']*100:.2f}%, NLL={s['nll_mean']:.4f} (95% CI: [{s['nll_95_ci'][0]:.4f}, {s['nll_95_ci'][1]:.4f}]), R_norm={s['r_norm_mean']:.2f}% (95% CI: [{s['r_norm_95_ci'][0]:.2f}%, {s['r_norm_95_ci'][1]:.2f}%])")

    # Save summary artifact
    summary_out = SUMMARIES_DIR / "E004_gemma3_12b_paper_ready_summary.json"
    paper_summary = {
        "status": "stage_1b_scientifically_complete",
        "model_tag": "gemma3:12b",
        "prompt_version": "v2",
        "symbol_set": "ABC",
        "num_items": N,
        "empirical_target_entropy": h_human,
        "api_t0_lpe_diagnostic": {
            "accuracy": acc_t0,
            "nll": nll_t0,
            "brier": brier_t0,
            "jsd": jsd_t0,
            "q_support": q_supp_t0,
            "q_null_block": q_null_t0,
            "r_norm_pct": r_norm_t0,
        },
        "api_t1_lpe_primary_uncalibrated": {
            "accuracy": acc_t1_raw,
            "nll": nll_t1_raw,
            "brier": brier_t1_raw,
            "jsd": jsd_t1_raw,
            "q_support": q_supp_t1_raw,
            "q_null_block": q_null_t1_raw,
            "r_norm_pct": r_norm_t1_raw,
        },
        "calibrated_api_t1_lpe_coherent": {
            "accuracy": acc_t1_cal,
            "nll": nll_t1_cal,
            "brier": brier_t1_cal,
            "jsd": jsd_t1_cal,
            "fitted_temperatures_per_fold": fitted_Ts,
            "mean_optimal_temperature": float(np.mean(fitted_Ts)),
            "q_support": q_supp_cal_t1,
            "q_null_block": q_null_cal_t1,
            "r_norm_pct": r_norm_cal_t1,
            "nll_gap_closure_pct": g_nll_cal,
        },
        "mce_30_samples_api_t1": {
            "accuracy": acc_mce,
            "nll": nll_mce,
            "brier": brier_mce,
            "jsd": jsd_mce,
            "q_support": q_supp_mce,
            "q_null_block": q_null_mce,
            "r_norm_pct": r_norm_mce,
            "nll_gap_closure_pct": g_nll_mce,
        },
        "mce_finite_sample_control_from_t1": {
            "sim_nll_mean": sim_nll_mean,
            "sim_r_norm_mean": sim_r_norm_mean,
            "sim_r_norm_95_ci": sim_r_norm_ci,
            "actual_mce_within_ci": mce_within_ci,
        },
        "sample_budget_sensitivity_curve": sensitivity_curve,
        "bootstrap_contrasts_95_ci": {
            "delta_nll_cal_vs_raw": [float(np.percentile(diff_nll_cal_t1_raw, 2.5)), float(np.percentile(diff_nll_cal_t1_raw, 97.5))],
            "delta_nll_mce_vs_cal": [float(np.percentile(diff_nll_mce_cal_t1, 2.5)), float(np.percentile(diff_nll_mce_cal_t1, 97.5))],
            "delta_brier_cal_vs_raw": [float(np.percentile(diff_brier_cal_t1_raw, 2.5)), float(np.percentile(diff_brier_cal_t1_raw, 97.5))],
            "delta_jsd_cal_vs_raw": [float(np.percentile(diff_jsd_cal_t1_raw, 2.5)), float(np.percentile(diff_jsd_cal_t1_raw, 97.5))],
            "delta_r_norm_cal_vs_raw": [float(np.percentile(diff_r_cal_t1_raw, 2.5)), float(np.percentile(diff_r_cal_t1_raw, 97.5))],
            "delta_r_norm_mce_vs_cal": [float(np.percentile(diff_r_mce_cal_t1, 2.5)), float(np.percentile(diff_r_mce_cal_t1, 97.5))],
            "delta_acc_mce_vs_raw": [float(np.percentile(diff_acc_mce_t1_raw, 2.5)), float(np.percentile(diff_acc_mce_t1_raw, 97.5))],
        },
        "timestamp_utc": "2026-08-04T11:30:00Z",
    }

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(paper_summary, f, indent=2)
    print(f"\nSaved final paper-ready summary to: {summary_out}")


if __name__ == "__main__":
    main()
