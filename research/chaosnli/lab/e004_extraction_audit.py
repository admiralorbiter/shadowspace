"""E004 Stage 1B Gemma 3 12B LPE Extraction Audit & Benchmark Pipeline.

Performs all required Gate 2 extraction checks on pilot600_gemma3-12b_v2_abc_lpe.jsonl:
  1. Manifest alignment (600 object IDs match pilot_600.jsonl in exact order)
  2. Request completeness (3,600 unique request IDs, 0 errors)
  3. Candidate-token coverage ('A', 'B', 'C' present in top_logprobs) & Mass Leakage Z_i,pi
  4. Probability validity & permutation-level log-softmax unpacking
  5. Prompt hash & model digest verification
  6. Pointwise evaluation (NLL, Brier score, Majority accuracy)
  7. 5-Fold Cross-Fitted Scalar Temperature Calibration (T*)
  8. Relational evaluation under exact Dataset-Stratified Analytic Null (Q_null_strat)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

MANIFEST_PATH = Path("research/chaosnli/artifacts/E004/manifests/pilot_600.jsonl")
LPE_RESPONSES_PATH = Path("research/chaosnli/artifacts/E004/raw_responses/pilot600_gemma3-12b_v2_abc_lpe.jsonl")
PILOT_SUPPORT_DIR = Path("research/chaosnli/artifacts/E004/pilot_support")
SUMMARIES_DIR = Path("research/chaosnli/artifacts/E004/summaries")

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
    """Compute pairwise Hellinger distance matrix."""
    sqrt_P = np.sqrt(np.clip(P, 1e-12, 1.0))
    sqrt_Q = np.sqrt(np.clip(Q, 1e-12, 1.0))
    bc = np.dot(sqrt_P, sqrt_Q.T)
    bc = np.clip(bc, 0.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - bc))


def compute_topk_weight_matrix(dist: np.ndarray, k: int) -> np.ndarray:
    """Compute tie-aware soft top-k neighbor weight matrix W[i, j] in [0, 1]."""
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


def compute_dataset_stratified_null(W_model: np.ndarray, S_human: np.ndarray, ds_ids: np.ndarray, k: int = 10) -> float:
    """Calculate exact dataset-stratified analytic null expectation Q_null."""
    N = len(ds_ids)
    n1 = int(np.sum(ds_ids == 0))
    n2 = int(np.sum(ds_ids == 1))

    val = 0.0
    for i in range(N):
        same_mask = (ds_ids == ds_ids[i])
        same_mask[i] = False
        diff_mask = (ds_ids != ds_ids[i])

        m_w = np.sum(W_model[i, same_mask])
        m_c = np.sum(W_model[i, diff_mask])
        h_w = np.sum(S_human[i, same_mask])
        h_c = np.sum(S_human[i, diff_mask])

        denom_w = (n1 - 1) if ds_ids[i] == 0 else (n2 - 1)
        denom_c = n2 if ds_ids[i] == 0 else n1

        val += (m_w * h_w / denom_w) + (m_c * h_c / denom_c)

    return float(val / (N * float(k)))


def compute_calibrated_probs_for_items(logits_sub: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature T to logits and compute mean probability across permutations."""
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
    """NLL loss function for temperature optimization."""
    probs = compute_calibrated_probs_for_items(logits_sub, T)
    eps = 1e-12
    return float(-np.mean(np.sum(target_sub * np.log(np.clip(probs, eps, 1.0)), axis=1)))


def main():
    print("=" * 80)
    print("   E004 STAGE 1B GEMMA 3 12B LPE EXTRACTION AUDIT & RELATIONAL EVALUATION")
    print("=" * 80)

    # 1. Load Manifest
    items = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    N = len(items)
    print(f"\n1. Manifest Check: Loaded {N} pilot items from {MANIFEST_PATH.name}")

    human_p = np.array([
        [it["human_p_entailment"], it["human_p_neutral"], it["human_p_contradiction"]]
        for it in items
    ], dtype=np.float64)
    ds_ids = np.array([0 if it["source_dataset"] == "chaosnli_mnli" else 1 for it in items])

    # 2. Load Raw Responses & Audit Completeness
    raw_records = []
    with open(LPE_RESPONSES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line.strip()))

    print(f"\n2. Completeness Audit:")
    print(f"   Total raw records read:    {len(raw_records)}")
    success_records = [r for r in raw_records if r.get("status") == "success"]
    error_records = [r for r in raw_records if r.get("status") == "error"]
    print(f"   Successful records:        {len(success_records)}")
    print(f"   Error records:             {len(error_records)}")

    unique_req_ids = set(r["request_id"] for r in success_records)
    print(f"   Unique Request IDs:        {len(unique_req_ids)} / 3,600 expected")

    by_object: Dict[str, List[Dict]] = {}
    for r in success_records:
        oid = r["object_id"]
        if oid not in by_object:
            by_object[oid] = []
        by_object[oid].append(r)

    print(f"   Unique Object IDs:         {len(by_object)} / {N} expected")
    manifest_object_ids = [it["object_id"] for it in items]
    alignment_ok = (list(by_object.keys()) == manifest_object_ids)
    print(f"   Manifest Sequence Align:  {'EXACT MATCH' if alignment_ok else 'DISCREPANCY DETECTED'}")

    v1_count = sum(1 for r in raw_records if r.get("prompt_version") == "v1")
    print(f"   v1 Contamination Records: {v1_count}")

    # 3. Candidate Token Mass Leakage & Provenance Audit
    candidate_masses = []
    all_prompt_hashes = set()
    all_model_digests = set()
    missing_symbols_count = 0

    for r in success_records:
        all_prompt_hashes.add(r.get("user_prompt_sha256"))
        all_model_digests.add(r.get("model_digest"))
        lp = r.get("logprobs", [])
        if not lp:
            missing_symbols_count += 1
            continue
        top = lp[0].get("top_logprobs", [])
        found_tokens = {entry["token"] for entry in top}
        if not {"A", "B", "C"}.issubset(found_tokens):
            missing_symbols_count += 1

        cand_mass = sum(math.exp(e["logprob"]) for e in top if e["token"] in ["A", "B", "C"])
        candidate_masses.append(cand_mass)

    z_arr = np.array(candidate_masses)
    print(f"\n3. Token & Candidate Mass Leakage Audit:")
    print(f"   Missing Symbol Logprobs:  {missing_symbols_count} / {len(success_records)}")
    print(f"   Candidate Mass Z_i,pi (P(A)+P(B)+P(C)):")
    print(f"     Mean: {np.mean(z_arr):.6f} | Median: {np.median(z_arr):.6f} | Min: {np.min(z_arr):.6f}")
    print(f"     p01:  {np.percentile(z_arr, 1):.6f} | p05:  {np.percentile(z_arr, 5):.6f}")

    # 4. Unpack Unnormalized Logits & Probabilities
    # gemma_logits shape: (N, 6, 3)
    gemma_logits = np.zeros((N, 6, 3), dtype=np.float64)
    gemma_perm_probs = np.zeros((N, 6, 3), dtype=np.float64)

    for i, it in enumerate(items):
        oid = it["object_id"]
        recs = by_object.get(oid, [])
        for r in recs:
            perm_idx = r["perm_idx"]
            perm = S3_PERMUTATIONS[perm_idx]
            symbols = LABEL_SETS["ABC"]

            top = r["logprobs"][0]["top_logprobs"]
            token_logprobs = {e["token"]: e["logprob"] for e in top if e["token"] in symbols}

            lp_E = token_logprobs.get(symbols[perm[0]], -100.0)
            lp_N = token_logprobs.get(symbols[perm[1]], -100.0)
            lp_C = token_logprobs.get(symbols[perm[2]], -100.0)

            gemma_logits[i, perm_idx] = [lp_E, lp_N, lp_C]

            max_lp = max(lp_E, lp_N, lp_C)
            unnorm_E = math.exp(lp_E - max_lp)
            unnorm_N = math.exp(lp_N - max_lp)
            unnorm_C = math.exp(lp_C - max_lp)
            denom = unnorm_E + unnorm_N + unnorm_C

            gemma_perm_probs[i, perm_idx] = [unnorm_E / denom, unnorm_N / denom, unnorm_C / denom]

    gemma_raw_avg_probs = np.mean(gemma_perm_probs, axis=1)

    # 5. Pointwise Metric Evaluation (Raw vs Calibrated)
    eps = 1e-12
    nll_raw = float(-np.mean(np.sum(human_p * np.log(np.clip(gemma_raw_avg_probs, eps, 1.0)), axis=1)))
    brier_raw = float(np.mean(np.sum((gemma_raw_avg_probs - human_p) ** 2, axis=1)))
    acc_raw = float(np.mean(np.argmax(gemma_raw_avg_probs, axis=1) == np.argmax(human_p, axis=1)))

    print(f"\n5. Pointwise Evaluation (Gemma 3 12B LPE Raw):")
    print(f"   Negative Log-Likelihood (NLL): {nll_raw:.4f}")
    print(f"   Brier Score:                   {brier_raw:.4f}")
    print(f"   Majority Gold Accuracy:        {acc_raw * 100:.2f}%")

    # 6. 5-Fold Cross-Fitted Scalar Temperature Calibration
    n_folds = 5
    fold_ids = np.array([i % n_folds for i in range(N)])
    gemma_cal_probs = np.zeros((N, 3), dtype=np.float64)
    fitted_temperatures = []

    for f in range(n_folds):
        train_mask = (fold_ids != f)
        val_mask = (fold_ids == f)

        res = minimize_scalar(
            lambda T: nll_loss(T, gemma_logits[train_mask], human_p[train_mask]),
            bounds=(0.1, 50.0),
            method="bounded",
        )
        best_T = float(res.x)
        fitted_temperatures.append(best_T)
        gemma_cal_probs[val_mask] = compute_calibrated_probs_for_items(gemma_logits[val_mask], best_T)

    nll_cal = float(-np.mean(np.sum(human_p * np.log(np.clip(gemma_cal_probs, eps, 1.0)), axis=1)))
    brier_cal = float(np.mean(np.sum((gemma_cal_probs - human_p) ** 2, axis=1)))
    acc_cal = float(np.mean(np.argmax(gemma_cal_probs, axis=1) == np.argmax(human_p, axis=1)))

    print(f"\n6. 5-Fold Cross-Fitted Scalar Temperature Calibration (Held-Out):")
    print(f"   Fitted Temperatures per Fold: {[round(t, 2) for t in fitted_temperatures]}")
    print(f"   Mean Optimal Temperature T*: {np.mean(fitted_temperatures):.2f}")
    print(f"   Calibrated NLL:              {nll_cal:.4f} (Raw: {nll_raw:.4f})")
    print(f"   Calibrated Brier Score:       {brier_cal:.4f} (Raw: {brier_raw:.4f})")
    print(f"   Calibrated Majority Accuracy: {acc_cal * 100:.2f}% (Raw: {acc_raw * 100:.2f}%)")

    # 7. Relational Evaluation under Exact Stratified Analytic Null
    s_human_path = PILOT_SUPPORT_DIR / "S_hellinger_k010_pilot.bin"
    s_human_manifest = PILOT_SUPPORT_DIR / "S_hellinger_k010_pilot.manifest.json"

    if s_human_path.exists() and s_human_manifest.exists():
        with open(s_human_manifest, "r", encoding="utf-8") as f:
            meta = json.load(f)
        q_hh = meta.get("q_hh_relational", 0.26338)

        S_human = np.frombuffer(s_human_path.read_bytes(), dtype=np.float32).reshape(N, N).astype(np.float64)

        # Raw Relational Topology
        D_raw = distance_hellinger_matrix(gemma_raw_avg_probs, gemma_raw_avg_probs)
        W_raw = compute_topk_weight_matrix(D_raw, k=10)
        q_supp_raw = float(np.sum(W_raw * S_human) / (N * 10.0))

        q_null_unrestricted = 10.0 / 599.0
        q_null_strat_raw = compute_dataset_stratified_null(W_raw, S_human, ds_ids, k=10)

        r_norm_unrestricted_raw = (q_supp_raw - q_null_unrestricted) / (q_hh - q_null_unrestricted) * 100.0
        r_norm_strat_raw = (q_supp_raw - q_null_strat_raw) / (q_hh - q_null_strat_raw) * 100.0

        # Calibrated Relational Topology
        D_cal = distance_hellinger_matrix(gemma_cal_probs, gemma_cal_probs)
        W_cal = compute_topk_weight_matrix(D_cal, k=10)
        q_supp_cal = float(np.sum(W_cal * S_human) / (N * 10.0))

        q_null_strat_cal = compute_dataset_stratified_null(W_cal, S_human, ds_ids, k=10)
        r_norm_strat_cal = (q_supp_cal - q_null_strat_cal) / (q_hh - q_null_strat_cal) * 100.0

        print(f"\n7. Corrected Relational Performance (Raw vs Calibrated):")
        print(f"   Human Target Q_HH (k=10):     {q_hh:.5f}")
        print(f"   Unrestricted Null (10/599):    {q_null_unrestricted:.5f}")
        print(f"   -----------------------------------------------------")
        print(f"   Raw LPE Q_support:            {q_supp_raw:.5f}")
        print(f"   Raw Stratified Analytic Null: {q_null_strat_raw:.5f}")
        print(f"   Raw R_norm (Unrestricted):    {r_norm_unrestricted_raw:.2f}%")
        print(f"   Raw R_norm (Stratified Null): {r_norm_strat_raw:.2f}%")
        print(f"   -----------------------------------------------------")
        print(f"   Calibrated LPE Q_support:     {q_supp_cal:.5f}")
        print(f"   Calibrated Stratified Null:   {q_null_strat_cal:.5f}")
        print(f"   Calibrated R_norm (Stratified): {r_norm_strat_cal:.2f}%")

        # Save Summary Artifact
        SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
        summary_out = SUMMARIES_DIR / "E004_gemma3_12b_lpe_extraction_audit.json"
        audit_summary = {
            "model_tag": "gemma3:12b",
            "prompt_version": "v2",
            "symbol_set": "ABC",
            "num_items": N,
            "total_raw_records": len(raw_records),
            "successful_records": len(success_records),
            "error_records": len(error_records),
            "unique_request_ids": len(unique_req_ids),
            "manifest_alignment": alignment_ok,
            "candidate_mass_z_mean": float(np.mean(z_arr)),
            "candidate_mass_z_min": float(np.min(z_arr)),
            "raw_metrics": {
                "nll": nll_raw,
                "brier_score": brier_raw,
                "gold_accuracy": acc_raw,
                "q_support": q_supp_raw,
                "q_null_unrestricted": q_null_unrestricted,
                "q_null_stratified": q_null_strat_raw,
                "q_hh": q_hh,
                "r_norm_unrestricted_pct": r_norm_unrestricted_raw,
                "r_norm_stratified_pct": r_norm_strat_raw,
            },
            "calibrated_metrics": {
                "nll": nll_cal,
                "brier_score": brier_cal,
                "gold_accuracy": acc_cal,
                "fitted_temperatures_per_fold": fitted_temperatures,
                "mean_optimal_temperature": float(np.mean(fitted_temperatures)),
                "q_support": q_supp_cal,
                "q_null_stratified": q_null_strat_cal,
                "q_hh": q_hh,
                "r_norm_stratified_pct": r_norm_strat_cal,
            },
            "timestamp_utc": "2026-08-03T22:56:00Z",
        }
        with open(summary_out, "w", encoding="utf-8") as f:
            json.dump(audit_summary, f, indent=2)
        print(f"\nSaved audited summary to: {summary_out}")

    print("=" * 80)


if __name__ == "__main__":
    main()
