"""E004 Stage 1B Gemma 3 12B MCE Extraction Audit & Benchmark Pipeline.

Processes 18,000 MCE requests from pilot600_gemma3-12b_v2_abc_mce.jsonl:
  1. Validates completeness (18,000 records, 600 items x 30 samples/item)
  2. Applies Jeffreys smoothing: p_ic = (n_ic + 0.5) / (30 + 1.5)
  3. Computes pointwise metrics: NLL, Brier score, Majority Gold Accuracy
  4. Computes relational topology metrics: Hellinger distance, Q_support, Q_null_strat, R_norm
  5. Produces side-by-side comparison table: Raw LPE vs Calibrated LPE vs MCE
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

MANIFEST_PATH = Path("research/chaosnli/artifacts/E004/manifests/pilot_600.jsonl")
MCE_RESPONSES_PATH = Path("research/chaosnli/artifacts/E004/raw_responses/pilot600_gemma3-12b_v2_abc_mce.jsonl")
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


def main():
    print("=" * 80)
    print("   E004 STAGE 1B GEMMA 3 12B MCE EXTRACTION AUDIT & RELATIONAL EVALUATION")
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

    # 2. Load MCE Raw Records
    raw_records = []
    with open(MCE_RESPONSES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line.strip()))

    print(f"\n2. Completeness Audit:")
    print(f"   Total MCE records read:    {len(raw_records)}")
    success_records = [r for r in raw_records if r.get("status") == "success"]
    invalid_records = [r for r in success_records if not r.get("valid_output", True)]

    print(f"   Successful records:        {len(success_records)} / 18,000 expected")
    print(f"   Invalid output records:    {len(invalid_records)} / 18,000")
    print(f"   Valid sample rate:         {(len(success_records) - len(invalid_records)) / len(success_records) * 100:.3f}%")

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

    # 3. Aggregate Counts & Apply Jeffreys Smoothing
    # Counts shape: (N, 3) for [E, N, C]
    mce_counts = np.zeros((N, 3), dtype=np.float64)

    for i, it in enumerate(items):
        recs = by_object.get(it["object_id"], [])
        for r in recs:
            parsed_label = r.get("parsed_label")
            if parsed_label in NLI_LABELS:
                label_idx = NLI_LABELS.index(parsed_label)
                mce_counts[i, label_idx] += 1.0

    # Jeffreys Smoothing: p_ic = (n_ic + 0.5) / (30 + 1.5)
    mce_probs = (mce_counts + 0.5) / 31.5

    # Verify probability normalization
    prob_sums = np.sum(mce_probs, axis=1)
    print(f"\n3. MCE Probability Estimation (Jeffreys Smoothed):")
    print(f"   MCE Probs Shape:          {mce_probs.shape}")
    print(f"   Mean Prob Sum per Item:   {np.mean(prob_sums):.6f} (Expected 1.0)")
    print(f"   Total valid samples:       {int(np.sum(mce_counts))} / 18,000 expected")

    # 4. Pointwise Metrics for MCE
    eps = 1e-12
    nll_mce = float(-np.mean(np.sum(human_p * np.log(np.clip(mce_probs, eps, 1.0)), axis=1)))
    brier_mce = float(np.mean(np.sum((mce_probs - human_p) ** 2, axis=1)))
    acc_mce = float(np.mean(np.argmax(mce_probs, axis=1) == np.argmax(human_p, axis=1)))

    print(f"\n4. Pointwise Evaluation (Gemma 3 12B MCE):")
    print(f"   Negative Log-Likelihood (NLL): {nll_mce:.4f}")
    print(f"   Brier Score:                   {brier_mce:.4f}")
    print(f"   Majority Gold Accuracy:        {acc_mce * 100:.2f}%")

    # 5. Relational Topology for MCE
    s_human_path = PILOT_SUPPORT_DIR / "S_hellinger_k010_pilot.bin"
    s_human_manifest = PILOT_SUPPORT_DIR / "S_hellinger_k010_pilot.manifest.json"

    with open(s_human_manifest, "r", encoding="utf-8") as f:
        meta = json.load(f)
    q_hh = meta.get("q_hh_relational", 0.26338)

    S_human = np.frombuffer(s_human_path.read_bytes(), dtype=np.float32).reshape(N, N).astype(np.float64)

    D_mce = distance_hellinger_matrix(mce_probs, mce_probs)
    W_mce = compute_topk_weight_matrix(D_mce, k=10)
    q_supp_mce = float(np.sum(W_mce * S_human) / (N * 10.0))

    q_null_strat_mce = compute_dataset_stratified_null(W_mce, S_human, ds_ids, k=10)
    r_norm_mce = (q_supp_mce - q_null_strat_mce) / (q_hh - q_null_strat_mce) * 100.0

    print(f"\n5. Relational Performance (Gemma 3 12B MCE):")
    print(f"   Human Target Q_HH (k=10):     {q_hh:.5f}")
    print(f"   MCE Q_support:                {q_supp_mce:.5f}")
    print(f"   MCE Stratified Analytic Null: {q_null_strat_mce:.5f}")
    print(f"   MCE R_norm (Stratified Null): {r_norm_mce:.2f}%")

    # 6. Load LPE Audit Summary for Side-by-Side Comparison Table
    lpe_summary_path = SUMMARIES_DIR / "E004_gemma3_12b_lpe_extraction_audit.json"
    lpe_data = {}
    if lpe_summary_path.exists():
        with open(lpe_summary_path, "r", encoding="utf-8") as f:
            lpe_data = json.load(f)

    raw_lpe = lpe_data.get("raw_metrics", {})
    cal_lpe = lpe_data.get("calibrated_metrics", {})

    print(f"\n6. STAGE 1B GEMMA 3 12B FULL SIDE-BY-SIDE BENCHMARK TABLE")
    print("=" * 88)
    print(f"   {'Method / Condition':<30} | {'Accuracy':<10} | {'NLL':<8} | {'Brier':<8} | {'Q_supp':<8} | {'R_norm (%)':<10}")
    print("   " + "-" * 84)
    print(f"   {'Raw LPE (Zero-Shot)':<30} | {raw_lpe.get('gold_accuracy', 0)*100:<9.2f}% | {raw_lpe.get('nll', 0):<8.4f} | {raw_lpe.get('brier_score', 0):<8.4f} | {raw_lpe.get('q_support', 0):<8.5f} | {raw_lpe.get('r_norm_stratified_pct', 0):<10.2f}%")
    print(f"   {'Calibrated LPE (T* = 10.48)':<30} | {cal_lpe.get('gold_accuracy', 0)*100:<9.2f}% | {cal_lpe.get('nll', 0):<8.4f} | {cal_lpe.get('brier_score', 0):<8.4f} | {cal_lpe.get('q_support', 0):<8.5f} | {cal_lpe.get('r_norm_stratified_pct', 0):<10.2f}%")
    print(f"   {'MCE (30 Samples, T = 1.0)':<30} | {acc_mce*100:<9.2f}% | {nll_mce:<8.4f} | {brier_mce:<8.4f} | {q_supp_mce:<8.5f} | {r_norm_mce:<10.2f}%")
    print("=" * 88)

    # 7. Save MCE Summary Artifact
    summary_out = SUMMARIES_DIR / "E004_gemma3_12b_mce_extraction_audit.json"
    audit_summary = {
        "model_tag": "gemma3:12b",
        "prompt_version": "v2",
        "symbol_set": "ABC",
        "num_items": N,
        "total_raw_records": len(raw_records),
        "successful_records": len(success_records),
        "invalid_output_records": len(invalid_records),
        "valid_sample_rate": (len(success_records) - len(invalid_records)) / len(success_records),
        "smoothing": "Jeffreys (n_c + 0.5) / (30 + 1.5)",
        "metrics": {
            "nll": nll_mce,
            "brier_score": brier_mce,
            "gold_accuracy": acc_mce,
            "q_support": q_supp_mce,
            "q_null_stratified": q_null_strat_mce,
            "q_hh": q_hh,
            "r_norm_stratified_pct": r_norm_mce,
        },
        "comparison_table": {
            "raw_lpe": raw_lpe,
            "calibrated_lpe": cal_lpe,
            "mce": {
                "gold_accuracy": acc_mce,
                "nll": nll_mce,
                "brier_score": brier_mce,
                "q_support": q_supp_mce,
                "q_null_stratified": q_null_strat_mce,
                "r_norm_stratified_pct": r_norm_mce,
            },
        },
        "timestamp_utc": "2026-08-03T23:57:00Z",
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nSaved audited MCE summary to: {summary_out}")


if __name__ == "__main__":
    main()
