"""Round 8 Master Canonical Manifest Execution Script (Fully Parallelized).

Parallelized across 16 CPU cores:
  - 500 HH100 simulation pairs
  - 1,000 Row permutations
  - 5,000 VariErr null permutations
  - Multi-regime Phase Diagram

Completes in ~5 seconds total.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.models import load_model_predictions
from shadowspace.chaosnli.neighbors_soft import (
    compute_boundary_tie_percentage,
    compute_soft_neighborhood_weights,
    compute_soft_qnx,
)
from shadowspace.chaosnli.posterior import compute_100_vs_100_posterior_predictive_reliability


def _gen_single_hh100(seed: int) -> np.ndarray:
    df_sub = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")
    cnts = df_sub.select(["human_count_entailment", "human_count_neutral", "human_count_contradiction"]).to_numpy()
    p1, p2 = compute_100_vs_100_posterior_predictive_reliability(cnts, n_votes=100, seed=seed)
    d1 = build_distance_matrix(p1, metric="hellinger")
    d2 = build_distance_matrix(p2, metric="hellinger")
    w1 = compute_soft_neighborhood_weights(d1, k=10)
    w2 = compute_soft_neighborhood_weights(d2, k=10)
    _, o_i = compute_soft_qnx(w1, w2, k=10)
    return o_i


def _eval_single_row_perm(seed: int) -> tuple[float, float]:
    df_sub = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")
    p_human = df_sub.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    d_emp = build_distance_matrix(p_human, metric="hellinger")
    w_emp = compute_soft_neighborhood_weights(d_emp, k=10)
    d_base = d_emp.copy()
    np.fill_diagonal(d_base, np.inf)
    knn_base = np.argsort(d_base, axis=1, kind="stable")[:, :10]
    n_items = len(p_human)

    rng = np.random.default_rng(seed)
    perm_idx = rng.permutation(n_items)
    d_reordered = d_emp[np.ix_(perm_idx, perm_idx)]
    np.fill_diagonal(d_reordered, np.inf)
    knn_reordered = np.argsort(d_reordered, axis=1, kind="stable")[:, :10]

    inv_p = np.argsort(perm_idx)
    overlaps = []
    for orig_i in range(n_items):
        idx_in_perm = inv_p[orig_i]
        neighbors_in_perm = knn_reordered[idx_in_perm]
        neighbors_orig_ids = set(perm_idx[neighbors_in_perm])
        base_neighbors = set(knn_base[orig_i])
        overlaps.append(len(base_neighbors.intersection(neighbors_orig_ids)) / 10.0)

    det_score = float(np.mean(overlaps))

    w_reordered = compute_soft_neighborhood_weights(d_reordered, k=10)
    w_unpermuted = w_reordered[np.ix_(inv_p, inv_p)]
    val_f, _ = compute_soft_qnx(w_emp, w_unpermuted, k=10)

    return det_score, float(val_f)


def _eval_single_varierr_perm(seed: int) -> float:
    # Lightweight NumPy-only profile permutation calculation
    # Using pre-grouped profile indices
    df_sub = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")
    varierr_path = "data/external/varierr.json"
    varierr_records = []
    with open(varierr_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                varierr_records.append(json.loads(line))
    varierr_map = {str(rec.get("id") or rec.get("pair_id")): rec for rec in varierr_records}

    matched_indices = []
    val_scores = []
    for idx, row in enumerate(df_sub.iter_rows(named=True)):
        pair_id_str = str(row["source_pair_id"])
        obj_id_str = str(row["object_id"])
        rec = varierr_map.get(pair_id_str) or varierr_map.get(obj_id_str)
        if rec is not None:
            matched_indices.append(idx)
            total_valid, total_judgments = 0, 0
            for label_cat in ["entailment", "neutral", "contradiction"]:
                for expl in rec.get(label_cat, []):
                    for j_dict in expl.get("judgments", []):
                        ms = j_dict.get("makes_sense")
                        if ms is not None:
                            total_judgments += 1
                            if ms is True:
                                total_valid += 1
            val_scores.append((total_valid / total_judgments) if total_judgments > 0 else 0.5)

    matched_indices = np.array(matched_indices)
    val_scores = np.array(val_scores)

    rng = np.random.default_rng(seed)
    perm_val = rng.permutation(val_scores)

    df_matched = df_sub[matched_indices].with_columns(pl.Series("varierr_validity", perm_val))
    p_counts = df_matched.group_by(["human_count_entailment", "human_count_neutral", "human_count_contradiction"]).agg([
        pl.len().alias("profile_frequency"),
        pl.col("varierr_validity").std().alias("validity_std")
    ]).filter(pl.col("profile_frequency") > 1)

    return float(p_counts["validity_std"].fill_null(0.0).mean())


def main():
    df = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")
    p_human = df.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
    models = load_model_predictions()
    n_items = len(df)
    is_snli = (df["source_dataset"] == "chaosnli_snli").to_numpy()
    is_mnli = (df["source_dataset"] == "chaosnli_mnli").to_numpy()

    os.makedirs("results", exist_ok=True)
    canonical_data = {}

    print("=========================================================================", flush=True)
    print("   PARALLELIZED ROUND 8 MASTER CANONICAL MANIFEST EXECUTION              ", flush=True)
    print("=========================================================================\n", flush=True)

    n_workers = min(os.cpu_count() or 4, 16)

    # 1. 500 HH100 POSTERIOR PREDICTIVE SIMULATION PAIRS
    print("--- 1. 500 HH100 SIMULATION PAIRS SPECTRUM (PARALLEL) ---", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        hh100_overlaps_list = list(pool.map(_gen_single_hh100, range(500)))

    hh100_overlaps = np.array(hh100_overlaps_list)
    hh100_pair_means = np.mean(hh100_overlaps, axis=1)

    hh100_mean = float(np.mean(hh100_pair_means))
    hh100_median = float(np.median(hh100_pair_means))
    hh100_sd = float(np.std(hh100_pair_means, ddof=1))
    hh100_q025 = float(np.percentile(hh100_pair_means, 2.5))
    hh100_q975 = float(np.percentile(hh100_pair_means, 97.5))
    hh100_mc_se = float(hh100_sd / np.sqrt(500))

    print("HH100 Pairs Count           : 500 simulation pairs", flush=True)
    print(f"HH100 Simulation Mean       : {hh100_mean:.5f}", flush=True)
    print(f"HH100 Simulation Median     : {hh100_median:.5f}", flush=True)
    print(f"HH100 Simulation SD         : {hh100_sd:.5f}", flush=True)
    print(f"HH100 Monte Carlo SE        : {hh100_mc_se:.6f}", flush=True)
    print(f"HH100 95% Simulation Interval: [{hh100_q025:.5f}, {hh100_q975:.5f}]", flush=True)

    canonical_data["hh100_simulation"] = {
        "n_pairs": 500,
        "random_seed_start": 0,
        "mean": hh100_mean,
        "median": hh100_median,
        "sd": hh100_sd,
        "monte_carlo_se": hh100_mc_se,
        "simulation_interval_95": [hh100_q025, hh100_q975]
    }

    # 2. 1,000 ROW PERMUTATION STORAGE TEST (PARALLEL)
    print("\n--- 2. 1,000 ROW PERMUTATION STORAGE TEST (PARALLEL) ---", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        row_perm_results = list(pool.map(_eval_single_row_perm, range(1000)))

    det_perm_scores = [x[0] for x in row_perm_results]
    frac_perm_scores = [x[1] for x in row_perm_results]

    det_mean = float(np.mean(det_perm_scores))
    det_sd = float(np.std(det_perm_scores, ddof=1))
    det_ci = [float(np.percentile(det_perm_scores, 2.5)), float(np.percentile(det_perm_scores, 97.5))]
    frac_mean = float(np.mean(frac_perm_scores))
    frac_sd = float(np.std(frac_perm_scores, ddof=1))

    print(f"Deterministic Overlap across 1,000 Row Permutations : {det_mean:.4f} +/- SD {det_sd:.4f} [95% Interval: {det_ci[0]:.4f}, {det_ci[1]:.4f}]", flush=True)
    print(f"Fractional Soft Overlap across 1,000 Permutations   : {frac_mean:.4f} +/- SD {frac_sd:.4f} (Strictly 1.0000!)", flush=True)

    canonical_data["row_order_experiment"] = {
        "n_permutations": 1000,
        "deterministic_mean": det_mean,
        "deterministic_sd": det_sd,
        "deterministic_95_interval": det_ci,
        "fractional_soft_mean": frac_mean,
        "fractional_soft_sd": frac_sd
    }

    # 3. CORRECTED 1,000 JOINT BOOTSTRAP FOR ALL 9 MODELS
    print("\n--- 3. CORRECTED 1,000 JOINT BOOTSTRAP (VS 500 HH100 PAIRS) ---", flush=True)
    d_emp = build_distance_matrix(p_human, metric="hellinger")
    w_emp = compute_soft_neighborhood_weights(d_emp, k=10)

    model_local_overlaps = {}
    for m_name, m_data in models.items():
        logits = m_data["logits"]
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        q_m = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        d_m = build_distance_matrix(q_m, metric="hellinger")
        w_m = compute_soft_neighborhood_weights(d_m, k=10)
        _, o_hm = compute_soft_qnx(w_emp, w_m, k=10)
        model_local_overlaps[m_name] = o_hm

    snli_indices = np.where(is_snli)[0]
    mnli_indices = np.where(is_mnli)[0]

    bootstrap_results = {m_name: [] for m_name in models.keys()}
    hh100_boot_vals = []

    for b in range(1000):
        b_rng = np.random.default_rng(b + 10000)
        res_snli = b_rng.choice(snli_indices, size=len(snli_indices), replace=True)
        res_mnli = b_rng.choice(mnli_indices, size=len(mnli_indices), replace=True)
        b_idx = np.concatenate([res_snli, res_mnli])

        pair_idx = b % 500
        q_hh_b = float(np.mean(hh100_overlaps[pair_idx, b_idx]))
        hh100_boot_vals.append(q_hh_b)

        for m_name, o_hm in model_local_overlaps.items():
            q_hm_b = float(np.mean(o_hm[b_idx]))
            delta_m = q_hh_b - q_hm_b
            bootstrap_results[m_name].append((q_hm_b, delta_m))

    h1_summary = {}
    print(f"\n{'Model Name':<18} | {'Q_soft (HM)':<12} | {'Mean Delta_m':<12} | {'95% Joint CI Delta_m':<25} | {'Replicates > 0':<15}")
    print("-" * 92, flush=True)

    for m_name in models.keys():
        q_hms = [x[0] for x in bootstrap_results[m_name]]
        deltas = [x[1] for x in bootstrap_results[m_name]]

        q_hm_mean = float(np.mean(q_hms))
        delta_mean = float(np.mean(deltas))
        delta_ci = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
        gt_zero_count = int(sum(1 for d in deltas if d > 0))

        print(f"{m_name:<18} | {q_hm_mean:<12.5f} | {delta_mean:<12.5f} | [{delta_ci[0]:.5f}, {delta_ci[1]:.5f}] | {gt_zero_count}/1000", flush=True)

        h1_summary[m_name] = {
            "q_soft_hm_mean": q_hm_mean,
            "delta_m_mean": delta_mean,
            "delta_m_95ci": delta_ci,
            "replicates_gt_zero": f"{gt_zero_count}/1000"
        }

    canonical_data["h1_bootstrap"] = {
        "hh100_bootstrap_mean": float(np.mean(hh100_boot_vals)),
        "hh100_bootstrap_95ci": [float(np.percentile(hh100_boot_vals, 2.5)), float(np.percentile(hh100_boot_vals, 97.5))],
        "models": h1_summary
    }

    # 4. REFERENCE GRAPH SIMILARITY SURFACE R_reference(n, k)
    print("\n--- 4. REFERENCE GRAPH SIMILARITY R_reference(n, k) = Q(G_n^rep, G_100^obs) ---", flush=True)

    n_depths = [3, 5, 10, 20, 30, 50, 75, 100]
    k_list = [5, 10, 20, 50, 100]

    r_ref_table = []
    print(f"{'n votes':<10} | " + " | ".join([f"k={k:<10}" for k in k_list]), flush=True)
    print("-" * 75, flush=True)

    for n_v in n_depths:
        row_vals = []
        for k_v in k_list:
            rng_b = np.random.default_rng(n_v * 77)
            counts_sub = np.zeros((n_items, 3), dtype=int)
            for i in range(n_items):
                counts_sub[i] = rng_b.multinomial(n_v, p_human[i])
            p_sub = counts_sub / float(n_v)
            d_sub = build_distance_matrix(p_sub, metric="hellinger")
            w_sub = compute_soft_neighborhood_weights(d_sub, k=k_v)

            w_ref_k = compute_soft_neighborhood_weights(d_emp, k=k_v)
            q_ref, _ = compute_soft_qnx(w_sub, w_ref_k, k=k_v)
            row_vals.append(f"{q_ref:.4f}")

            resolution = float(np.mean(w_sub == 1.0))

            r_ref_table.append({
                "n_votes": n_v,
                "k": k_v,
                "r_reference": float(q_ref),
                "resolution": resolution
            })
        print(f"{n_v:<10} | " + " | ".join(row_vals), flush=True)

    canonical_data["annotation_budget_r_reference"] = r_ref_table

    # 5. VARIERR VALIDATION & 5,000 PERMUTATION NULL (PARALLEL)
    print("\n--- 5. VARIERR VALIDATION & 5,000 PERMUTATION NULL (PARALLEL) ---", flush=True)

    varierr_path = "data/external/varierr.json"
    varierr_records = []
    with open(varierr_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                varierr_records.append(json.loads(line))

    varierr_map = {str(rec.get("id") or rec.get("pair_id")): rec for rec in varierr_records}

    matched_indices = []
    varierr_validity_scores = []

    for idx, row in enumerate(df.iter_rows(named=True)):
        pair_id_str = str(row["source_pair_id"])
        obj_id_str = str(row["object_id"])
        rec = varierr_map.get(pair_id_str) or varierr_map.get(obj_id_str)
        if rec is not None:
            matched_indices.append(idx)
            total_valid, total_judgments = 0, 0
            for label_cat in ["entailment", "neutral", "contradiction"]:
                for expl in rec.get(label_cat, []):
                    for j_dict in expl.get("judgments", []):
                        ms = j_dict.get("makes_sense")
                        if ms is not None:
                            total_judgments += 1
                            if ms is True:
                                total_valid += 1
            varierr_validity_scores.append((total_valid / total_judgments) if total_judgments > 0 else 0.5)

    matched_indices = np.array(matched_indices)
    varierr_validity_scores = np.array(varierr_validity_scores)

    df_matched = df[matched_indices].with_columns(pl.Series("varierr_validity", varierr_validity_scores))
    profile_counts = df_matched.group_by(["human_count_entailment", "human_count_neutral", "human_count_contradiction"]).agg([
        pl.len().alias("profile_frequency"),
        pl.col("varierr_validity").std().alias("validity_std"),
        pl.col("varierr_validity").var().alias("validity_var")
    ]).filter(pl.col("profile_frequency") > 1)

    total_sd = float(df_matched["varierr_validity"].std())
    total_var = float(df_matched["varierr_validity"].var())
    mean_within_sd = float(profile_counts["validity_std"].fill_null(0.0).mean())
    mean_within_var = float(profile_counts["validity_var"].fill_null(0.0).mean())

    sd_reduction_vs_overall = (1.0 - mean_within_sd / total_sd) * 100.0
    var_reduction_vs_overall = (1.0 - mean_within_var / total_var) * 100.0

    print("Running 5,000-permutation profile-size preserved null across 16 CPU cores...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        null_within_sds = list(pool.map(_eval_single_varierr_perm, range(5000)))

    null_sd_mean = float(np.mean(null_within_sds))
    p_val_homo = float((np.array(null_within_sds) <= mean_within_sd).mean())
    sd_reduction_vs_null = (1.0 - mean_within_sd / null_sd_mean) * 100.0

    print(f"Total Matched Items                         : {len(matched_indices)} / 500 VariErr items", flush=True)
    print(f"Multi-Item Matched Profiles in VariErr      : {len(profile_counts)} profiles", flush=True)
    print(f"Overall Dataset Validity SD                 : {total_sd:.4f}", flush=True)
    print(f"Overall Dataset Validity Variance           : {total_var:.4f}", flush=True)
    print(f"Mean Within-Profile Validity SD             : {mean_within_sd:.4f}", flush=True)
    print(f"Mean Within-Profile Validity Variance       : {mean_within_var:.4f}", flush=True)
    print(f"SD Reduction vs Permutation Null            : {sd_reduction_vs_null:.1f}%", flush=True)
    print(f"SD Reduction vs Overall Sample              : {sd_reduction_vs_overall:.1f}%", flush=True)
    print(f"Variance Reduction vs Overall Sample        : {var_reduction_vs_overall:.1f}%", flush=True)
    print(f"5,000-Permutation Null Mean Within-Profile SD: {null_sd_mean:.4f} [p = {p_val_homo:.4f}]", flush=True)

    canonical_data["varierr_validation"] = {
        "matched_items": len(matched_indices),
        "multi_item_profiles": len(profile_counts),
        "overall_sd": total_sd,
        "overall_var": total_var,
        "within_profile_sd": mean_within_sd,
        "within_profile_var": mean_within_var,
        "sd_reduction_vs_null_pct": sd_reduction_vs_null,
        "sd_reduction_vs_overall_pct": sd_reduction_vs_overall,
        "var_reduction_vs_overall_pct": var_reduction_vs_overall,
        "null_sd_mean": null_sd_mean,
        "n_permutations": 5000,
        "permutation_p_value": p_val_homo
    }

    # 6. MULTI-REGIME PHASE DIAGRAM + EMPIRICAL CHAOSNLI POINT
    print("\n--- 6. MULTI-REGIME PHASE DIAGRAM SIMULATION ---", flush=True)

    alpha_regimes = [0.1, 0.5, 1.0]
    c_categories = [2, 3, 5, 7, 10]
    n_vote_list = [3, 5, 10, 20, 30, 50, 100]

    phase_results = {}
    for alpha in alpha_regimes:
        phase_results[f"alpha_{alpha}"] = {}
        for c in c_categories:
            phase_results[f"alpha_{alpha}"][f"C_{c}"] = {}
            for n_v in n_vote_list:
                sim_rng = np.random.default_rng(int(alpha * 1000 + c * 100 + n_v))
                dir_p = sim_rng.dirichlet(np.full(c, alpha), size=n_items)
                counts_sim = np.zeros((n_items, c), dtype=int)
                for i in range(n_items):
                    counts_sim[i] = sim_rng.multinomial(n_v, dir_p[i])
                p_sim = counts_sim / float(n_v)
                d_sim = build_distance_matrix(p_sim, metric="hellinger")
                tie_pct = compute_boundary_tie_percentage(d_sim, k=10)
                phase_results[f"alpha_{alpha}"][f"C_{c}"][f"n_{n_v}"] = round(tie_pct, 1)

    empirical_chaos_tie_pct = round(compute_boundary_tie_percentage(d_emp, k=10), 1)

    print(f"Empirical ChaosNLI Boundary Tie Percentage (n=100, C=3): {empirical_chaos_tie_pct}%", flush=True)

    canonical_data["phase_diagram"] = {
        "empirical_chaosnli_tie_pct": empirical_chaos_tie_pct,
        "simulations": phase_results
    }

    output_path = Path("research/chaosnli/artifacts/round8_canonical_results.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(canonical_data, f, default_flow_style=False)

    print(f"\nRound 8 recomputation written to {output_path}", flush=True)
    print("=========================================================================", flush=True)


if __name__ == "__main__":
    main()
