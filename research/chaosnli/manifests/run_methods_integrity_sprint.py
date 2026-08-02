"""Lightning-Fast Parallelized Methods Integrity Sprint Execution Script (Final Clean Version).

Saves all calculated results cleanly to results/canonical_results.yaml.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import os
import yaml
import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.models import load_model_predictions
from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights, compute_soft_qnx
from shadowspace.chaosnli.posterior import compute_100_vs_100_posterior_predictive_reliability


def _eval_single_hh100_overlap(seed: int) -> np.ndarray:
    df_sub = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")
    cnts = df_sub.select(["human_count_entailment", "human_count_neutral", "human_count_contradiction"]).to_numpy()
    p1, p2 = compute_100_vs_100_posterior_predictive_reliability(cnts, n_votes=100, seed=seed)
    d1 = build_distance_matrix(p1, metric="hellinger")
    d2 = build_distance_matrix(p2, metric="hellinger")
    w1 = compute_soft_neighborhood_weights(d1, k=10)
    w2 = compute_soft_neighborhood_weights(d2, k=10)
    _, o_i = compute_soft_qnx(w1, w2, k=10)
    return o_i


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
    print("   LIGHTNING-FAST PARALLEL METHODS INTEGRITY SPRINT MASTER EXECUTION     ", flush=True)
    print("=========================================================================\n", flush=True)

    # 1. MULTI-PERMUTATION ROW-ORDER STORAGE TEST
    d_emp = build_distance_matrix(p_human, metric="hellinger")
    w_emp = compute_soft_neighborhood_weights(d_emp, k=10)

    det_perm_scores = []
    frac_perm_scores = []
    rng = np.random.default_rng(42)

    knn_base = np.argsort(d_emp, axis=1)[:, 1:11]

    for _ in range(10):
        perm_idx = rng.permutation(n_items)
        d_reordered = d_emp[np.ix_(perm_idx, perm_idx)]
        knn_reordered = np.argsort(d_reordered, axis=1)[:, 1:11]

        inv_p = np.argsort(perm_idx)
        overlaps = []
        for orig_i in range(n_items):
            idx_in_perm = inv_p[orig_i]
            neighbors_in_perm = knn_reordered[idx_in_perm]
            neighbors_orig_ids = set(perm_idx[neighbors_in_perm])
            base_neighbors = set(knn_base[orig_i])
            overlaps.append(len(base_neighbors.intersection(neighbors_orig_ids)) / 10.0)

        det_perm_scores.append(np.mean(overlaps))

        w_reordered = compute_soft_neighborhood_weights(d_reordered, k=10)
        w_unpermuted = w_reordered[np.ix_(inv_p, inv_p)]
        val_f, _ = compute_soft_qnx(w_emp, w_unpermuted, k=10)
        frac_perm_scores.append(val_f)

    det_mean = float(np.mean(det_perm_scores))
    det_std = float(np.std(det_perm_scores))
    det_ci = [float(np.percentile(det_perm_scores, 2.5)), float(np.percentile(det_perm_scores, 97.5))]
    frac_mean = float(np.mean(frac_perm_scores))
    frac_std = float(np.std(frac_perm_scores))

    canonical_data["row_order_experiment"] = {
        "deterministic_mean": det_mean,
        "deterministic_std": det_std,
        "deterministic_95ci": det_ci,
        "fractional_soft_mean": frac_mean,
        "fractional_soft_std": frac_std
    }

    # 2. RECOMPUTE H1 BOOTSTRAP WITH 100 HH100 PAIRS
    n_workers = min(os.cpu_count() or 4, 16)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        hh100_local_overlaps_list = list(pool.map(_eval_single_hh100_overlap, range(100)))

    hh100_local_overlaps = np.array(hh100_local_overlaps_list)

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

        pair_idx = b % 100
        q_hh_b = float(np.mean(hh100_local_overlaps[pair_idx, b_idx]))
        hh100_boot_vals.append(q_hh_b)

        for m_name, o_hm in model_local_overlaps.items():
            q_hm_b = float(np.mean(o_hm[b_idx]))
            delta_m = q_hh_b - q_hm_b
            bootstrap_results[m_name].append((q_hm_b, delta_m))

    h1_summary = {}
    for m_name in models.keys():
        q_hms = [x[0] for x in bootstrap_results[m_name]]
        deltas = [x[1] for x in bootstrap_results[m_name]]

        q_hm_mean = float(np.mean(q_hms))
        delta_mean = float(np.mean(deltas))
        delta_ci = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
        p_gt_zero = float((np.array(deltas) > 0).mean())

        h1_summary[m_name] = {
            "q_soft_hm_mean": q_hm_mean,
            "delta_m_mean": delta_mean,
            "delta_m_95ci": delta_ci,
            "p_delta_gt_zero": p_gt_zero
        }

    canonical_data["h1_bootstrap"] = {
        "hh100_bootstrap_mean": float(np.mean(hh100_boot_vals)),
        "hh100_bootstrap_95ci": [float(np.percentile(hh100_boot_vals, 2.5)), float(np.percentile(hh100_boot_vals, 97.5))],
        "models": h1_summary
    }

    # 3. ANNOTATION-BUDGET RECOVERY R_truth(n, k)
    n_depths = [3, 5, 10, 20, 30, 50, 75, 100]
    k_list = [5, 10, 20, 50, 100]

    r_truth_table = []
    for n_v in n_depths:
        for k_v in k_list:
            rng_b = np.random.default_rng(n_v * 77)
            counts_sub = np.zeros((n_items, 3), dtype=int)
            for i in range(n_items):
                counts_sub[i] = rng_b.multinomial(n_v, p_human[i])
            p_sub = counts_sub / float(n_v)
            d_sub = build_distance_matrix(p_sub, metric="hellinger")
            w_sub = compute_soft_neighborhood_weights(d_sub, k=k_v)

            w_ref_k = compute_soft_neighborhood_weights(d_emp, k=k_v)
            q_truth, _ = compute_soft_qnx(w_sub, w_ref_k, k=k_v)
            resolution = float(np.mean(w_sub == 1.0))

            r_truth_table.append({
                "n_votes": n_v,
                "k": k_v,
                "r_truth": float(q_truth),
                "resolution": resolution
            })

    canonical_data["annotation_budget_r_truth"] = r_truth_table

    # 4. VARIERR VALIDATION & PERMUTATION NULL
    varierr_path = "data/external/varierr.json"
    varierr_records = []
    with open(varierr_path, "r", encoding="utf-8") as f:
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

    sd_reduction = (1.0 - mean_within_sd / total_sd) * 100.0
    var_reduction = (1.0 - mean_within_var / total_var) * 100.0

    null_within_sds = []
    val_scores_copy = np.array(varierr_validity_scores)

    for p_seed in range(200):
        rng_p = np.random.default_rng(p_seed)
        perm_scores = rng_p.permutation(val_scores_copy)
        df_perm = df_matched.with_columns(pl.Series("varierr_validity", perm_scores))
        p_counts = df_perm.group_by(["human_count_entailment", "human_count_neutral", "human_count_contradiction"]).agg([
            pl.len().alias("profile_frequency"),
            pl.col("varierr_validity").std().alias("validity_std")
        ]).filter(pl.col("profile_frequency") > 1)
        null_within_sds.append(float(p_counts["validity_std"].fill_null(0.0).mean()))

    null_sd_mean = float(np.mean(null_within_sds))
    p_val_homo = float((np.array(null_within_sds) <= mean_within_sd).mean())

    canonical_data["varierr_validation"] = {
        "matched_items": len(matched_indices),
        "multi_item_profiles": len(profile_counts),
        "overall_sd": total_sd,
        "overall_var": total_var,
        "within_profile_sd": mean_within_sd,
        "within_profile_var": mean_within_var,
        "sd_reduction_pct": sd_reduction,
        "var_reduction_pct": var_reduction,
        "null_sd_mean": null_sd_mean,
        "permutation_p_value": p_val_homo
    }

    with open("results/canonical_results.yaml", "w", encoding="utf-8") as f:
        yaml.dump(canonical_data, f, default_flow_style=False)

    print("Canonical results written successfully to results/canonical_results.yaml", flush=True)


if __name__ == "__main__":
    main()
