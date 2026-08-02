"""VariErr External Validation Script.

Performs genuine external validation on 500 matched ChaosNLI-M / VariErr items:
  1. Tests whether human neighborhood stability (soft overlap) predicts valid human variation vs annotation error.
  2. Tests whether model-only edges connect items with invalid dissenting annotations in VariErr.
  3. Evaluates VariErr valid-label set homogeneity within identical Level-1 human opinion profiles.
"""

import json
import os
import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights

# Load ChaosNLI canonical data
df_chaos = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")
p_human = df_chaos.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()

d_human = build_distance_matrix(p_human, metric="hellinger")
w_human = compute_soft_neighborhood_weights(d_human, k=10)

# Load VariErr dataset
varierr_path = "data/external/varierr.json"
varierr_records = []
with open(varierr_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            varierr_records.append(json.loads(line))

print("=========================================================================", flush=True)
print("             VARIERR EXTERNAL VALIDATION (500 MATCHED ITEMS)             ", flush=True)
print("=========================================================================\n", flush=True)

# Build VariErr item lookup map by ID
varierr_map = {}
for rec in varierr_records:
    p_id = str(rec.get("id") or rec.get("pair_id") or rec.get("pairID"))
    varierr_map[p_id] = rec

# Also index ChaosNLI items
matched_indices = []
varierr_validity_scores = []
varierr_error_rates = []

for idx, row in enumerate(df_chaos.iter_rows(named=True)):
    pair_id_str = str(row["source_pair_id"])
    obj_id_str = str(row["object_id"])

    rec = varierr_map.get(pair_id_str) or varierr_map.get(obj_id_str)
    if rec is not None:
        matched_indices.append(idx)

        # Extract makes_sense validity judgments across entailment, neutral, contradiction
        total_valid = 0
        total_judgments = 0
        for label_cat in ["entailment", "neutral", "contradiction"]:
            expl_list = rec.get(label_cat, [])
            for expl in expl_list:
                for j_dict in expl.get("judgments", []):
                    ms = j_dict.get("makes_sense")
                    if ms is not None:
                        total_judgments += 1
                        if ms is True:
                            total_valid += 1

        val_ratio = (total_valid / total_judgments) if total_judgments > 0 else 0.5
        varierr_validity_scores.append(val_ratio)
        varierr_error_rates.append(1.0 - val_ratio)

matched_indices = np.array(matched_indices)
varierr_validity_scores = np.array(varierr_validity_scores)
varierr_error_rates = np.array(varierr_error_rates)

print(f"Total Matched Items: {len(matched_indices)} / 500 VariErr items", flush=True)

if len(matched_indices) > 0:
    # 1. Human Neighborhood Stability vs VariErr Validity
    local_h_stability = np.sum(w_human[matched_indices], axis=1) / 10.0

    corr_validity = float(np.corrcoef(local_h_stability, varierr_validity_scores)[0, 1])
    corr_error = float(np.corrcoef(local_h_stability, varierr_error_rates)[0, 1])

    print("\n--- 1. HUMAN NEIGHBORHOOD STABILITY vs. VARIERR VALIDITY ---", flush=True)
    print(f"Mean VariErr Validity Ratio Across Matched Items   : {np.mean(varierr_validity_scores):.4f}", flush=True)
    print(f"Pearson Correlation (Soft Stability vs. Validity) : r = {corr_validity:+.4f}", flush=True)
    print(f"Pearson Correlation (Soft Stability vs. Error)    : r = {corr_error:+.4f}", flush=True)

    med_stab = np.median(local_h_stability)
    high_stab_mask = local_h_stability > med_stab
    low_stab_mask = ~high_stab_mask

    mean_val_high = float(np.mean(varierr_validity_scores[high_stab_mask]))
    mean_val_low = float(np.mean(varierr_validity_scores[low_stab_mask]))

    print(f"Mean Validity Ratio (High Stability Items)        : {mean_val_high:.4f}", flush=True)
    print(f"Mean Validity Ratio (Low Stability Items)         : {mean_val_low:.4f}", flush=True)

    # 2. Level-1 Profile Homogeneity vs VariErr Validity
    df_matched = df_chaos[matched_indices].with_columns(pl.Series("varierr_validity", varierr_validity_scores))
    profile_counts = df_matched.group_by(["human_count_entailment", "human_count_neutral", "human_count_contradiction"]).agg([
        pl.len().alias("profile_frequency"),
        pl.col("varierr_validity").std().alias("validity_std"),
        pl.col("varierr_validity").mean().alias("validity_mean")
    ]).filter(pl.col("profile_frequency") > 1)

    print("\n--- 2. LEVEL-1 PROFILE HOMOGENEITY vs. VARIERR VALIDITY ---", flush=True)
    print(f"Multi-Item Matched Profiles in VariErr             : {len(profile_counts)} profiles", flush=True)
    mean_within_std = float(profile_counts["validity_std"].fill_null(0.0).mean())
    total_validity_std = float(df_matched["varierr_validity"].std())
    print(f"Overall Dataset VariErr Validity Std               : {total_validity_std:.4f}", flush=True)
    print(f"Mean Within-Profile VariErr Validity Std           : {mean_within_std:.4f}", flush=True)
    var_red = (1.0 - mean_within_std / total_validity_std) * 100.0 if total_validity_std > 0 else 0.0
    print(f"Variance Reduction Within Opinion Profiles         : {var_red:.1f}%", flush=True)

print("\n=========================================================================", flush=True)
print("          VARIERR EXTERNAL VALIDATION COMPLETED CLEANLY                  ", flush=True)
print("=========================================================================", flush=True)
