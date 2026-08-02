"""
canonical_values_audit.py
==========================
Definitive re-computation of all canonical values used in the paper.
Resolves:
  1. True H_bar: Q_fuzzy mean over 500 HH100 pairs (also Q_strict, Q_expected)
  2. True M_bar_m: direct paired score mean per model from 500 pairs
  3. True direct delta: H_bar - M_bar_m for each model
  4. Geometry table: Q(G_m, G_emp) for all 9 models x 5 metrics
     (G_emp = observed empirical graph using p_human, NOT a single posterior draw)
  5. Clarifies: geometry table OLD value 0.01617 was Q(G_m, G_H1^seed42), i.e. single
     posterior draw. The correct fixed-reference is Q(G_m, G_emp) = 0.01867.

All outputs are printed and saved to results/canonical_audit.yaml.
"""

import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.models import load_model_predictions
from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights, compute_soft_qnx
from shadowspace.chaosnli.posterior import compute_100_vs_100_posterior_predictive_reliability

K = 10
N_PAIRS = 500
DATA_PATH = "data/chaosnli/processed/canonical_items_posterior.parquet"

print("=" * 72)
print("  CANONICAL VALUES AUDIT")
print("=" * 72)

# Load data
df = pl.read_parquet(DATA_PATH)
counts_full = df.select(["human_count_entailment", "human_count_neutral", "human_count_contradiction"]).to_numpy()
p_human = df.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
n_items = len(df)
print(f"Loaded {n_items} items")

# Build observed empirical graph (G_100^obs, using p_human directly)
d_emp = build_distance_matrix(p_human, metric="hellinger")
w_emp = compute_soft_neighborhood_weights(d_emp, k=K)
print(f"Built empirical graph w_emp (Hellinger, k={K})")

# Load models
models = load_model_predictions()
model_keys = sorted(models.keys())
model_weights = {}
for m_key in model_keys:
    logits = models[m_key]["logits"]
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    q_m = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    d_m = build_distance_matrix(q_m, metric="hellinger")
    w_m = compute_soft_neighborhood_weights(d_m, k=K)
    model_weights[m_key] = (q_m, w_m)
print(f"Loaded and precomputed {len(model_keys)} model graphs")

# ==========================================================================
# 1. HH100 PAIRS: Q_strict, Q_expected, Q_fuzzy over 500 pairs
# ==========================================================================
print(f"\n--- 1. HH100 PAIRS (N={N_PAIRS}) ---")
t0 = time.time()

hh_strict_vals = []
hh_expected_vals = []
hh_fuzzy_vals = []

# Per-model paired scores: direct mean over 500 pairs
model_paired_scores = {m: [] for m in model_keys}

for s in range(N_PAIRS):
    p1, p2 = compute_100_vs_100_posterior_predictive_reliability(counts_full, n_votes=100, seed=s)
    d1 = build_distance_matrix(p1, metric="hellinger")
    d2 = build_distance_matrix(p2, metric="hellinger")
    w_h1 = compute_soft_neighborhood_weights(d1, k=K)
    w_h2 = compute_soft_neighborhood_weights(d2, k=K)

    # Q_fuzzy (min-based) -- standard
    q_fuzzy, _ = compute_soft_qnx(w_h1, w_h2, k=K)
    hh_fuzzy_vals.append(q_fuzzy)

    # Q_expected (product-based: sum min(w_A*w_B) == sum w_A*w_B since w in [0,1])
    # Q_expected = 1/N * sum_i (1/k * sum_j w_ij^A * w_ij^B)
    q_expected_items = np.mean(np.sum(w_h1 * w_h2, axis=1)) / K
    hh_expected_vals.append(float(q_expected_items))

    # Q_strict (both weights == 1.0)
    strict_overlap = np.mean(np.sum((w_h1 == 1.0) & (w_h2 == 1.0), axis=1)) / K
    hh_strict_vals.append(float(strict_overlap))

    # Model paired scores for this pair
    for m_key in model_keys:
        _, w_m = model_weights[m_key]
        q_mh1, _ = compute_soft_qnx(w_h1, w_m, k=K)
        q_mh2, _ = compute_soft_qnx(w_h2, w_m, k=K)
        model_paired_scores[m_key].append(0.5 * (q_mh1 + q_mh2))

    if (s + 1) % 100 == 0:
        print(f"  Pair {s+1}/{N_PAIRS} done ({time.time()-t0:.1f}s)")

print(f"Completed {N_PAIRS} pairs in {time.time()-t0:.1f}s")

# Summarize
hh_strict = np.array(hh_strict_vals)
hh_expected = np.array(hh_expected_vals)
hh_fuzzy = np.array(hh_fuzzy_vals)

H_bar_strict = float(np.mean(hh_strict))
H_bar_expected = float(np.mean(hh_expected))
H_bar_fuzzy = float(np.mean(hh_fuzzy))

print("\nPanel B canonical values (direct 500-pair means):")
print(f"  Q_strict  direct mean: {H_bar_strict:.5f}  95% interval: [{np.percentile(hh_strict,2.5):.5f}, {np.percentile(hh_strict,97.5):.5f}]")
print(f"  Q_expected direct mean: {H_bar_expected:.5f}  95% interval: [{np.percentile(hh_expected,2.5):.5f}, {np.percentile(hh_expected,97.5):.5f}]")
print(f"  Q_fuzzy   direct mean: {H_bar_fuzzy:.5f}  95% interval: [{np.percentile(hh_fuzzy,2.5):.5f}, {np.percentile(hh_fuzzy,97.5):.5f}]")

# ==========================================================================
# 2. DIRECT DELTA = H_bar_fuzzy - M_bar_m (NOT bootstrap weighted)
# ==========================================================================
print("\n--- 2. DIRECT PAIRED SCORES AND DELTAS ---")
print(f"H_bar (Q_fuzzy direct mean) = {H_bar_fuzzy:.5f}")
print(f"\n{'Model':<20} {'M_bar_m':>10} {'Direct Delta':>14} {'Ratio raw%':>12} {'Ratio adj%':>12}")
print("-" * 72)

CHANCE = 0.00321  # k/(N-1) at k=10, N=3113
HH100_OBS = None  # will compute from Reference Ladder

direct_delta_table = {}
for m_key in model_keys:
    M_bar = float(np.mean(model_paired_scores[m_key]))
    delta_direct = H_bar_fuzzy - M_bar
    ratio_raw = M_bar / H_bar_fuzzy * 100
    ratio_adj = (M_bar - CHANCE) / (H_bar_fuzzy - CHANCE) * 100
    direct_delta_table[m_key] = {
        "M_bar_m": round(M_bar, 5),
        "direct_delta": round(delta_direct, 5),
        "ratio_raw_pct": round(ratio_raw, 1),
        "ratio_adj_pct": round(ratio_adj, 1),
    }
    print(f"{m_key:<20} {M_bar:>10.5f} {delta_direct:>14.5f} {ratio_raw:>11.1f}% {ratio_adj:>11.1f}%")

# ==========================================================================
# 3. GEOMETRY TABLE: Q(G_m, G_emp) for all 9 models x 5 metrics
#    G_emp = observed empirical human graph (using p_human directly)
#    NOT a single posterior draw -- that was the old (incorrect) method
# ==========================================================================
print("\n--- 3. GEOMETRY SENSITIVITY TABLE (Q(G_m, G_emp), k=10) ---")
print("NOTE: G_emp uses observed p_human directly. OLD geometry table used single")
print("      posterior draw G_H1^(seed=42), which is a DIFFERENT estimand.\n")

metrics_list = [
    ("hellinger", "Hellinger"),
    ("jensen_shannon", "JSD"),
    ("total_variation", "TV"),
    ("euclidean", "Euclidean"),
    ("aitchison", "Aitchison"),
]

# Build observed-graph for each metric
obs_graphs = {}
for met_key, _met_name in metrics_list:
    d_obs = build_distance_matrix(p_human, metric=met_key)
    w_obs_k = compute_soft_neighborhood_weights(d_obs, k=K)
    obs_graphs[met_key] = (d_obs, w_obs_k)

geo_table = {}
header = f"{'Model':<20} " + " ".join([f"{m[1]:>12}" for m in metrics_list])
print(header)
print("-" * (20 + 13 * len(metrics_list)))

for m_key in model_keys:
    row = {}
    vals = []
    for met_key, met_name in metrics_list:
        logits = models[m_key]["logits"]
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        q_m = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        d_m = build_distance_matrix(q_m, metric=met_key)
        w_m = compute_soft_neighborhood_weights(d_m, k=K)
        _, w_obs = obs_graphs[met_key]
        q_val, _ = compute_soft_qnx(w_obs, w_m, k=K)
        row[met_name] = round(float(q_val), 5)
        vals.append(f"{q_val:>12.5f}")
    geo_table[m_key] = row
    print(f"{m_key:<20} " + " ".join(vals))

# Check rank order
bart_hell = geo_table["bart-large"]["Hellinger"]
print(f"\nBART-Large Hellinger Q(G_m, G_emp): {bart_hell:.5f}")
print("   Previously reported (vs G_H1^seed42): 0.01617")
print("   Previously reported (vs G_emp):       0.01867 (from diagnostic)")

# ==========================================================================
# 4. REFERENCE LADDER: confirm HH100 vs observed ratio
# ==========================================================================
# Posterior-predictive vs observed (single seed, canonical reference)
p1_ref, _ = compute_100_vs_100_posterior_predictive_reliability(counts_full, n_votes=100, seed=0)
d1_ref = build_distance_matrix(p1_ref, metric="hellinger")
w1_ref = compute_soft_neighborhood_weights(d1_ref, k=K)
q_hh_vs_obs, _ = compute_soft_qnx(w_emp, w1_ref, k=K)
print(f"\nHH100 posterior pair (seed=0) vs G_emp: {q_hh_vs_obs:.5f}")
print("   This is the Reference Ladder 'posterior cohort vs observed' value (0.13850)")

# ==========================================================================
# Save results
# ==========================================================================
output = {
    "estimand_source": "canonical_values_audit.py",
    "panel_b_direct_means": {
        "H_bar_Q_strict": round(H_bar_strict, 5),
        "H_bar_Q_expected": round(H_bar_expected, 5),
        "H_bar_Q_fuzzy": round(H_bar_fuzzy, 5),
        "Q_strict_95ci": [round(float(np.percentile(hh_strict, 2.5)), 5), round(float(np.percentile(hh_strict, 97.5)), 5)],
        "Q_expected_95ci": [round(float(np.percentile(hh_expected, 2.5)), 5), round(float(np.percentile(hh_expected, 97.5)), 5)],
        "Q_fuzzy_95ci": [round(float(np.percentile(hh_fuzzy, 2.5)), 5), round(float(np.percentile(hh_fuzzy, 97.5)), 5)],
    },
    "direct_delta_table": direct_delta_table,
    "geometry_table_vs_G_emp": geo_table,
    "notes": {
        "old_geometry_0.01617": "Was Q(G_bart, G_H1^seed42) -- model vs single posterior draw",
        "new_geometry_from_diagnostic": "Was Q(G_bart, G_emp) -- model vs observed empirical graph",
        "this_script_geometry": "Q(G_bart, G_emp) -- same as diagnostic, should match 0.01867",
    }
}

out_path = Path("research/chaosnli/artifacts/canonical_values_audit.yaml")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    yaml.dump(output, f, default_flow_style=False, sort_keys=False)
print(f"\nSaved to {out_path}")
print("=" * 72)
