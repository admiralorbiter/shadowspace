"""
geometry_and_hbar.py
======================
Fast targeted audit:
  1. Recomputes geometry sensitivity table: Q(G_m, G_emp) for all 9 models x 5 metrics
     G_emp = observed empirical graph from p_human (NOT a single posterior draw)
  2. Confirms H_bar from the single canonical JSON release
  3. Computes direct deltas: H_bar - M_bar_m using stored q_paired_hm_mean values

Takes ~30-60 seconds (no pair loop needed).
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.models import load_model_predictions
from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights, compute_soft_qnx

K = 10
DATA_PATH = "data/chaosnli/processed/canonical_items_posterior.parquet"

print("=" * 72)
print("  GEOMETRY TABLE & H_BAR AUDIT (fast)")
print("=" * 72)

# Load data
df = pl.read_parquet(DATA_PATH)
p_human = df.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
n_items = len(df)
print(f"Loaded {n_items} items")

# Load the single committed canonical release.
with open("results/canonical_results.json", encoding="utf-8") as f:
    canon = json.load(f)

# === 1. H_bar from canonical ===
hh100_sim_mean = canon["human_reference"]["hh100_direct_pair"]["mean"]
h1_boot_mean = canon["human_reference"]["hh100_paired_focal_bootstrap"]["mean"]
print("\n--- H_bar values from stored files ---")
print(f"  hh100_simulation.mean      : {hh100_sim_mean} (direct 500-pair mean, ≈ H_bar)")
print(f"  h1_bootstrap.hh100_mean    : {h1_boot_mean} (bootstrap mean)")
print(f"  Canonical H_bar to use     : {h1_boot_mean} (paired estimand bootstrap mean)")

# === 2. Direct delta = H_bar - M_bar_m ===
H_bar = h1_boot_mean  # Use paired bootstrap mean as best available H_bar
print(f"\n--- Direct deltas using H_bar={H_bar} ---")
print(f"{'Model':<20} {'M_bar_m':>10} {'Direct Delta':>14} {'Stored Delta':>14} {'Match?':>8}")
print("-" * 72)

model_keys_display = {
    "ALBERT-xxLarge": "albert-xxlarge",
    "BART-Large": "bart-large",
    "BERT-Base": "bert-base",
    "BERT-Large": "bert-large",
    "DistilBERT": "distilbert",
    "RoBERTa-Base": "roberta-base",
    "RoBERTa-Large": "roberta-large",
    "XLNet-Base": "xlnet-base",
    "XLNet-Large": "xlnet-large",
}

delta_results = {}
for disp_key, model_slug in model_keys_display.items():
    m_data_canon = canon["model_recovery"]["models"].get(model_slug, {})
    M_bar = m_data_canon.get("q_paired_hm_mean")
    stored_delta = m_data_canon.get("delta_m_mean")

    if M_bar is None:
        print(f"{disp_key:<20} NOT FOUND")
        continue

    direct_delta = round(H_bar - M_bar, 5)
    match = abs(direct_delta - stored_delta) < 0.0001 if stored_delta else "?"

    delta_results[disp_key] = {
        "M_bar_m": M_bar,
        "direct_delta": direct_delta,
        "stored_delta": stored_delta,
    }
    print(f"{disp_key:<20} {M_bar:>10.5f} {direct_delta:>14.5f} {stored_delta:>14.5f} {'OK' if match else 'DIFF':>8}")

# === 3. Geometry sensitivity table: Q(G_m, G_emp) ===
print(f"\n--- Geometry Sensitivity Table: Q(G_m, G_emp), k={K} ---")
print("NOTE: G_emp = observed empirical graph from p_human directly")
print("      OLD table used G_H1^seed42 (single posterior draw) = DIFFERENT estimand\n")

models = load_model_predictions()
model_keys_sorted = sorted(models.keys())

metrics_list = [
    ("hellinger", "Hellinger"),
    ("jensen_shannon", "JSD"),
    ("total_variation", "TV"),
    ("euclidean", "Euclidean"),
    ("aitchison", "Aitchison"),
]

# Precompute G_emp for each metric
print("Building G_emp for all 5 metrics...")
w_emp_all = {}
for met_key, met_name in metrics_list:
    d_obs = build_distance_matrix(p_human, metric=met_key)
    w_obs = compute_soft_neighborhood_weights(d_obs, k=K)
    w_emp_all[met_key] = w_obs
    print(f"  Built {met_name} G_emp")

print()
header = f"{'Model':<20} " + " ".join([f"{m[1]:>12}" for m in metrics_list])
print(header)
print("-" * (20 + 13 * len(metrics_list)))

geo_results = {}
for m_key in model_keys_sorted:
    logits = models[m_key]["logits"]
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    q_m = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    row_vals = {}
    cells = []
    for met_key, met_name in metrics_list:
        d_m = build_distance_matrix(q_m, metric=met_key)
        w_m = compute_soft_neighborhood_weights(d_m, k=K)
        q_val, _ = compute_soft_qnx(w_emp_all[met_key], w_m, k=K)
        row_vals[met_name] = round(float(q_val), 5)
        cells.append(f"{q_val:>12.5f}")

    geo_results[m_key] = row_vals
    print(f"{m_key:<20} " + " ".join(cells))

print(f"\nBART-Large Hellinger (G_emp): {geo_results['bart-large']['Hellinger']:.5f}")
print("OLD table Hellinger (G_H1^seed42): 0.01617")
print("Diagnostic Hellinger (G_emp):      0.01867")
print(f"THIS RESULT matches diagnostic:    {'YES' if abs(geo_results['bart-large']['Hellinger'] - 0.01867) < 0.0002 else 'NO'}")

# === Save ===
output = {
    "H_bar_hh100_simulation_mean": hh100_sim_mean,
    "H_bar_h1_bootstrap_mean": h1_boot_mean,
    "direct_deltas": {k: v for k, v in delta_results.items()},
    "geometry_table_Q_G_m_G_emp": geo_results,
}

out_path = Path("research/chaosnli/artifacts/geometry_and_hbar_audit.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
    f.write("\n")
print(f"\nSaved to {out_path}")
print("=" * 72)
