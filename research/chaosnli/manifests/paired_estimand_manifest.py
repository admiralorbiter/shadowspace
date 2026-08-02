"""
paired_estimand_manifest.py
============================
Computes the fully paired model-human estimand:

    H_b  = Q_fuzzy(G_H1^(b), G_H2^(b))                      [human vs. human]
    M_m,b = 0.5 * [Q_fuzzy(G_m, G_H1^(b)) + Q_fuzzy(G_m, G_H2^(b))]  [model vs. same cohorts]
    Delta_m,b = H_b - M_m,b

This replaces the asymmetric design (model vs. fixed observed graph) with a
fully symmetric comparison where both human and model scores are evaluated
against the same two posterior-predictive cohorts.

The fixed full-data scores Q(G_m, G_100^obs) are retained as a separate
descriptive benchmark (Panel A of the reference ladder).

Output: research/chaosnli/artifacts/paired_estimand_results.json
"""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.models import load_model_predictions
from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights, compute_soft_qnx
from shadowspace.chaosnli.posterior import compute_100_vs_100_posterior_predictive_reliability

# ── Config ──────────────────────────────────────────────────────────────────
K = 10
N_PAIRS = 500
N_BOOT = 1000
SEED_START = 0
DATA_PATH = "data/chaosnli/processed/canonical_items_posterior.parquet"
OUTPUT_PATH = Path("research/chaosnli/artifacts/paired_estimand_results.json")

print("=" * 72)
print("   PAIRED ESTIMAND MANIFEST (Q_m vs POSTERIOR COHORTS)")
print("=" * 72)

# ── Load Data ────────────────────────────────────────────────────────────────
df = pl.read_parquet(DATA_PATH)
p_human = df.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
counts_full = df.select(["human_count_entailment", "human_count_neutral", "human_count_contradiction"]).to_numpy()
is_snli = (df["source_dataset"] == "chaosnli_snli").to_numpy()
is_mnli = (df["source_dataset"] == "chaosnli_mnli").to_numpy()
snli_indices = np.where(is_snli)[0]
mnli_indices = np.where(is_mnli)[0]
n_items = len(df)

print(f"Loaded {n_items} items ({len(snli_indices)} SNLI, {len(mnli_indices)} MNLI)")

# ── Observed human graph ─────────────────────────────────────────────────────
d_emp = build_distance_matrix(p_human, metric="hellinger")
w_emp = compute_soft_neighborhood_weights(d_emp, k=K)
print(f"Built empirical human graph w_emp (Hellinger, k={K})")

# ── Load models ───────────────────────────────────────────────────────────────
models = load_model_predictions()
model_keys = sorted(models.keys())
print(f"Loaded {len(model_keys)} models: {model_keys}")

# ── Pre-compute per-item overlaps for each model vs observed ─────────────────
# o_hm_fixed[m_key][i] = Q_fuzzy_item(G_m, G_100^obs)  — fixed reference
o_hm_fixed = {}
w_models = {}
for m_key in model_keys:
    logits = models[m_key]["logits"]
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    q_m = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    d_m = build_distance_matrix(q_m, metric="hellinger")
    w_m = compute_soft_neighborhood_weights(d_m, k=K)
    w_models[m_key] = w_m
    val_full, o_hm = compute_soft_qnx(w_emp, w_m, k=K)
    o_hm_fixed[m_key] = o_hm
    # Map display name
    display = m_key.replace('-', ' ').title().replace('Xxlarge', 'xxLarge').replace('Bert', 'BERT').replace('Bart', 'BART').replace('Roberta', 'RoBERTa').replace('Xlnet', 'XLNet').replace('Albert', 'ALBERT').replace('Distilbert', 'DistilBERT')
    print(f"  {display}: fixed-reference Q={val_full:.5f}")

# ── Pre-compute 500 posterior pairs ──────────────────────────────────────────
print(f"\nPre-computing {N_PAIRS} posterior-predictive pairs...")
t0 = time.time()

# For each pair s, store per-item overlaps:
# hh_overlaps[s][i] = Q_fuzzy_item(G_H1^(s), G_H2^(s))
# For each model and pair: o_mh1[m][s][i] = Q_fuzzy_item(G_m, G_H1^(s))
#                          o_mh2[m][s][i] = Q_fuzzy_item(G_m, G_H2^(s))

hh_overlaps = np.zeros((N_PAIRS, n_items))    # [500, 3113]
o_mh1 = {m: np.zeros((N_PAIRS, n_items)) for m in model_keys}  # [500, 3113] per model
o_mh2 = {m: np.zeros((N_PAIRS, n_items)) for m in model_keys}

for s in range(N_PAIRS):
    p1, p2 = compute_100_vs_100_posterior_predictive_reliability(counts_full, n_votes=100, seed=s)
    d1 = build_distance_matrix(p1, metric="hellinger")
    d2 = build_distance_matrix(p2, metric="hellinger")
    w_h1 = compute_soft_neighborhood_weights(d1, k=K)
    w_h2 = compute_soft_neighborhood_weights(d2, k=K)

    # HH overlaps
    _, hh_item = compute_soft_qnx(w_h1, w_h2, k=K)
    hh_overlaps[s] = hh_item

    # Model vs. each cohort
    for m_key in model_keys:
        w_m = w_models[m_key]
        _, o1 = compute_soft_qnx(w_h1, w_m, k=K)
        _, o2 = compute_soft_qnx(w_h2, w_m, k=K)
        o_mh1[m_key][s] = o1
        o_mh2[m_key][s] = o2

    if (s + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  Completed pair {s+1}/{N_PAIRS} in {elapsed:.1f}s")

print(f"All {N_PAIRS} pairs computed in {time.time()-t0:.1f}s")

# ── Bootstrap ─────────────────────────────────────────────────────────────────
print(f"\nRunning {N_BOOT} stratified joint bootstrap replicates...")

rng_boot = np.random.default_rng(42)
q_hhs_boot = []
q_hms_paired = {m: [] for m in model_keys}
delta_ms = {m: [] for m in model_keys}

for b in range(N_BOOT):
    s = b % N_PAIRS
    # Stratified focal-item resample
    b_snli = rng_boot.choice(snli_indices, size=len(snli_indices), replace=True)
    b_mnli = rng_boot.choice(mnli_indices, size=len(mnli_indices), replace=True)
    b_idx = np.concatenate([b_snli, b_mnli])

    h_b = float(np.mean(hh_overlaps[s][b_idx]))
    q_hhs_boot.append(h_b)

    for m_key in model_keys:
        # Paired model score: average of Q(G_m, G_H1^(s)) and Q(G_m, G_H2^(s))
        m_b = 0.5 * (float(np.mean(o_mh1[m_key][s][b_idx])) + float(np.mean(o_mh2[m_key][s][b_idx])))
        q_hms_paired[m_key].append(m_b)
        delta_ms[m_key].append(h_b - m_b)

print("Bootstrap complete.")

# ── Summary statistics ────────────────────────────────────────────────────────
hh_boot_arr = np.array(q_hhs_boot)
hh_boot_mean = float(np.mean(hh_boot_arr))
hh_boot_ci = [float(np.percentile(hh_boot_arr, 2.5)), float(np.percentile(hh_boot_arr, 97.5))]

print(f"\nHH100 Bootstrap Mean (paired):  {hh_boot_mean:.5f}")
print(f"HH100 Bootstrap 95% CI:         [{hh_boot_ci[0]:.5f}, {hh_boot_ci[1]:.5f}]")
print()
print(f"{'Model':<20} {'Paired Q_m':>12} {'Delta_m':>10} {'95% CI Low':>12} {'95% CI Hi':>12} {'Fixed Q_m':>12}")
print("-" * 82)

model_results = {}
for m_key in model_keys:
    arr = np.array(q_hms_paired[m_key])
    d_arr = np.array(delta_ms[m_key])
    q_mean = float(np.mean(arr))
    d_mean = float(np.mean(d_arr))
    d_ci = [float(np.percentile(d_arr, 2.5)), float(np.percentile(d_arr, 97.5))]
    n_gt_zero = int(np.sum(d_arr > 0))
    # Fixed reference (all items, no bootstrap)
    q_fixed = float(np.mean(o_hm_fixed[m_key]))

    print(f"{m_key:<20} {q_mean:>12.5f} {d_mean:>10.5f} {d_ci[0]:>12.5f} {d_ci[1]:>12.5f} {q_fixed:>12.5f}")

    model_results[m_key] = {
        "q_paired_hm_mean": round(q_mean, 5),
        "delta_m_mean": round(d_mean, 5),
        "delta_m_95ci": [round(d_ci[0], 5), round(d_ci[1], 5)],
        "replicates_gt_zero": f"{n_gt_zero}/{N_BOOT}",
        "q_fixed_reference": round(q_fixed, 5),
    }

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    "estimand": "paired",
    "description": (
        "Paired estimand: H_b = Q(G_H1^(b), G_H2^(b)), "
        "M_m,b = 0.5 * [Q(G_m, G_H1^(b)) + Q(G_m, G_H2^(b))]. "
        "Both human and model scores evaluated against the same two posterior cohorts."
    ),
    "n_pairs": N_PAIRS,
    "n_bootstrap": N_BOOT,
    "k": K,
    "hh100_bootstrap_mean": round(hh_boot_mean, 5),
    "hh100_bootstrap_95ci": hh_boot_ci,
    "models": model_results,
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
    f.write("\n")

print(f"\nSaved paired estimand results to {OUTPUT_PATH}")
print("=" * 72)
print("PAIRED ESTIMAND MANIFEST COMPLETE")
print("=" * 72)
