import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.models import load_model_predictions
from shadowspace.chaosnli.model_topology import evaluate_hypothesis2_temperature_scaling, evaluate_model_topology_recovery
from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights, compute_soft_qnx
from shadowspace.chaosnli.posterior import (
    compute_100_vs_100_posterior_predictive_reliability,
    compute_split_half_distributions,
)

df = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")
counts = df.select(["human_count_entailment", "human_count_neutral", "human_count_contradiction"]).to_numpy()

print("=========================================================================")
print("          ROUND 2 PEER-REVIEW SYNTHESIS & AUDIT REPORT                   ")
print("=========================================================================\n")

# 1. Complete Human Reference Spectrum
print("--- 1. HUMAN RELIABILITY REFERENCE SPECTRUM (k=10, Hellinger) ---")

# A. Complementary 50/50 split half
p1_50, p2_50 = compute_split_half_distributions(counts, seed=42)
d1_50 = build_distance_matrix(p1_50, metric="hellinger")
d2_50 = build_distance_matrix(p2_50, metric="hellinger")
w1_50 = compute_soft_neighborhood_weights(d1_50, k=10)
w2_50 = compute_soft_neighborhood_weights(d2_50, k=10)
qnx_50_50, _ = compute_soft_qnx(w1_50, w2_50, k=10)

# B. Independent 100/100 posterior predictive replicate
p1_100, p2_100 = compute_100_vs_100_posterior_predictive_reliability(counts, n_votes=100, seed=42)
d1_100 = build_distance_matrix(p1_100, metric="hellinger")
d2_100 = build_distance_matrix(p2_100, metric="hellinger")
w1_100 = compute_soft_neighborhood_weights(d1_100, k=10)
w2_100 = compute_soft_neighborhood_weights(d2_100, k=10)
qnx_100_100, _ = compute_soft_qnx(w1_100, w2_100, k=10)

chance_baseline = 10.0 / (len(df) - 1)

print(f"Complementary 50/50 Observed Split-Half Soft Q_NX : {qnx_50_50:.4f}")
print(f"Independent 100/100 Posterior-Predictive Soft Q_NX: {qnx_100_100:.4f}")
print(f"Random Overlap Chance Baseline k/(N-1)             : {chance_baseline:.5f}")

# 2. Model Evaluation with Stratified 95% Bootstrap CIs
print("\n--- 2. MODEL BENCHMARK WITH STRATIFIED 95% BOOTSTRAP CIs ---")
models = load_model_predictions()
eval_res = evaluate_model_topology_recovery(models, canonical_items_path=df, k=10, qnx_hh_soft=qnx_100_100)

print(f"{'Model Name':<16} | {'Soft Q_NX_HM':<12} | {'95% Bootstrap CI':<18} | {'Excess Ratio (vs 100-vote)':<26}")
print("-" * 80)
for m_name, res in eval_res.items():
    ex_ratio_100 = (res["qnx_soft_hm"] - chance_baseline) / (qnx_100_100 - chance_baseline)
    print(f"{m_name:<16} | {res['qnx_soft_hm']:<12.5f} | [{res['ci_95_lower']:.5f}, {res['ci_95_upper']:.5f}] | {ex_ratio_100*100:<25.1f}%")

# 3. Direct Model Graph Turnover across Temperatures (H2)
print("\n--- 3. DIRECT MODEL GRAPH TURNOVER ACROSS TEMPERATURES (RoBERTa-Large) ---")
h2_res = evaluate_hypothesis2_temperature_scaling(models, df, temperatures=[0.5, 0.8, 1.0, 1.2, 1.5, 2.0], k=10)
print(f"{'T':<5} | {'Pointwise JSD (bits)':<20} | {'Soft Q_NX_HM (vs Human)':<24} | {'Self Q_NX (vs T=1.0)':<20} | {'Edge Turnover':<14}")
print("-" * 90)
for row in h2_res["roberta-large"]:
    print(f"{row['temperature']:<5.1f} | {row['mean_jsd_bits']:<20.4f} | {row['qnx_soft_hm']:<24.5f} | {row['qnx_model_self_vs_t1']:<20.5f} | {row['edge_turnover_vs_t1']*100:<13.1f}%")

# 4. SNLI vs MNLI Stratified Reporting
print("\n--- 4. SNLI vs MNLI STRATIFIED REPORTING ---")
snli_df = df.filter(pl.col("source_dataset") == "chaosnli_snli")
mnli_df = df.filter(pl.col("source_dataset") == "chaosnli_mnli")

snli_eval = evaluate_model_topology_recovery(models, canonical_items_path=snli_df, k=10, qnx_hh_soft=qnx_100_100)
mnli_eval = evaluate_model_topology_recovery(models, canonical_items_path=mnli_df, k=10, qnx_hh_soft=qnx_100_100)

print(f"{'Model Name':<16} | {'SNLI Q_NX_HM':<14} | {'MNLI Q_NX_HM':<14} | {'Pooled Q_NX_HM':<14}")
print("-" * 65)
for m_name in eval_res.keys():
    q_snli = snli_eval[m_name]["qnx_soft_hm"]
    q_mnli = mnli_eval[m_name]["qnx_soft_hm"]
    q_pool = eval_res[m_name]["qnx_soft_hm"]
    print(f"{m_name:<16} | {q_snli:<14.5f} | {q_mnli:<14.5f} | {q_pool:<14.5f}")

print("\n=========================================================================")
print("             ROUND 2 PEER REVIEW SYNTHESIS COMPLETE                      ")
print("=========================================================================")
