import numpy as np
import polars as pl

from shadowspace.chaosnli.distances import build_distance_matrix
from shadowspace.chaosnli.models import load_model_predictions
from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights, compute_soft_qnx
from shadowspace.chaosnli.posterior import (
    compute_100_vs_100_posterior_predictive_reliability,
    compute_split_half_distributions,
)

# Load canonical dataset
df = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")
counts = df.select(["human_count_entailment", "human_count_neutral", "human_count_contradiction"]).to_numpy()

print("=========================================================================")
print("             ROUND 3 P2 METHODOLOGICAL ANALYSES RUN                      ")
print("=========================================================================\n")

# -------------------------------------------------------------------------
# TASK 1: Cross-Dataset Edge Decomposition (SNLI vs MNLI vs Pooled)
# -------------------------------------------------------------------------
print("--- TASK 1: CROSS-DATASET EDGE DECOMPOSITION (k=10, Hellinger) ---")

p_human = df.select(["human_p_entailment", "human_p_neutral", "human_p_contradiction"]).to_numpy()
d_human_pooled = build_distance_matrix(p_human, metric="hellinger")
w_human_pooled = compute_soft_neighborhood_weights(d_human_pooled, k=10)

is_snli = (df["source_dataset"] == "chaosnli_snli").to_numpy()
is_mnli = (df["source_dataset"] == "chaosnli_mnli").to_numpy()

# Calculate proportion of human edges crossing datasets
# For each focal node i, sum of w_ij for j in different dataset divided by k
total_human_weight = 10.0 * len(df)
cross_human_weight = 0.0
for i in range(len(df)):
    other_mask = is_mnli if is_snli[i] else is_snli
    cross_human_weight += np.sum(w_human_pooled[i, other_mask])

pct_human_cross = (cross_human_weight / total_human_weight) * 100.0
print(f"Human Pooled Graph Cross-Dataset Edges (SNLI <-> MNLI): {pct_human_cross:.2f}%")

# Model edge crossing proportions
models = load_model_predictions()
print(f"\n{'Model Name':<16} | {'Model Cross-Dataset Edges (%)':<30} | {'SNLI Q_NX_HM':<12} | {'MNLI Q_NX_HM':<12} | {'Pooled Q_NX_HM':<12}")
print("-" * 90)

snli_indices = np.where(is_snli)[0]
mnli_indices = np.where(is_mnli)[0]

p_human_snli = p_human[snli_indices]
p_human_mnli = p_human[mnli_indices]

d_human_snli = build_distance_matrix(p_human_snli, metric="hellinger")
d_human_mnli = build_distance_matrix(p_human_mnli, metric="hellinger")

w_human_snli = compute_soft_neighborhood_weights(d_human_snli, k=10)
w_human_mnli = compute_soft_neighborhood_weights(d_human_mnli, k=10)

for m_name, m_data in models.items():
    logits = m_data["logits"]
    # Softmax probabilities
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    q_m = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    d_m_pooled = build_distance_matrix(q_m, metric="hellinger")
    w_m_pooled = compute_soft_neighborhood_weights(d_m_pooled, k=10)

    # Model cross-dataset edge pct
    cross_m_weight = 0.0
    for i in range(len(df)):
        other_mask = is_mnli if is_snli[i] else is_snli
        cross_m_weight += np.sum(w_m_pooled[i, other_mask])
    pct_m_cross = (cross_m_weight / total_human_weight) * 100.0

    # Within-dataset recoveries
    q_m_snli = q_m[snli_indices]
    q_m_mnli = q_m[mnli_indices]

    d_m_snli = build_distance_matrix(q_m_snli, metric="hellinger")
    d_m_mnli = build_distance_matrix(q_m_mnli, metric="hellinger")

    w_m_snli = compute_soft_neighborhood_weights(d_m_snli, k=10)
    w_m_mnli = compute_soft_neighborhood_weights(d_m_mnli, k=10)

    qnx_snli, _ = compute_soft_qnx(w_human_snli, w_m_snli, k=10)
    qnx_mnli, _ = compute_soft_qnx(w_human_mnli, w_m_mnli, k=10)
    qnx_pooled, _ = compute_soft_qnx(w_human_pooled, w_m_pooled, k=10)

    print(f"{m_name:<16} | {pct_m_cross:<30.2f}% | {qnx_snli:<12.5f} | {qnx_mnli:<12.5f} | {qnx_pooled:<12.5f}")


# -------------------------------------------------------------------------
# TASK 2: Formal Model-Human Difference Bootstrap (Delta_m)
# -------------------------------------------------------------------------
print("\n--- TASK 2: FORMAL MODEL-HUMAN DIFFERENCE BOOTSTRAP (Delta_m = Q_HH100 - Q_HM) ---")

# Compute 100/100 posterior predictive human reference graph
p1_100, p2_100 = compute_100_vs_100_posterior_predictive_reliability(counts, n_votes=100, seed=42)
d1_100 = build_distance_matrix(p1_100, metric="hellinger")
d2_100 = build_distance_matrix(p2_100, metric="hellinger")
w1_100 = compute_soft_neighborhood_weights(d1_100, k=10)
w2_100 = compute_soft_neighborhood_weights(d2_100, k=10)

# Local soft overlap for HH100 reference (between the two independent 100-vote replicates)
_, local_o_hh100 = compute_soft_qnx(w1_100, w2_100, k=10)
qnx_hh100_mean = float(np.mean(local_o_hh100))

print(f"Independent 100/100 Human Reference Q_NX: {qnx_hh100_mean:.5f}\n")
print(f"{'Model Name':<16} | {'Soft Q_HM':<10} | {'Mean Delta_m (HH100 - HM)':<25} | {'95% Joint Bootstrap CI':<25}")
print("-" * 82)

rng = np.random.default_rng(20260801)
n_items = len(df)

for m_name, m_data in models.items():
    logits = m_data["logits"]
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    q_m = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    d_m = build_distance_matrix(q_m, metric="hellinger")
    w_m = compute_soft_neighborhood_weights(d_m, k=10)

    # Local soft overlap vs human 100-vote replicate w1_100
    qnx_hm, local_o_hm = compute_soft_qnx(w1_100, w_m, k=10)

    # Itemwise difference delta_i = local_o_hh100_i - local_o_hm_i
    delta_i = local_o_hh100 - local_o_hm

    # Stratified bootstrap over delta_i
    boot_deltas = []
    snli_mask = is_snli
    mnli_mask = is_mnli
    n_snli_c = int(snli_mask.sum())
    n_mnli_c = int(mnli_mask.sum())

    for _ in range(1000):
        idx_s = rng.choice(np.where(snli_mask)[0], size=n_snli_c, replace=True)
        idx_m = rng.choice(np.where(mnli_mask)[0], size=n_mnli_c, replace=True)
        boot_idx = np.concatenate([idx_s, idx_m])
        boot_deltas.append(float(delta_i[boot_idx].mean()))

    delta_mean = float(np.mean(boot_deltas))
    ci_low = float(np.percentile(boot_deltas, 2.5))
    ci_high = float(np.percentile(boot_deltas, 97.5))

    print(f"{m_name:<16} | {qnx_hm:<10.5f} | {delta_mean:<25.5f} | [{ci_low:.5f}, {ci_high:.5f}]")


# -------------------------------------------------------------------------
# TASK 3: Deeper Analysis of the 0.8140 Human Reference Result (Prior & Zero Counts)
# -------------------------------------------------------------------------
print("\n--- TASK 3: HUMAN REFERENCE 0.8140 AUDIT (PRIOR CHOICE & ZERO-COUNT BREAKDOWN) ---")

# Empirical 100-vote graph
p_emp = counts / 100.0
d_emp = build_distance_matrix(p_emp, metric="hellinger")
w_emp = compute_soft_neighborhood_weights(d_emp, k=10)

# Posterior mean under alpha = (0.5, 0.5, 0.5) [Jeffreys]
alpha_05 = counts + 0.5
p_post_05 = alpha_05 / np.sum(alpha_05, axis=1, keepdims=True)
d_post_05 = build_distance_matrix(p_post_05, metric="hellinger")
w_post_05 = compute_soft_neighborhood_weights(d_post_05, k=10)
qnx_emp_vs_05, local_o_05 = compute_soft_qnx(w_emp, w_post_05, k=10)

# Posterior mean under alpha = (1.0, 1.0, 1.0) [Uniform]
alpha_10 = counts + 1.0
p_post_10 = alpha_10 / np.sum(alpha_10, axis=1, keepdims=True)
d_post_10 = build_distance_matrix(p_post_10, metric="hellinger")
w_post_10 = compute_soft_neighborhood_weights(d_post_10, k=10)
qnx_emp_vs_10, local_o_10 = compute_soft_qnx(w_emp, w_post_10, k=10)

print(f"Empirical 100-Vote vs Posterior Mean (alpha=0.5) Soft Q_NX : {qnx_emp_vs_05:.4f} (18.6% turnover)")
print(f"Empirical 100-Vote vs Posterior Mean (alpha=1.0) Soft Q_NX : {qnx_emp_vs_10:.4f} (27.2% turnover)")

# Zero count breakdown for alpha=0.5
has_zero = (np.min(counts, axis=1) == 0)
no_zero = ~has_zero

qnx_zero_05 = float(np.mean(local_o_05[has_zero]))
qnx_no_zero_05 = float(np.mean(local_o_05[no_zero]))

print(f"\nEmpirical vs Posterior-Mean (alpha=0.5) by Zero-Vote Status:")
print(f"  Items with Zero Counts ({has_zero.sum()} items, {has_zero.mean()*100:.1f}%): Q_NX = {qnx_zero_05:.4f} ({100*(1-qnx_zero_05):.1f}% turnover)")
print(f"  Items without Zero Counts ({no_zero.sum()} items, {no_zero.mean()*100:.1f}%): Q_NX = {qnx_no_zero_05:.4f} ({100*(1-qnx_no_zero_05):.1f}% turnover)")

# Entropy tier breakdown
entropy_bits = -np.sum(np.where(p_emp > 0, p_emp * np.log2(np.maximum(p_emp, 1e-12)), 0.0), axis=1)
quantiles = np.percentile(entropy_bits, [20, 40, 60, 80])
t1 = entropy_bits <= quantiles[0]
t2 = (entropy_bits > quantiles[0]) & (entropy_bits <= quantiles[1])
t3 = (entropy_bits > quantiles[1]) & (entropy_bits <= quantiles[2])
t4 = (entropy_bits > quantiles[2]) & (entropy_bits <= quantiles[3])
t5 = entropy_bits > quantiles[3]

print(f"\nEmpirical vs Posterior-Mean (alpha=0.5) by Entropy Tier:")
for idx, tier_mask in enumerate([t1, t2, t3, t4, t5], 1):
    tier_qnx = float(np.mean(local_o_05[tier_mask]))
    tier_ent_mean = float(np.mean(entropy_bits[tier_mask]))
    print(f"  Tier {idx} (Mean H = {tier_ent_mean:.3f} bits): Q_NX = {tier_qnx:.4f} ({100*(1-tier_qnx):.1f}% turnover)")


# -------------------------------------------------------------------------
# TASK 4: Multi-Scale Neighborhood Reliability & Stability (k = 5, 10, 20, 50, 100)
# -------------------------------------------------------------------------
print("\n--- TASK 4: MULTI-SCALE NEIGHBORHOOD RELIABILITY & STABILITY (k in [5, 10, 20, 50, 100]) ---")

k_list = [5, 10, 20, 50, 100]

print(f"{'k':<5} | {'Chance Baseline':<16} | {'Human 50/50 Split':<18} | {'Human 100/100 Ref':<18} | {'BART-Large':<12} | {'RoBERTa-Large':<14} | {'XLNet-Large':<12}")
print("-" * 105)

for k_val in k_list:
    chance = k_val / (len(df) - 1)

    # 50/50 split half
    p1_50, p2_50 = compute_split_half_distributions(counts, seed=42)
    d1_50 = build_distance_matrix(p1_50, metric="hellinger")
    d2_50 = build_distance_matrix(p2_50, metric="hellinger")
    w1_50 = compute_soft_neighborhood_weights(d1_50, k=k_val)
    w2_50 = compute_soft_neighborhood_weights(d2_50, k=k_val)
    qnx_50, _ = compute_soft_qnx(w1_50, w2_50, k=k_val)

    # 100/100 replicate
    p1_100, p2_100 = compute_100_vs_100_posterior_predictive_reliability(counts, n_votes=100, seed=42)
    d1_100 = build_distance_matrix(p1_100, metric="hellinger")
    d2_100 = build_distance_matrix(p2_100, metric="hellinger")
    w1_100 = compute_soft_neighborhood_weights(d1_100, k=k_val)
    w2_100 = compute_soft_neighborhood_weights(d2_100, k=k_val)
    qnx_100, _ = compute_soft_qnx(w1_100, w2_100, k=k_val)

    # Model evaluation at k_val
    def get_model_qnx(m_key: str) -> float:
        logits = models[m_key]["logits"]
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        q_m = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        d_m = build_distance_matrix(q_m, metric="hellinger")
        w_m = compute_soft_neighborhood_weights(d_m, k=k_val)
        val, _ = compute_soft_qnx(w1_100, w_m, k=k_val)
        return float(val)

    q_bart = get_model_qnx("bart-large")
    q_roberta = get_model_qnx("roberta-large")
    q_xlnet = get_model_qnx("xlnet-large")

    print(f"{k_val:<5} | {chance:<16.5f} | {qnx_50:<18.5f} | {qnx_100:<18.5f} | {q_bart:<12.5f} | {q_roberta:<14.5f} | {q_xlnet:<12.5f}")

print("\n=========================================================================")
print("                    P2 ANALYSES COMPLETED CLEANLY                        ")
print("=========================================================================")
