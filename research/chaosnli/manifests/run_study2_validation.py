import numpy as np
import polars as pl

from shadowspace.chaosnli.edge_ledger import build_persistent_edge_ledger
from shadowspace.chaosnli.joint_spaces import (
    compute_joint_distance_matrix,
    compute_lexicographic_tie_breaking,
    compute_random_tie_breaking_baseline,
)
from shadowspace.chaosnli.linguistic_validation import extract_linguistic_disagreement_taxonomy, evaluate_taxonomy_retrieval
from shadowspace.chaosnli.models import load_model_predictions
from shadowspace.chaosnli.neighbors import extract_knn_graph
from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights, compute_soft_qnx
from shadowspace.chaosnli.profile_graph import (
    analyze_model_dispersion_drivers,
    build_level1_profile_graph,
    compute_profile_level_model_dispersion,
)

df = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")
d_opinion = np.load("data/chaosnli/processed/distance_matrix_human_hellinger.npy")
d_text = np.load("data/chaosnli/processed/distance_matrix_text_cosine.npy")
models = load_model_predictions()

print("=========================================================================")
print("    STUDY 2 VALIDATION SPRINT: OPTION B (DEEP AUDIT) & OPTION A (EXTERNAL) ")
print("=========================================================================\n")

# --- PHASE 1 (OPTION B): REFINED EDGE LEDGER & DISPERSION DRIVERS ---
print("--- 1. QUANTILE-BASED EDGE LEDGER TAXONOMY (OPTION B) ---")
ledger_df = build_persistent_edge_ledger(df, models, d_text=d_text, k=10, metric="hellinger", use_quantiles=True)
cat_counts = ledger_df["diagnostic_category"].value_counts().sort("count", descending=True)

print(f"Total Candidate Edges Evaluated : {len(ledger_df)}\n")
print(f"{'Diagnostic Category':<45} | {'Edge Count':<12} | {'Percentage':<10}")
print("-" * 75)
for row in cat_counts.iter_rows(named=True):
    pct = (row["count"] / len(ledger_df)) * 100.0
    print(f"{row['diagnostic_category']:<45} | {row['count']:<12} | {pct:<9.2f}%")

print("\n--- 2. PROFILE-LEVEL MODEL DISPERSION DRIVERS (OPTION B) ---")
level1_res = build_level1_profile_graph(df, metric="hellinger", k=10)
dispersion_df = compute_profile_level_model_dispersion(df, level1_res["profile_df"], models, metric="hellinger")
drivers = analyze_model_dispersion_drivers(df, dispersion_df)

print(f"Correlation (Model Dispersion vs Human Shannon Entropy)    : r = {drivers['corr_dispersion_entropy']:.4f}")
print(f"Correlation (Model Dispersion vs Profile Frequency)        : r = {drivers['corr_dispersion_frequency']:.4f}")
print(f"Correlation (Model Dispersion vs Consensus Dominance max_p): r = {drivers['corr_dispersion_max_class_p']:.4f}")


# --- PHASE 2 (OPTION A): EXTERNAL LINGUISTIC TAXONOMY VALIDATION ---
print("\n--- 3. EXTERNAL LINGUISTIC DISAGREEMENT VALIDATION (OPTION A) ---")
taxonomy_df = extract_linguistic_disagreement_taxonomy(df)
tax_dist = taxonomy_df["primary_linguistic_category"].value_counts().sort("count", descending=True)
print("Extracted Structural Linguistic Disagreement Taxonomy Breakdown:")
for row in tax_dist.iter_rows(named=True):
    pct = (row["count"] / len(df)) * 100.0
    print(f"  {row['primary_linguistic_category']:<35} : {row['count']:<5} ({pct:.2f}%)")

print("\n--- 4. BENCHMARKING 5 TIE-RESOLUTION STRATEGIES VS EXTERNAL TAXONOMY ---")

# Strategy 1: Random Tie-Breaking (Baseline)
rand_knn_idx, _ = compute_random_tie_breaking_baseline(d_opinion, k=10, seed=42)
rand_metrics = evaluate_taxonomy_retrieval(rand_knn_idx, taxonomy_df, df, k=10)

# Strategy 2: Lexicographic Tie-Breaking (d_opinion primary, d_text secondary)
lex_knn_idx, _ = compute_lexicographic_tie_breaking(d_opinion, d_text, k=10)
lex_metrics = evaluate_taxonomy_retrieval(lex_knn_idx, taxonomy_df, df, k=10)

# Strategy 3: Global Lambda-Blend (lambda=0.05)
d_joint_005 = compute_joint_distance_matrix(d_opinion, d_text, lambda_weight=0.05)
blend_knn_idx, _ = extract_knn_graph(d_joint_005, df["object_id"].to_list(), k=10, space_id="joint_005")
blend_metrics = evaluate_taxonomy_retrieval(blend_knn_idx, taxonomy_df, df, k=10)

# Strategy 4: Pure Text Similarity (lambda=1.00)
text_knn_idx, _ = extract_knn_graph(d_text, df["object_id"].to_list(), k=10, space_id="pure_text")
text_metrics = evaluate_taxonomy_retrieval(text_knn_idx, taxonomy_df, df, k=10)

print(f"\n{'Tie-Resolution Strategy':<35} | {'Jaccard@10':<12} | {'MAP@10':<12} | {'NDCG@10':<12}")
print("-" * 75)
print(f"{'1. Random Tie-Breaking (Baseline)':<35} | {rand_metrics['mean_taxonomy_jaccard_at_k']:<12.4f} | {rand_metrics['mean_average_precision_map_at_k']:<12.4f} | {rand_metrics['mean_ndcg_at_k']:<12.4f}")
print(f"{'2. Lexicographic Tie-Breaking':<35} | {lex_metrics['mean_taxonomy_jaccard_at_k']:<12.4f} | {lex_metrics['mean_average_precision_map_at_k']:<12.4f} | {lex_metrics['mean_ndcg_at_k']:<12.4f}")
print(f"{'3. Global Lambda-Blend (lambda=0.05)':<35} | {blend_metrics['mean_taxonomy_jaccard_at_k']:<12.4f} | {blend_metrics['mean_average_precision_map_at_k']:<12.4f} | {blend_metrics['mean_ndcg_at_k']:<12.4f}")
print(f"{'4. Pure Text Embedding Space':<35} | {text_metrics['mean_taxonomy_jaccard_at_k']:<12.4f} | {text_metrics['mean_average_precision_map_at_k']:<12.4f} | {text_metrics['mean_ndcg_at_k']:<12.4f}")

# Relative improvement of Lexicographic over Random
jaccard_gain = ((lex_metrics['mean_taxonomy_jaccard_at_k'] - rand_metrics['mean_taxonomy_jaccard_at_k']) / rand_metrics['mean_taxonomy_jaccard_at_k']) * 100.0
map_gain = ((lex_metrics['mean_average_precision_map_at_k'] - rand_metrics['mean_average_precision_map_at_k']) / rand_metrics['mean_average_precision_map_at_k']) * 100.0

print(f"\nLexicographic vs Random Relative Gain: Jaccard@10 = +{jaccard_gain:.2f}%, MAP@10 = +{map_gain:.2f}%")

print("\n=========================================================================")
print("         STUDY 2 VALIDATION PIPELINE EXECUTION COMPLETE                  ")
print("=========================================================================")
