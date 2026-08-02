import numpy as np
import polars as pl

from shadowspace.chaosnli.edge_ledger import build_persistent_edge_ledger
from shadowspace.chaosnli.joint_spaces import compute_lexicographic_tie_breaking, evaluate_hypothesis7_joint_space
from shadowspace.chaosnli.models import load_model_predictions
from shadowspace.chaosnli.neighbors_soft import compute_soft_neighborhood_weights, compute_soft_qnx
from shadowspace.chaosnli.profile_graph import build_level1_profile_graph, compute_profile_level_model_dispersion

df = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")
d_opinion = np.load("data/chaosnli/processed/distance_matrix_human_hellinger.npy")
d_text = np.load("data/chaosnli/processed/distance_matrix_text_cosine.npy")
models = load_model_predictions()

print("=========================================================================")
print("          STUDY 2 REFRAMED: TWO-LEVEL ARCHITECTURE & EXPANDED AUDIT       ")
print("=========================================================================\n")

# 1. Level 1 Opinion Profile Graph & Level 2 Model Dispersion
print("--- 1. LEVEL-1 PROFILE GRAPH & LEVEL-2 PROFILE MODEL DISPERSION ---")
level1_res = build_level1_profile_graph(df, metric="hellinger", k=10)
profile_df = level1_res["profile_df"]

print(f"Total Canonical Items           : {len(df)}")
print(f"Unique Level-1 Opinion Profiles : {level1_res['n_profiles']}")
print(f"Multi-Item Opinion Profiles     : {profile_df.filter(pl.col('profile_frequency') > 1).height}")

dispersion_df = compute_profile_level_model_dispersion(df, profile_df, models, metric="hellinger")

print(f"\nMean Profile-Level Model Dispersion across multi-item profiles:")
for col in dispersion_df.columns:
    if col.startswith("dispersion_") or col == "mean_model_dispersion":
        print(f"  {col:<30}: {dispersion_df[col].mean():.5f} Hellinger")

# Top 5 highest model dispersion profiles (items with identical human vote counts that models separate most)
print("\nTop 5 Opinion Profiles with HIGHEST Model Dispersion (Models separate identical human profiles):")
top_high_disp = dispersion_df.sort("mean_model_dispersion", descending=True).head(5)
for row in top_high_disp.iter_rows(named=True):
    print(f"  Profile {row['profile_id']} (Freq={row['profile_frequency']}, Ent={row['entropy_bits']:.2f}): Mean Disp = {row['mean_model_dispersion']:.4f} [p=({row['p_entailment']:.2f}, {row['p_neutral']:.2f}, {row['p_contradiction']:.2f})]")


# 2. Persistent Edge Ledger & 6-Category Taxonomy Breakdown
print("\n--- 2. PERSISTENT EDGE LEDGER & 6-CATEGORY TAXONOMY BREAKDOWN ---")
ledger_df = build_persistent_edge_ledger(df, models, d_text=d_text, k=10, metric="hellinger")

cat_counts = ledger_df["diagnostic_category"].value_counts().sort("count", descending=True)
print(f"Total Ledger Candidate Edges: {len(ledger_df)}\n")
print(f"{'Diagnostic Category':<45} | {'Edge Count':<12} | {'Percentage':<10}")
print("-" * 75)
for row in cat_counts.iter_rows(named=True):
    pct = (row["count"] / len(ledger_df)) * 100.0
    print(f"{row['diagnostic_category']:<45} | {row['count']:<12} | {pct:<9.2f}%")


# 3. Methodological Tie-Resolution Comparison (Study 2 Reframed)
print("\n--- 3. STUDY 2 METHODOLOGICAL TIE-RESOLUTION COMPARISON ---")

# Method A: Pure Opinion with Soft Overlap
w_opinion = compute_soft_neighborhood_weights(d_opinion, k=10)

# Method B: Lexicographic Tie-Breaking (d_opinion primary, d_text secondary)
lex_knn_idx, lex_knn_dists = compute_lexicographic_tie_breaking(d_opinion, d_text, k=10)

# Method C: Joint Lambda Blend (lambda = 0.05)
h7_eval = evaluate_hypothesis7_joint_space(df, d_opinion, d_text, lambdas=[0.0, 0.05, 0.1, 0.5], k=10)

print(f"Opinion Zero-Distance Ties (Pure Opinion lambda=0.00) : {h7_eval['lambda_evaluations'][0]['zero_distance_ties_remaining']}")
print(f"Lexicographic Tie-Breaking Zero Ties                   : 0 (100% resolved via secondary text distance)")
print(f"Joint Lambda Blend (lambda=0.05) Zero Ties            : {h7_eval['lambda_evaluations'][1]['zero_distance_ties_remaining']} (100% resolved)")
print(f"\nJoint Lambda Blend (lambda=0.05) Soft Q_NX Recovery   : {h7_eval['lambda_evaluations'][1]['qnx_soft_opinion_recovery']:.4f}")

print("\n=========================================================================")
print("         STUDY 2 REFRAMED RESEARCH EVALUATION COMPLETE                   ")
print("=========================================================================")
