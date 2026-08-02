import numpy as np
import polars as pl

from shadowspace.chaosnli.joint_spaces import evaluate_hypothesis7_joint_space

df = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")

d_opinion = np.load("data/chaosnli/processed/distance_matrix_human_hellinger.npy")
d_text = np.load("data/chaosnli/processed/distance_matrix_text_cosine.npy")

print("=========================================================================")
print("         HYPOTHESIS 7 EVALUATION: JOINT OPINION-TEXT SPACES              ")
print("=========================================================================\n")

h7_res = evaluate_hypothesis7_joint_space(df, d_opinion, d_text, lambdas=[0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0], k=10)

print(f"Total Items                        : {h7_res['n_items']}")
print(f"Multi-Item Opinion Profiles        : {h7_res['n_multi_item_profiles']}")
print(f"Mean Intra-Profile Text Distance  : {h7_res['mean_intra_profile_text_distance']:.4f} (Cosine)")
print(f"Mean Overall Dataset Text Distance : {h7_res['mean_overall_text_distance']:.4f} (Cosine)")

print("\n--- Multi-View Lambda Blend Curve ---")
print("  Lambda (Text Wt) | Zero Distance Ties Remaining | Soft Q_NX Opinion Recovery")
print("  -------------------------------------------------------------------------")
for row in h7_res["lambda_evaluations"]:
    lam = row["lambda"]
    ties = row["zero_distance_ties_remaining"]
    qnx_rec = row["qnx_soft_opinion_recovery"]
    print(f"  {lam:<16.2f} | {ties:<28} | {qnx_rec:.4f}")

print("\nHypothesis 7 Confirmed:", h7_res["h7_confirmed"])
print("=========================================================================")
