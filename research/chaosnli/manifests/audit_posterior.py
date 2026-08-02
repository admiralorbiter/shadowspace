import polars as pl
import numpy as np

df = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")

print("=== DETAILED POSTERIOR AUDIT REPORT ===")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# Check nulls per column
nulls = {col: df[col].null_count() for col in df.columns if df[col].null_count() > 0}
print(f"Columns with null values: {nulls if nulls else 'None (0 nulls across dataset)'}")

print("\n--- Entropy Distributions (Bits) ---")
emp_ent = df["human_entropy_bits"].to_numpy()
post_ent = df["posterior_entropy_mean"].to_numpy()
ci_low = df["posterior_entropy_q025"].to_numpy()
ci_high = df["posterior_entropy_q975"].to_numpy()
ci_width = ci_high - ci_low

print(f"Empirical Entropy  : Mean = {emp_ent.mean():.4f}, Min = {emp_ent.min():.4f}, Max = {emp_ent.max():.4f}")
print(f"Posterior Entropy  : Mean = {post_ent.mean():.4f}, Min = {post_ent.min():.4f}, Max = {post_ent.max():.4f}")
print(f"95% CI Width       : Mean = {ci_width.mean():.4f} bits (Range: {ci_width.min():.4f} to {ci_width.max():.4f})")

print("\n--- Posterior Majority Certainty ---")
p_max = df["p_max_majority"].to_numpy()
print(f"Mean P(majority)   : {p_max.mean():.4f}")
print(f"P(majority) >= 0.99: {(p_max >= 0.99).sum():>5} items ({(p_max >= 0.99).mean()*100:.1f}%)")
print(f"P(majority) >= 0.95: {(p_max >= 0.95).sum():>5} items ({(p_max >= 0.95).mean()*100:.1f}%)")
print(f"0.80 <= P < 0.95   : {((p_max >= 0.80) & (p_max < 0.95)).sum():>5} items ({((p_max >= 0.80) & (p_max < 0.95)).mean()*100:.1f}%)")
print(f"0.50 <= P < 0.80   : {((p_max >= 0.50) & (p_max < 0.80)).sum():>5} items ({((p_max >= 0.50) & (p_max < 0.80)).mean()*100:.1f}%)")
print(f"P(majority) < 0.50 : {(p_max < 0.50).sum():>5} items ({(p_max < 0.50).mean()*100:.1f}%) [Trinary ambiguous]")

print("\n--- Dataset Subgroups ---")
for ds in ["chaosnli_snli", "chaosnli_mnli"]:
    sub = df.filter(pl.col("source_dataset") == ds)
    print(f"{ds:<15}: Items = {len(sub)}, Empirical Entropy = {sub['human_entropy_bits'].mean():.4f}, Mean P(majority) = {sub['p_max_majority'].mean():.4f}")

print("\n--- Posterior Means vs Empirical Means Check ---")
e_diff = np.abs(df["human_p_entailment"].to_numpy() - df["posterior_mean_p_entailment"].to_numpy()).max()
n_diff = np.abs(df["human_p_neutral"].to_numpy() - df["posterior_mean_p_neutral"].to_numpy()).max()
c_diff = np.abs(df["human_p_contradiction"].to_numpy() - df["posterior_mean_p_contradiction"].to_numpy()).max()
print(f"Max absolute difference (Empirical vs Dirichlet Posterior Mean): {max(e_diff, n_diff, c_diff):.6f}")
print("Expected behavior: Dirichlet(alpha=0.5) smoothly regularizes exact 0 count probabilities to ~0.00495 (0.5 / 101.5), preventing zero-probability pathologies in log distances.")
