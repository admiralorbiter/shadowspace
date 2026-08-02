import polars as pl

df = pl.read_parquet("data/chaosnli/processed/canonical_items_posterior.parquet")

print("=== EXAMPLES FOR STUDY 1 REPORT ===\n")

# 1. Exact count ties e.g. E=50, N=50
print("--- EXAMPLE TYPE A: Exact Count Ties (e.g. 50 Entailment / 50 Neutral / 0 Contradiction) ---")
ties = df.filter((pl.col("human_count_entailment") == 50) & (pl.col("human_count_neutral") == 50))
print(f"Total items with exact (50, 50, 0) counts: {len(ties)}\n")
for i, r in enumerate(ties.head(3).iter_rows(named=True), 1):
    print(f"Example A{i} [ID: {r['object_id']}]")
    print(f"  Premise   : \"{r['premise']}\"")
    print(f"  Hypothesis: \"{r['hypothesis']}\"")
    print(f"  Counts    : E={r['human_count_entailment']}, N={r['human_count_neutral']}, C={r['human_count_contradiction']}")
    print(f"  Entropy   : {r['human_entropy_bits']:.4f} bits")
    print()

# 2. Trinary ambiguous items (P_max < 0.50)
print("--- EXAMPLE TYPE B: Trinary Ambiguous Items (No single majority label) ---")
trinary = df.filter(pl.col("p_max_majority") < 0.50)
print(f"Total trinary ambiguous items: {len(trinary)}\n")
for i, r in enumerate(trinary.iter_rows(named=True), 1):
    print(f"Example B{i} [ID: {r['object_id']}]")
    print(f"  Premise   : \"{r['premise']}\"")
    print(f"  Hypothesis: \"{r['hypothesis']}\"")
    print(f"  Counts    : E={r['human_count_entailment']}, N={r['human_count_neutral']}, C={r['human_count_contradiction']}")
    print(f"  Entropy   : {r['human_entropy_bits']:.4f} bits")
    print(f"  P(max_maj): {r['p_max_majority']:.4f}")
    print()

# 3. High-entropy item vs Zero-count high-entropy item
print("--- EXAMPLE TYPE C: High Entropy with Zero Count vs Balanced 3-Way ---")
zero_high = df.filter(pl.col("has_zero_count") & (pl.col("human_entropy_bits") > 0.98)).sort("human_entropy_bits", descending=True)
for i, r in enumerate(zero_high.head(2).iter_rows(named=True), 1):
    print(f"Example C{i} [ID: {r['object_id']}]")
    print(f"  Premise   : \"{r['premise']}\"")
    print(f"  Hypothesis: \"{r['hypothesis']}\"")
    print(f"  Counts    : E={r['human_count_entailment']}, N={r['human_count_neutral']}, C={r['human_count_contradiction']}")
    print(f"  Entropy   : {r['human_entropy_bits']:.4f} bits (Zero Count: {r['has_zero_count']})")
    print()
