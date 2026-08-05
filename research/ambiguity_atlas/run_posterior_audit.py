"""Run Dirichlet posterior stability audit on strict ambiguity doppelgängers."""

import os
import polars as pl
import numpy as np
from shadowspace.ambiguity_atlas.posterior import audit_pair_posterior_stability

CANONICAL_PATH = "data/chaosnli/processed/canonical_items.parquet"
STRICT_PAIRS_PATH = "results/ambiguity_atlas/strict_pairs.parquet"
OUTPUT_PATH = "results/ambiguity_atlas/posterior_stability.parquet"


def run_posterior_audit():
    """Execute Dirichlet posterior uncertainty audit using fixed original majority estimand."""
    print("=== Running Dirichlet Posterior Stability Audit (Joint Estimand) ===")
    
    df_canon = pl.read_parquet(CANONICAL_PATH)
    df_strict = pl.read_parquet(STRICT_PAIRS_PATH)
    
    if df_strict.height == 0:
        print("No strict pairs available for posterior audit.")
        return
        
    canon_dict = {row["object_id"]: row for row in df_canon.to_dicts()}
    
    records = []
    print(f"Auditing joint posterior stability across {df_strict.height} strict pairs...")
    
    for pair in df_strict.to_dicts():
        obj_a = canon_dict[pair["object_id_a"]]
        obj_b = canon_dict[pair["object_id_b"]]
        
        counts_a = np.array([
            obj_a["human_count_entailment"],
            obj_a["human_count_neutral"],
            obj_a["human_count_contradiction"]
        ], dtype=np.int64)
        
        counts_b = np.array([
            obj_b["human_count_entailment"],
            obj_b["human_count_neutral"],
            obj_b["human_count_contradiction"]
        ], dtype=np.int64)
        
        maj_lbl = pair["majority_label"]
        maj_idx = 0 if maj_lbl == "entailment" else (1 if maj_lbl == "neutral" else 2)
        
        res = audit_pair_posterior_stability(
            counts_a, counts_b,
            majority_idx=maj_idx,
            pair_id=pair["pair_id"],
            n_draws=2000,
            alpha=0.5,
        )
        
        record = dict(pair)
        record.update(res)
        records.append(record)

    df_res = pl.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_res.write_parquet(OUTPUT_PATH)
    
    cats = df_res["stability_category"].value_counts().to_dicts()
    print("Posterior Stability Classification Breakdown:")
    for c in cats:
        print(f"  - {c['stability_category']}: {c['count']} pairs")
        
    print(f"Posterior stability audit written to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_posterior_audit()
