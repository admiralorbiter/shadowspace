"""Generate theoretical surface grid and validate analytical distance formulas."""

import os
import numpy as np
import polars as pl
from shadowspace.ambiguity_atlas.geometry import (
    summary_entropy,
    hellinger_mirror_distance,
    fisher_rao_mirror_distance,
    js_mirror_distance,
    aitchison_mirror_distance,
    mirror_distribution,
)

OUTPUT_PATH = "results/ambiguity_atlas/theory_surface.parquet"


def generate_theory_surface():
    """Generate dense theoretical grid over majority_probability and minority_orientation."""
    print("=== Generating Theory Surface Grid ===")
    
    m_grid = np.linspace(0.34, 0.98, 65)
    delta_grid = np.linspace(-0.99, 0.99, 101)
    
    records = []
    for m in m_grid:
        for delta in delta_grid:
            # Check valid majority region
            max_delta_valid = 1.0 if m >= 0.5 else (3.0 * m - 1.0) / (1.0 - m)
            is_valid = abs(delta) <= max_delta_valid + 1e-9
            
            entropy_b = float(summary_entropy(m, delta))
            dh = float(hellinger_mirror_distance(m, delta))
            dfr = float(fisher_rao_mirror_distance(m, delta))
            djs = float(js_mirror_distance(m, delta))
            da = float(aitchison_mirror_distance(m, delta)) if abs(delta) < 1.0 else float("inf")
            
            records.append({
                "majority_probability": float(m),
                "minority_orientation": float(delta),
                "entropy_bits": entropy_b,
                "hellinger_distance": dh,
                "fisher_rao_distance": dfr,
                "js_distance": djs,
                "aitchison_distance": da,
                "valid_majority_region": bool(is_valid),
            })
            
    df = pl.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.write_parquet(OUTPUT_PATH)
    
    print(f"Theory surface written to {OUTPUT_PATH} ({df.height} points)")
    print(f"Valid majority points: {df.filter(pl.col('valid_majority_region')).height}")


if __name__ == "__main__":
    generate_theory_surface()
