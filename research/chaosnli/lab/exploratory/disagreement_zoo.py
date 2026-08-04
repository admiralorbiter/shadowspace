"""Synthetic Disagreement Zoo — Testing metric behavior across canonical distribution shapes.
"""

import os
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import json
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from metric_atlas import (
    distance_hellinger_matrix,
    distance_fisher_rao_matrix,
    distance_jsd_matrix,
    distance_aitchison_matrix,
)

ARCHETYPES = {
    "consensus_entailment": np.array([1.0, 0.0, 0.0]),
    "consensus_neutral": np.array([0.0, 1.0, 0.0]),
    "consensus_contradiction": np.array([0.0, 0.0, 1.0]),
    "binary_ambiguity_EN": np.array([0.5, 0.5, 0.0]),
    "binary_ambiguity_EC": np.array([0.5, 0.0, 0.5]),
    "binary_ambiguity_NC": np.array([0.0, 0.5, 0.5]),
    "diffuse_ambiguity": np.array([1/3, 1/3, 1/3]),
    "dominant_shoulder_EN": np.array([0.7, 0.25, 0.05]),
    "dominant_shoulder_EC": np.array([0.7, 0.05, 0.25]),
}

def run_disagreement_zoo() -> dict:
    labels = list(ARCHETYPES.keys())
    P = np.array(list(ARCHETYPES.values()), dtype=np.float64)
    
    d_hellinger = distance_hellinger_matrix(P)
    d_fisher_rao = distance_fisher_rao_matrix(P)
    d_jsd = distance_jsd_matrix(P)
    d_aitchison = distance_aitchison_matrix(P)
    
    results = {}
    for i, name1 in enumerate(labels):
        results[name1] = {}
        for j, name2 in enumerate(labels):
            results[name1][name2] = {
                "hellinger": float(d_hellinger[i, j]),
                "fisher_rao": float(d_fisher_rao[i, j]),
                "jsd": float(d_jsd[i, j]),
                "aitchison": float(d_aitchison[i, j]),
            }
            
    summary = {
        "archetypes": {k: v.tolist() for k, v in ARCHETYPES.items()},
        "pairwise_distances": results,
    }
    return summary

if __name__ == "__main__":
    out_dir = Path("research/chaosnli/artifacts/exploratory")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = run_disagreement_zoo()
    out_file = out_dir / "disagreement_zoo_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Disagreement Zoo summary written to {out_file}")
