"""E004 Pilot Human Support Target & Baseline Matrix Generator.

Constructs 600x600 Hellinger posterior edge-support matrices (k=10 primary, k=50 core)
from 500 Dirichlet posterior draws for the 600 pilot items.
Computes pilot-specific human-human split-half reference Q_HH and extracts
baseline classifier probability matrices (BART, RoBERTa, XLNet, E003 Ensemble).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

MANIFEST_DIR = Path("research/chaosnli/artifacts/E004/manifests")
PILOT_SUPPORT_DIR = Path("research/chaosnli/artifacts/E004/pilot_support")
E001_DIR = Path("research/chaosnli/artifacts/E001")

def compute_dirichlet_draws(counts: np.ndarray, alpha: Tuple[float, float, float] = (0.5, 0.5, 0.5), n_draws: int = 500, seed: int = 42) -> np.ndarray:
    """Generate (N, n_draws, 3) Dirichlet posterior probability draws."""
    N = len(counts)
    draws = np.zeros((N, n_draws, 3), dtype=np.float64)
    rng = np.random.default_rng(seed)

    for i in range(N):
        alpha_post = counts[i] + np.array(alpha)
        draws[i] = rng.dirichlet(alpha_post, size=n_draws)

    return draws

def compute_topk_weight_matrix(dist: np.ndarray, k: int) -> np.ndarray:
    """Compute tie-aware soft top-k neighbor weight matrix W[i, j] in [0, 1]."""
    N = dist.shape[0]
    ATOL = 1e-7
    dist_self = dist.copy()
    np.fill_diagonal(dist_self, np.inf)

    k_dists = np.partition(dist_self, k - 1, axis=1)[:, k - 1, np.newaxis]

    closer_mask = dist_self < (k_dists - ATOL)
    tied_mask = np.abs(dist_self - k_dists) <= ATOL

    n_closer = np.sum(closer_mask, axis=1, keepdims=True)
    n_tied = np.sum(tied_mask, axis=1, keepdims=True)

    frac = np.where(n_tied > 0, (k - n_closer) / np.maximum(1.0, n_tied.astype(float)), 0.0)

    W = np.where(closer_mask, 1.0, np.where(tied_mask, frac, 0.0))
    np.fill_diagonal(W, 0.0)
    return W

def distance_hellinger_matrix(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Compute pairwise Hellinger distance matrix."""
    sqrt_P = np.sqrt(np.clip(P, 1e-12, 1.0))
    sqrt_Q = np.sqrt(np.clip(Q, 1e-12, 1.0))
    bc = np.dot(sqrt_P, sqrt_Q.T)
    bc = np.clip(bc, 0.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - bc))

def compute_posterior_support_matrix(draws: np.ndarray, k: int) -> np.ndarray:
    """Compute posterior edge support S_ij(k) = E[W_ij(k)]."""
    N, n_draws, _ = draws.shape
    S_sum = np.zeros((N, N), dtype=np.float64)

    for d in range(n_draws):
        P_d = draws[:, d, :]
        dist_d = distance_hellinger_matrix(P_d, P_d)
        W_d = compute_topk_weight_matrix(dist_d, k)
        S_sum += W_d

    return S_sum / float(n_draws)

def build_pilot_support_matrices(subset: str = "pilot") -> None:
    manifest_file = MANIFEST_DIR / f"{subset}_600.jsonl" if subset == "pilot" else MANIFEST_DIR / f"{subset}_60.jsonl"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_file}")

    items = []
    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    N = len(items)
    counts = np.array([[it["human_count_entailment"], it["human_count_neutral"], it["human_count_contradiction"]] for it in items])

    print("=========================================================================")
    print(f"   GENERATING PILOT HUMAN SUPPORT MATRIX ({N} items, 500 draws)")
    print("=========================================================================")

    draws = compute_dirichlet_draws(counts, alpha=(0.5, 0.5, 0.5), n_draws=500, seed=42)

    # Compute primary k=10 and core k=50 support matrices
    print("  Computing S_hellinger_k010_pilot...")
    S_k10 = compute_posterior_support_matrix(draws, k=10)
    print("  Computing S_hellinger_k050_pilot...")
    S_k50 = compute_posterior_support_matrix(draws, k=50)

    # Split-half human-human reference Q_HH on pilot
    draws_a = draws[:, :250, :]
    draws_b = draws[:, 250:, :]
    S_a = compute_posterior_support_matrix(draws_a, k=10)
    S_b = compute_posterior_support_matrix(draws_b, k=10)
    q_hh = float(np.sum(np.minimum(S_a, S_b)) / (N * 10.0))
    print(f"  Pilot Human-Human Relational Reference Q_HH (k=10) = {q_hh:.5f}")

    PILOT_SUPPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save binaries and manifests
    k10_bin = PILOT_SUPPORT_DIR / f"S_hellinger_k010_{subset}.bin"
    k50_bin = PILOT_SUPPORT_DIR / f"S_hellinger_k050_{subset}.bin"

    f32_k10 = S_k10.astype(np.float32).tobytes()
    f32_k50 = S_k50.astype(np.float32).tobytes()

    k10_bin.write_bytes(f32_k10)
    k50_bin.write_bytes(f32_k50)

    k10_hash = hashlib.sha256(f32_k10).hexdigest()
    k50_hash = hashlib.sha256(f32_k50).hexdigest()

    manifest_k10 = {
        "artifact_id": f"E004-pilot-hellinger-k010-v1",
        "matrix_sha256": k10_hash,
        "object_count": N,
        "k": 10,
        "q_hh_relational": q_hh
    }
    with open(PILOT_SUPPORT_DIR / f"S_hellinger_k010_{subset}.manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_k10, f, indent=2)

    # Slice existing canonical models (from model_probs.json in E003 artifact / data) if available
    try:
        from shadowspace.chaosnli.models import load_model_predictions
        preds = load_model_predictions(allow_synthetic=True)
        row_indices = [it["row_index"] for it in items]

        classifier_probs = {}
        for m_name, m_data in preds.items():
            if "probs" in m_data:
                p_full = m_data["probs"]
                classifier_probs[m_name] = p_full[row_indices]

        if "bart-large" in classifier_probs and "roberta-large" in classifier_probs and "xlnet-large" in classifier_probs:
            ens_probs = (classifier_probs["bart-large"] + classifier_probs["roberta-large"] + classifier_probs["xlnet-large"]) / 3.0
            classifier_probs["e003_equal_ensemble"] = ens_probs

        np.save(PILOT_SUPPORT_DIR / f"baseline_classifiers_{subset}_probs.npy", classifier_probs)
        print(f"  Extracted {len(classifier_probs)} baseline classifier probability matrices on pilot subset.")
    except Exception as e:
        print(f"  Warning: Could not load baseline classifier predictions: {e}")

    print("=========================================================================")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["preflight", "pilot"], default="pilot")
    args = parser.parse_args()
    build_pilot_support_matrices(args.subset)

if __name__ == "__main__":
    main()
