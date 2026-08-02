"""Finite-annotation uncertainty estimation and split-half reliability module."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def compute_dirichlet_posteriors(
    counts: np.ndarray,
    alpha: tuple[float, float, float] = (0.5, 0.5, 0.5),
    n_draws: int = 2000,
    seed: int = 20260801,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate Dirichlet Monte Carlo draws and posterior summary metrics.

    Args:
        counts: (N, 3) array of integer label counts (entailment, neutral, contradiction).
        alpha: Prior Dirichlet parameters (default: Jeffreys prior (0.5, 0.5, 0.5)).
        n_draws: Number of Monte Carlo draws per item.
        seed: Random seed.

    Returns:
        draws: (N, n_draws, 3) float array of probability draws.
        summaries: Dictionary of summary arrays per item.
    """
    rng = np.random.default_rng(seed)
    n_items = len(counts)

    # Posterior Dirichlet parameters alpha_post = counts + alpha
    alpha_post = counts.astype(np.float64) + np.array(alpha, dtype=np.float64)

    # Vectorized Gamma sampling for Dirichlet draws: Gamma(shape=alpha_post, scale=1.0)
    gamma_draws = rng.gamma(shape=np.tile(alpha_post[:, np.newaxis, :], (1, n_draws, 1)))
    draws = gamma_draws / gamma_draws.sum(axis=-1, keepdims=True)  # (N, n_draws, 3)

    # Posterior means
    post_mean_p = draws.mean(axis=1)  # (N, 3)

    # Entropy in bits per draw: H = -sum(p * log2(p))
    # Clip probabilities to avoid log2(0)
    draws_clipped = np.clip(draws, 1e-12, 1.0)
    entropy_draws = -np.sum(draws_clipped * np.log2(draws_clipped), axis=-1)  # (N, n_draws)

    entropy_mean = entropy_draws.mean(axis=1)
    entropy_q025 = np.quantile(entropy_draws, 0.025, axis=1)
    entropy_q975 = np.quantile(entropy_draws, 0.975, axis=1)

    # Posterior majority probabilities
    # majority E: p_E > p_N and p_E > p_C
    maj_e = (draws[:, :, 0] > draws[:, :, 1]) & (draws[:, :, 0] > draws[:, :, 2])
    maj_n = (draws[:, :, 1] > draws[:, :, 0]) & (draws[:, :, 1] > draws[:, :, 2])
    maj_c = (draws[:, :, 2] > draws[:, :, 0]) & (draws[:, :, 2] > draws[:, :, 1])

    p_maj_e = maj_e.mean(axis=1)
    p_maj_n = maj_n.mean(axis=1)
    p_maj_c = maj_c.mean(axis=1)
    p_max_maj = np.maximum.reduce([p_maj_e, p_maj_n, p_maj_c])

    summaries = {
        "posterior_mean_p_entailment": post_mean_p[:, 0],
        "posterior_mean_p_neutral": post_mean_p[:, 1],
        "posterior_mean_p_contradiction": post_mean_p[:, 2],
        "posterior_entropy_mean": entropy_mean,
        "posterior_entropy_q025": entropy_q025,
        "posterior_entropy_q975": entropy_q975,
        "p_majority_entailment": p_maj_e,
        "p_majority_neutral": p_maj_n,
        "p_majority_contradiction": p_maj_c,
        "p_max_majority": p_max_maj,
    }

    return draws, summaries


def compute_split_half_distributions(
    counts: np.ndarray,
    seed: int = 20260801,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample two 50-vote split-half empirical probability distributions from 100-vote counts.

    Args:
        counts: (N, 3) integer count matrix (must sum to 100 per row).
        seed: Random seed.

    Returns:
        p_half1: (N, 3) empirical probability array for half 1.
        p_half2: (N, 3) empirical probability array for half 2.
    """
    rng = np.random.default_rng(seed)
    n_items = len(counts)

    half1_counts = np.zeros((n_items, 3), dtype=np.int64)
    half2_counts = np.zeros((n_items, 3), dtype=np.int64)

    for i in range(n_items):
        row_c = counts[i]
        total = int(row_c.sum())
        if total == 0:
            continue
        # Construct full list of category indices
        labels = np.repeat(np.array([0, 1, 2]), row_c)
        rng.shuffle(labels)

        # Split 50 / 50
        n_half = total // 2
        h1 = labels[:n_half]
        h2 = labels[n_half:total]

        half1_counts[i] = np.bincount(h1, minlength=3)
        half2_counts[i] = np.bincount(h2, minlength=3)

    p_half1 = half1_counts / np.maximum(half1_counts.sum(axis=1, keepdims=True), 1)
    p_half2 = half2_counts / np.maximum(half2_counts.sum(axis=1, keepdims=True), 1)

    return p_half1, p_half2


def compute_100_vs_100_posterior_predictive_reliability(
    counts: np.ndarray,
    n_votes: int = 100,
    alpha_prior: float = 0.5,
    seed: int = 20260801,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two independent 100-vote Dirichlet-Multinomial posterior predictive replicates.

    theta_i ~ Dirichlet(x_i + alpha)
    x_i1, x_i2 ~ Multinomial(100, theta_i)
    """
    rng = np.random.default_rng(seed)
    alpha = counts.astype(np.float64) + alpha_prior

    # Draw latent theta_i per item
    gamma_draws = rng.gamma(shape=alpha)
    theta = gamma_draws / gamma_draws.sum(axis=-1, keepdims=True)

    # Draw two independent 100-vote samples
    x1 = np.array([rng.multinomial(n_votes, p) for p in theta], dtype=np.float64)
    x2 = np.array([rng.multinomial(n_votes, p) for p in theta], dtype=np.float64)

    p1 = x1 / float(n_votes)
    p2 = x2 / float(n_votes)
    return p1, p2


def run_posterior_pipeline(
    canonical_parquet: Path = Path("data/chaosnli/processed/canonical_items.parquet"),
    output_dir: Path = Path("data/chaosnli/processed"),
    n_draws: int = 2000,
    seed: int = 20260801,
) -> dict[str, Any]:
    """Run full posterior calculation and append posterior summaries to canonical items."""
    if not canonical_parquet.exists():
        raise FileNotFoundError(f"Canonical dataset missing at {canonical_parquet}")

    df = pl.read_parquet(canonical_parquet)

    counts = df.select(
        ["human_count_entailment", "human_count_neutral", "human_count_contradiction"]
    ).to_numpy()

    draws, summaries = compute_dirichlet_posteriors(counts, n_draws=n_draws, seed=seed)

    # Append summary columns to dataframe
    for col_name, arr in summaries.items():
        df = df.with_columns(pl.Series(col_name, arr))

    out_parquet = output_dir / "canonical_items_posterior.parquet"
    df.write_parquet(out_parquet)

    # Save summary stats
    results = {
        "output_path": str(out_parquet),
        "n_items": len(df),
        "n_draws": n_draws,
        "mean_posterior_entropy": float(df["posterior_entropy_mean"].mean()),
        "mean_max_majority_prob": float(df["p_max_majority"].mean()),
    }

    return results
