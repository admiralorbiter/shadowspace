"""Text embedding extraction and text-distance space module for ChaosNLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist


def extract_text_embeddings(
    canon_df: pl.DataFrame,
    method: str = "sentence-transformer",
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[np.ndarray, str]:
    """Extract dense sentence embeddings for premise-hypothesis pairs.

    Format: Premise + " [SEP] " + Hypothesis
    Returns (embeddings, resolved_method_name).
    """
    texts = [
        f"{r['premise']} [SEP] {r['hypothesis']}" for r in canon_df.iter_rows(named=True)
    ]

    if method == "sentence-transformer":
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embeddings.astype(np.float32), f"sentence-transformer:{model_name}"
        except Exception as exc:
            raise RuntimeError(
                f"Failed to extract sentence-transformer embeddings with model {model_name}. "
                "Specify method='tfidf-svd' explicitly if TF-IDF representation is desired."
            ) from exc

    elif method == "tfidf-svd":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        x_tfidf = tfidf.fit_transform(texts)
        svd = TruncatedSVD(n_components=min(128, x_tfidf.shape[1] - 1), random_state=42)
        x_emb = svd.fit_transform(x_tfidf)
        # Normalize to unit length
        norms = np.linalg.norm(x_emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (x_emb / norms).astype(np.float32), "tfidf-svd:5000-128"

    else:
        raise ValueError(f"Unknown text embedding method '{method}'. Supported: 'sentence-transformer', 'tfidf-svd'.")


def compute_text_cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute NxN Cosine distance matrix for text embeddings."""
    dist_mat = cdist(embeddings, embeddings, metric="cosine").astype(np.float32)
    np.fill_diagonal(dist_mat, 0.0)
    return np.clip(dist_mat, 0.0, 2.0)


def build_text_distance_space(
    canonical_items_path: Path = Path("data/chaosnli/processed/canonical_items_posterior.parquet"),
    output_dir: Path = Path("data/chaosnli/processed"),
    method: str = "sentence-transformer",
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Build and save text embedding distance matrix."""
    if not canonical_items_path.exists():
        canonical_items_path = Path("data/chaosnli/processed/canonical_items.parquet")

    canon_df = pl.read_parquet(canonical_items_path)
    embeddings, resolved_method = extract_text_embeddings(
        canon_df, method=method, model_name=model_name
    )

    dist_matrix = compute_text_cosine_distance_matrix(embeddings)

    output_file = output_dir / "distance_matrix_text_cosine.npy"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_file, dist_matrix)

    emb_filename = "text_embeddings_minilm.npy" if "sentence-transformer" in resolved_method else "text_embeddings_tfidf_svd.npy"
    emb_file = output_dir / emb_filename
    np.save(emb_file, embeddings)

    return {
        "n_items": len(canon_df),
        "embedding_dim": embeddings.shape[1],
        "method": resolved_method,
        "dist_matrix_path": str(output_file),
        "embedding_path": str(emb_file),
    }

