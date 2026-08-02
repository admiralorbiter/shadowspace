"""Model predictions loading, temperature scaling, and pointwise calibration module."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.special import softmax

CANONICAL_MODEL_NAMES = (
    "albert-xxlarge",
    "bart-large",
    "bert-base",
    "bert-large",
    "distilbert",
    "roberta-base",
    "roberta-large",
    "xlnet-base",
    "xlnet-large",
)
DEFAULT_MODEL_PREDICTIONS_PATH = Path(
    "data/chaosnli/raw/model_predictions/model_predictions_for_snli_mnli.json"
)
DEFAULT_MODEL_PREDICTIONS_SHA256 = (
    "7c788bb9df6917f56a99a3bbf19c307bf2d02ad14d8f02537338b7bfb629bb53"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_predictions(
    model_json_path: Path = DEFAULT_MODEL_PREDICTIONS_PATH,
    canonical_items_path: Path = Path("data/chaosnli/processed/canonical_items_posterior.parquet"),
    *,
    allow_synthetic: bool = False,
    expected_model_names: Iterable[str] | None = CANONICAL_MODEL_NAMES,
    expected_sha256: str | None = DEFAULT_MODEL_PREDICTIONS_SHA256,
) -> dict[str, dict[str, np.ndarray]]:
    """Load real model logits aligned to every canonical item.

    Research-facing callers fail closed when the supplied artifact is absent,
    incomplete, or hash-mismatched. Synthetic predictions remain available only
    through the explicit ``allow_synthetic=True`` development escape hatch.

    Returns:
        Dict mapping model_name -> {
            'object_ids': List[str],
            'logits': (N, 3) float32 array [Entailment, Neutral, Contradiction]
        }
    """
    if not canonical_items_path.exists():
        canonical_items_path = Path("data/chaosnli/processed/canonical_items.parquet")

    canon_df = pl.read_parquet(canonical_items_path)
    canonical_uids = canon_df["object_id"].to_list()

    model_json_path = Path(model_json_path)
    if not model_json_path.exists():
        if not allow_synthetic:
            raise FileNotFoundError(
                f"Required real model artifact not found: {model_json_path}. "
                "Acquire and verify the ChaosNLI supplied predictions; synthetic "
                "fallback is disabled for research analyses."
            )

        model_names = list(expected_model_names or CANONICAL_MODEL_NAMES)
        model_results: dict[str, dict[str, np.ndarray]] = {}
        p_human = canon_df.select(
            ["human_p_entailment", "human_p_neutral", "human_p_contradiction"]
        ).to_numpy()

        for idx, m_name in enumerate(model_names):
            rng = np.random.default_rng(20260801 + idx)
            # Add model-specific noise to human distribution to generate realistic model logits
            noise_scale = 0.15 + 0.02 * idx
            noise = rng.normal(0, noise_scale, size=p_human.shape)
            noisy_p = np.clip(p_human + noise, 1e-4, None)
            noisy_p = noisy_p / np.sum(noisy_p, axis=1, keepdims=True)
            logits = np.log(noisy_p).astype(np.float32)

            model_results[m_name] = {
                "object_ids": np.array(canonical_uids),
                "logits": logits,
            }
        return model_results

    if expected_sha256 is not None:
        actual_sha256 = _sha256(model_json_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Model artifact SHA-256 mismatch for {model_json_path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

    with open(model_json_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    required_models = (
        list(expected_model_names) if expected_model_names is not None else sorted(raw_data)
    )
    missing_models = sorted(set(required_models) - set(raw_data))
    if missing_models:
        raise ValueError(f"Model artifact is missing required models: {missing_models}")

    model_results: dict[str, dict[str, np.ndarray]] = {}

    for model_name in required_models:
        preds = raw_data[model_name]
        logits_list = []
        missing_uids: list[str] = []

        for obj_id in canonical_uids:
            # Format of obj_id in canon_df is e.g. chaosnli_snli_2407214681.jpg#0r1n or chaosnli_mnli_50830c
            raw_uid = obj_id.replace("chaosnli_snli_", "").replace("chaosnli_mnli_", "")

            if raw_uid in preds:
                entry = preds[raw_uid]
                logits_list.append(entry["logits"])
            else:
                missing_uids.append(raw_uid)

        if missing_uids:
            preview = ", ".join(missing_uids[:5])
            raise ValueError(
                f"Model {model_name!r} is missing {len(missing_uids)} canonical item IDs "
                f"(first: {preview})"
            )

        logits = np.asarray(logits_list, dtype=np.float32)
        if logits.shape != (len(canonical_uids), 3):
            raise ValueError(
                f"Model {model_name!r} logits have shape {logits.shape}; "
                f"expected ({len(canonical_uids)}, 3) in [E, N, C] order"
            )
        if not np.isfinite(logits).all():
            raise ValueError(f"Model {model_name!r} contains non-finite logits")

        model_results[model_name] = {
            "object_ids": np.array(canonical_uids),
            "logits": logits,
        }

    return model_results


def compute_model_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax probabilities with optional temperature scaling.

    p = softmax(logits / T)
    """
    scaled_logits = logits / max(temperature, 1e-5)
    return softmax(scaled_logits, axis=-1)


def build_canonical_models_table(
    model_results: dict[str, dict[str, np.ndarray]],
    canonical_items_path: Path = Path("data/chaosnli/processed/canonical_items_posterior.parquet"),
    output_path: Path = Path("data/chaosnli/processed/canonical_models.parquet"),
) -> pl.DataFrame:
    """Construct canonical Parquet table containing probabilities and pointwise metrics for all models."""
    if not canonical_items_path.exists():
        canonical_items_path = Path("data/chaosnli/processed/canonical_items.parquet")

    canon_df = pl.read_parquet(canonical_items_path)
    p_human = canon_df.select(
        ["human_p_entailment", "human_p_neutral", "human_p_contradiction"]
    ).to_numpy()

    records: list[dict[str, Any]] = []

    for i, obj_id in enumerate(canon_df["object_id"]):
        p_i = p_human[i]

        for model_name, m_data in model_results.items():
            logits_i = m_data["logits"][i]
            q_i = compute_model_probabilities(logits_i, temperature=1.0)

            # Pointwise Hellinger distance
            d_h = float(np.sqrt(0.5 * np.sum((np.sqrt(p_i) - np.sqrt(q_i)) ** 2)))

            # Pointwise Jensen-Shannon Divergence
            m_mix = 0.5 * (p_i + q_i)
            kl_p = np.sum(np.where(p_i > 0, p_i * np.log(np.maximum(p_i, 1e-12) / m_mix), 0.0))
            kl_q = np.sum(np.where(q_i > 0, q_i * np.log(np.maximum(q_i, 1e-12) / m_mix), 0.0))
            jsd = float(0.5 * (kl_p + kl_q) / np.log(2.0))  # in bits

            # Brier Score
            brier = float(np.sum((q_i - p_i) ** 2))

            records.append(
                {
                    "object_id": obj_id,
                    "model_name": model_name,
                    "logit_entailment": float(logits_i[0]),
                    "logit_neutral": float(logits_i[1]),
                    "logit_contradiction": float(logits_i[2]),
                    "model_p_entailment": float(q_i[0]),
                    "model_p_neutral": float(q_i[1]),
                    "model_p_contradiction": float(q_i[2]),
                    "pointwise_hellinger": d_h,
                    "pointwise_jsd_bits": jsd,
                    "pointwise_brier": brier,
                }
            )

    model_df = pl.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_df.write_parquet(output_path)
    return model_df
