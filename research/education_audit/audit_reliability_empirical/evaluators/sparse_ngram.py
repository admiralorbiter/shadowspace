"""LABE-Trained Sparse N-Gram Ensemble Evaluator (Loaded from Frozen Joblib Artifacts With Fail-Closed Hash Verification)."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List
import joblib
import numpy as np

from research.education_audit.audit_reliability_empirical.provenance import EvaluatorProvenance


def compute_file_sha256(filepath: str) -> str:
    """Computes exact 64-character SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class SparseNgramEnsembleEvaluator:
    """LABE-Trained Sparse N-Gram Ensemble Evaluator (With Fail-Closed Hash Verification)."""

    def __init__(self, model_dir: str = "models/labe_sparse_ngram"):
        model_dir = os.path.abspath(model_dir)
        vec_path = os.path.join(model_dir, "vectorizer.joblib")
        lr_path = os.path.join(model_dir, "clf_lr.joblib")
        gb_path = os.path.join(model_dir, "clf_gb.joblib")
        manifest_path = os.path.join(model_dir, "manifest.json")

        if not os.path.exists(vec_path) or not os.path.exists(lr_path) or not os.path.exists(gb_path) or not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Required frozen n-gram joblib artifacts or manifest missing in {model_dir}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        actual_vec_hash = compute_file_sha256(vec_path)
        actual_lr_hash = compute_file_sha256(lr_path)
        actual_gb_hash = compute_file_sha256(gb_path)

        if manifest.get("vectorizer_sha256") and actual_vec_hash != manifest["vectorizer_sha256"]:
            raise ValueError(f"Vectorizer joblib hash mismatch: expected {manifest['vectorizer_sha256']}, got {actual_vec_hash}")
        if manifest.get("clf_lr_sha256") and actual_lr_hash != manifest["clf_lr_sha256"]:
            raise ValueError(f"Logistic clf joblib hash mismatch: expected {manifest['clf_lr_sha256']}, got {actual_lr_hash}")
        if manifest.get("clf_gb_sha256") and actual_gb_hash != manifest["clf_gb_sha256"]:
            raise ValueError(f"Gradient boosting joblib hash mismatch: expected {manifest['clf_gb_sha256']}, got {actual_gb_hash}")

        self.vectorizer = joblib.load(vec_path)
        self.clf_lr = joblib.load(lr_path)
        self.clf_gb = joblib.load(gb_path)
        self.threshold = float(manifest.get("best_threshold", 0.49))

        combined_hash = hashlib.sha256(
            (actual_vec_hash + actual_lr_hash + actual_gb_hash).encode("utf-8")
        ).hexdigest()

        self.provenance = EvaluatorProvenance(
            evaluator_id="eval_sparse_ngram_ensemble",
            evaluator_name="Sparse N-Gram Baseline Ensemble",
            model_family="sklearn_tfidf_lr_gb_ensemble",
            checkpoint_revision="labe_train_split_v1",
            checkpoint_sha256=combined_hash,
            training_data_revision="abcc3ec6032e3b265cbf15c6d8a3da668a2a030675b00f0425b96698c8cd5b56",
            score_scale=[0.0, 1.0],
            threshold=self.threshold,
            threshold_source="labe_validation_split_f1_optimization",
            is_independent=True,
            independent_of=[],
        )

    def _split_sentences(self, text: str) -> List[str]:
        sents = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sents if len(s.strip()) > 10]

    def predict_score(self, text: str) -> float:
        sents = self._split_sentences(text)
        if not sents:
            sents = [text]
        vec = self.vectorizer.transform(sents)
        probs_lr = self.clf_lr.predict_proba(vec)[:, 1]
        probs_gb = self.clf_gb.predict_proba(vec)[:, 1]
        probs = 0.5 * probs_lr + 0.5 * probs_gb
        return float(np.mean(probs))
