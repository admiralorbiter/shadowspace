"""LABE-Trained Sparse N-Gram Ensemble Evaluator (Logistic + Gradient Boosting)."""

from __future__ import annotations

import re
from typing import Any, Dict, List
import numpy as np

from research.education_audit.agency_classifier_validation.labe_classifier_trainer import train_and_evaluate_labe_classifier
from research.education_audit.audit_reliability_empirical.provenance import EvaluatorProvenance


class SparseNgramEnsembleEvaluator:
    """LABE-Trained Sparse N-Gram Ensemble Evaluator."""

    def __init__(self, model_artifacts: Dict[str, Any]):
        self.vectorizer = model_artifacts["vectorizer"]
        self.clf_lr = model_artifacts["clf_lr"]
        self.clf_gb = model_artifacts["clf_gb"]
        self.threshold = model_artifacts["best_threshold"]

        self.provenance = EvaluatorProvenance(
            evaluator_id="eval_sparse_ngram_ensemble",
            evaluator_name="Sparse N-Gram Baseline Ensemble",
            model_family="sklearn_tfidf_lr_gb_ensemble",
            checkpoint_revision="labe_train_split_v1",
            checkpoint_sha256="labe_ngram_ensemble_hash_v1",
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
