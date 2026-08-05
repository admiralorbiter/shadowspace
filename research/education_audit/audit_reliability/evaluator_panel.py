"""AR-1: Unified Multi-Instrument Evaluator Panel Architecture (Scaffold Prototype)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Protocol
import numpy as np

from research.education_audit.automated_text_audit.feature_registry import extract_lexical_features
from research.education_audit.agency_classifier_validation.labe_classifier_trainer import train_and_evaluate_labe_classifier


class BaseAgencyEvaluator(Protocol):
    """Protocol for agency evaluators in the panel."""
    evaluator_id: str
    evaluator_name: str
    evaluator_type: str
    is_independent: bool

    def predict_score(self, text: str) -> float:
        ...


class ExactLexiconEvaluator:
    """Exact Keyword Density Agency Evaluator (Zero-Drift Baseline Control)."""
    evaluator_id = "eval_exact_lexicon"
    evaluator_name = "Exact Lexicon Density"
    evaluator_type = "exact_lexicon"
    is_independent = True

    def predict_score(self, text: str) -> float:
        feats = extract_lexical_features(text)
        return float(feats["agentic_density"] + feats["leadership_density"])


class SparseNgramEnsembleEvaluator:
    """LABE-Trained Sparse N-Gram Logistic + Gradient Boosting Ensemble Evaluator."""
    evaluator_id = "eval_sparse_ngram_ensemble"
    evaluator_name = "Sparse N-Gram Baseline Ensemble"
    evaluator_type = "sparse_ngram_ensemble"
    is_independent = True

    def __init__(self, model_artifacts: Dict[str, Any]):
        self.vectorizer = model_artifacts["vectorizer"]
        self.clf_lr = model_artifacts["clf_lr"]
        self.clf_gb = model_artifacts["clf_gb"]
        self.threshold = model_artifacts["best_threshold"]

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


class LengthAdjustedNgramProxy:
    """Length-Adjusted N-Gram Proxy Evaluator (Non-Independent Diagnostic Transformation)."""
    evaluator_id = "eval_length_adjusted_ngram_proxy"
    evaluator_name = "Length-Adjusted N-Gram Proxy (Non-Independent)"
    evaluator_type = "length_adjusted_ngram_proxy"
    is_independent = False

    def __init__(self, ngram_evaluator: SparseNgramEnsembleEvaluator):
        self.ngram_evaluator = ngram_evaluator

    def predict_score(self, text: str) -> float:
        base_score = self.ngram_evaluator.predict_score(text)
        words = text.split()
        length_penalty = min(1.0, len(words) / 30.0)
        return float(np.clip(base_score * 0.95 + 0.05 * length_penalty, 0.0, 1.0))


def initialize_evaluator_panel() -> Dict[str, Any]:
    """Initializes and returns the evaluator panel (2 independent evaluators + 1 proxy)."""
    _, artifacts = train_and_evaluate_labe_classifier()

    lexicon_eval = ExactLexiconEvaluator()
    ngram_eval = SparseNgramEnsembleEvaluator(artifacts)
    proxy_eval = LengthAdjustedNgramProxy(ngram_eval)

    panel = {
        "exact_lexicon": lexicon_eval,
        "sparse_ngram_ensemble": ngram_eval,
        "length_adjusted_ngram_proxy": proxy_eval,
    }

    return {
        "status": "EVALUATOR_PANEL_INITIALIZED",
        "evaluators_count": len(panel),
        "independent_evaluators_count": sum(1 for e in panel.values() if e.is_independent),
        "panel": panel,
    }
