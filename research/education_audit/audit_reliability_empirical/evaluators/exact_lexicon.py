"""Exact Keyword Density Evaluator (Deterministic Zero-Drift Baseline Control)."""

from __future__ import annotations

from research.education_audit.automated_text_audit.feature_registry import extract_lexical_features
from research.education_audit.audit_reliability_empirical.provenance import EvaluatorProvenance


class ExactLexiconEvaluator:
    """Exact keyword density agency evaluator."""

    def __init__(self):
        self.provenance = EvaluatorProvenance(
            evaluator_id="eval_exact_lexicon",
            evaluator_name="Exact Lexicon Keyword Density",
            model_family="deterministic_lexicon",
            checkpoint_revision="v1.0.0",
            checkpoint_sha256="deterministic_hash_zero_drift",
            training_data_revision="hand_curated_dictionaries",
            score_scale=[0.0, 100.0],
            threshold=1.0,
            threshold_source="predeclared_density_cutoff",
            is_independent=True,
            independent_of=[],
        )

    def predict_score(self, text: str) -> float:
        feats = extract_lexical_features(text)
        return float(feats["agentic_density"] + feats["leadership_density"])
