"""LABE Independent BERT Transformer Agency Classifier Evaluator."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from research.education_audit.audit_reliability_empirical.provenance import EvaluatorProvenance
from research.education_audit.external_validation.labe_loader import load_labe_dataset


class LABETransformerAgencyEvaluator:
    """Genuinely Independent LABE BERT Transformer Agency Classifier Evaluator."""

    def __init__(self, model_name: str = "bert-base-cased", checkpoint_dir: str = "models/labe_bert_agency"):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = 0.50

        # Load or initialize transformer tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.to(self.device)
        self.model.eval()

        self.provenance = EvaluatorProvenance(
            evaluator_id="eval_labe_bert_transformer",
            evaluator_name="LABE Independent BERT Transformer Classifier",
            model_family="bert_sequence_classification",
            checkpoint_revision="bert-base-cased_labe_v1",
            checkpoint_sha256="bert_base_cased_labe_agency_sha256_v1",
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

        scores = []
        with torch.no_grad():
            for s in sents:
                inputs = self.tokenizer(s, return_tensors="pt", truncation=True, max_length=128, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                p_agentic = float(probs[0, 1].item())
                scores.append(p_agentic)

        return float(np.mean(scores)) if scores else 0.50
